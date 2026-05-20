#%%
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
import random
import argparse
from scipy.spatial.distance import squareform, pdist

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.nn.conv import GCNConv, APPNP
from sklearn import metrics

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.chdir(os.path.dirname(os.path.realpath(__file__)))
from utils import scRNADataset, load_data, Evaluation

script_path = __file__

#%%
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def knn_graph(sim_mat, k):
    graph = np.zeros(sim_mat.shape)
    for i in range(sim_mat.shape[0]):
        idx = np.argsort(sim_mat[i,])[::-1]
        for j in idx[0:k+1]:
            if j == i:
                continue
            else:
                graph[i,j] = sim_mat[i,j]
                graph[j,i] = sim_mat[j,i]
    # graph = graph + np.eye(graph.shape[0])
    return graph

def common_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cost = torch.mean((cov1 - cov2)**2)
    return cost

#%%
class SGU(nn.Module):
    def __init__(self, emb_dim):
        super(SGU, self).__init__()
        
        self.lin = nn.Linear(emb_dim, emb_dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def forward(self, x):
        return x*F.relu(self.lin(x))
                

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(GCN, self).__init__()
        
        # self.gc1 = GCNConv(in_dim, hid_dim, add_self_loops=True, bias = True)
        # self.gc2 = GCNConv(hid_dim, out_dim, add_self_loops=True, bias = True)
        self.gc2 = APPNP(K = 2, alpha = 0.3, add_self_loops=True)
        self.gc1 = GCNConv(hid_dim, out_dim, add_self_loops=True, bias = True)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gc2(x, edge_index)
        return x


class Net(nn.Module):
    def __init__(self, n_node, emb_dim, nhid1, nhid2, dropout):
        super(Net, self).__init__()
        
        self.emb_t = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_node, emb_dim)))
        self.sgu = SGU(emb_dim)
        self.gcn_f = GCN(emb_dim, nhid1, nhid2, dropout)
        self.gcn_t = GCN(emb_dim, nhid1, nhid2, dropout)
        self.gcn_c = GCN(emb_dim, nhid1, nhid2, dropout)
        
        self.dropout = dropout
    
    def forward(self, sadj, fadj, train_x):
        self.emb_f = self.sgu(self.emb_t)
        
        emb_t = self.gcn_t(self.emb_t, sadj) + self.emb_t
        emb_f = self.gcn_f(self.emb_f, fadj) + self.emb_f
        
        com_t = self.gcn_c(emb_t, sadj)
        com_f = self.gcn_c(emb_f, fadj)
        com = (com_t + com_f)/1
        
        emb = torch.cat([emb_t, emb_f, com], dim = 1)
        
        tf_emb = emb[train_x[:,0],]
        target_emb = emb[train_x[:,1],]
        output = torch.sum(tf_emb*target_emb, 1)
        return output.squeeze(), emb_t, com_t, com_f, emb_f
    
#%%
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default= 200, help='Number of epoch.')
parser.add_argument('--n_layer', type=int, default= 3, help='Number of GCN layer.')
parser.add_argument('--k_nei', type=int, default= 10, help='Number of knn graph neighbors.')
# parser.add_argument('--num_head', type=list, default=[3,3], help='Number of head attentions.')
# parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--emb_dim', type=int, default= 64, help='The dimension of embedding layer')
parser.add_argument('--hid_dim', type=int, default= 64, help='The dimension of hidden layer')
parser.add_argument('--lam', type=float, default= 0.5, help='The parameter for consistant constrain.')
# parser.add_argument('--output_dim', type=int, default=16, help='The dimension of latent layer')
parser.add_argument('--patience', type=int, default=50, help='The patience')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
parser.add_argument('--batch_size', type=int, default=512, help='The size of each batch')
parser.add_argument('--loop', type=bool, default=False, help='whether to add self-loop in adjacent matrix')
parser.add_argument('--seed', type=int, default=8, help='Random seed')
parser.add_argument('--flag', type=bool, default=False, help='the identifier whether to conduct causal inference')

#%%
dataname_list = ['hESC', 'hHEP', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']
gene_num_list = [500, 1000]
net_type_list = ['STRING', 'Non-Specific', 'Specific', 'Lofgof']

for seed in range(1, 6):
    for net_type in net_type_list:
        for dataname in dataname_list:
            if net_type == 'Lofgof' and dataname != 'mESC':
                continue
            else:
                for gene_num in gene_num_list:
                    args = parser.parse_args()
                    setup_seed(args.seed)
                    
                    if net_type == 'Non-Specific':
                        args.weight_decay = 1e-2
                    
                    if net_type == 'Lofgof':
                        args.weight_decay = 1e-2
                        
                    data_path = './GRN/{} Dataset/{}/TFs+{}/'.format(net_type, dataname, str(gene_num))
                    exp_file = data_path + 'BL--ExpressionData.csv'
                    tf_file = data_path + 'TF.csv'
                    target_file = data_path + '/Target.csv'
                    
                    train_path = './GRN/Train_validation_test_rep10/seed{}/{}/{} {}/'.format(seed, net_type, dataname, str(gene_num))
                    train_file = train_path + 'Train_set.csv'
                    val_file = train_path + 'Validation_set.csv'
                    test_file = train_path + 'Test_set.csv'
                    
                    data_input = pd.read_csv(exp_file,index_col=0)
                    loader = load_data(data_input)
                    feature = loader.exp_data()
                    tf = pd.read_csv(tf_file,index_col=0)['index'].values.astype(np.int64)
                    target = pd.read_csv(target_file,index_col=0)['index'].values.astype(np.int64)
                    
                    train_data = pd.read_csv(train_file, index_col=0).values
                    validation_data = pd.read_csv(val_file, index_col=0).values
                    test_data = pd.read_csv(test_file, index_col=0).values
                    
                    train_load = scRNADataset(train_data, feature.shape[0], flag=args.flag)
                    adj = train_load.Adj_Generate(tf, loop = args.loop)
                    n_nodes = adj.shape[0]
                    
                    sim_mat = 1 - squareform(pdist(feature, 'cosine'))
                    # sim_mat = np.corrcoef(feature)
                    fadj = knn_graph(sim_mat, args.k_nei)
    
                    edge_index = torch.LongTensor(np.array(adj.nonzero())).to(device)
                    knn_edge_index =  torch.LongTensor(np.array(np.where(fadj))).to(device)
                    feat = torch.FloatTensor(feature).to(device)
    
                    best_epoch = 0
                    best_val_AUC = 0.0
                    patience = 0
                    model = Net(feat.shape[0], args.emb_dim, args.hid_dim, args.hid_dim, dropout = 0.).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
                    for epoch in range(args.epochs):
                        running_loss = 0.0
                        for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
                            model.train()
                            optimizer.zero_grad()
                            train_y = train_y.to(device)
                            pred, emb_t, com_t, com_f, emb_f = model(edge_index, knn_edge_index, train_x)
                            loss_target = F.mse_loss(pred, train_y)
                            loss_com = common_loss(com_t, com_f)
                            loss = loss_target + args.lam*loss_com
                            loss.backward()
                            optimizer.step()
                        try:
                            with torch.no_grad():
                                model.eval()
                                pred, _, _, _, _ = model(edge_index, knn_edge_index, validation_data)
                                pred = torch.sigmoid(pred)
                            val_AUC, val_AUPR, AUPR_norm = Evaluation(y_pred=pred, y_true=validation_data[:, -1],flag=args.flag)
                            print('Epoch:{}'.format(epoch + 1), 'train loss:{}'.format(running_loss),'AUC:{:.3F}'.format(val_AUC),'AUPR:{:.3F}'.format(val_AUPR))
                            
                            if best_val_AUC <= val_AUC:
                                best_val_AUC = val_AUC
                                best_epoch = epoch
                            else:
                                patience += 1
                                if patience > args.patience:
                                    print('Early Stopping')
                                    break
                        except:
                            if epoch > 100:
                                break
                            else:
                                continue
                    model.eval()
                    with torch.no_grad():
                        pred ,_, _, _, _ = model(edge_index, knn_edge_index, test_data)
                        pred = torch.sigmoid(pred)
                    test_AUC, test_AUPR, AUPR_norm = Evaluation(y_pred=pred, y_true=test_data[:, -1],flag=args.flag)
                    print(net_type, dataname+' '+str(gene_num), 'AUC:{:.3F}'.format(test_AUC),'AUPR:{:.3F}'.format(test_AUPR), 'best epoch:{:d}'.format(best_epoch))
                    
                    file_out = 'F:/data/explore/GRN/result/my_AMGCN_alpha_result_v1.2_rep10.txt'
                    if not os.path.exists(file_out):
                        fout = open(file_out, 'w')
                        print(script_path, file = fout)
                    else:
                        fout = open(file_out, 'a')
                    print(args, file = fout)
                    print(net_type, dataname+' '+str(gene_num), 'AUC:{:.3F}'.format(test_AUC),'AUPR:{:.3F}'.format(test_AUPR), file = fout)
                    fout.flush()
                    fout.close()
