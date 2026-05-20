"""
The code is derived from and modified based on that of GENELink, and the benchmark data is also sourced from GENELink.
"""
import pandas as pd
import numpy as np
import os
from collections import Counter
from sklearn.model_selection import train_test_split
import argparse
# from tqdm import tqdm

os.chdir(os.path.dirname(os.path.realpath(__file__)))
from utils import Network_Statistic

#%%
def Hard_Negative_Specific_train_val_test(label_file, Gene_file, TF_file, train_set_file, val_set_file, test_set_file, ratio, p_val):
    if not os.path.exists(os.path.dirname(train_set_file)):
        os.makedirs(os.path.dirname(train_set_file))
    
    label = pd.read_csv(label_file, index_col=0)
    gene_set = pd.read_csv(Gene_file, index_col=0)['index'].values
    tf_set = pd.read_csv(TF_file, index_col=0)['index'].values
    
    tf = label['TF'].values
    tf_list = np.unique(tf)
    
    pos_dict = {}
    for i in tf_list:
        pos_dict[i] = []
    
    for i,j in label.values:
        pos_dict[i].append(j)
    
    neg_dict = {}
    for i in tf_set:
        neg_dict[i] = []
    
    for i in tf_set:
        if i in pos_dict.keys():
            pos_item = pos_dict[i]
            pos_item.append(i)
            neg_item = np.setdiff1d(gene_set, pos_item)
            neg_dict[i].extend(neg_item)
            pos_dict[i] = np.setdiff1d(pos_dict[i], i)
        else:
            neg_item = np.setdiff1d(gene_set, i)
            neg_dict[i].extend(neg_item)
    
    train_pos = {}
    val_pos = {}
    test_pos = {}
    for k in pos_dict.keys():
        if len(pos_dict[k]) == 1:
            p = np.random.uniform(0, 1)
            if p <= p_val:
                train_pos[k] = pos_dict[k]
            else:
                test_pos[k] = pos_dict[k]
        elif len(pos_dict[k]) == 2:
            np.random.shuffle(pos_dict[k])
            train_pos[k] = [pos_dict[k][0]]
            test_pos[k] = [pos_dict[k][1]]
        else:
            np.random.shuffle(pos_dict[k])
            train_pos[k] = pos_dict[k][:int(len(pos_dict[k])*ratio)]
            val_pos[k] = pos_dict[k][int(len(pos_dict[k])*ratio):int(len(pos_dict[k])*(ratio)+0.1)]
            test_pos[k] = pos_dict[k][int(len(pos_dict[k])*(ratio+0.1)):]
    
    train_neg = {}
    val_neg = {}
    test_neg = {}
    for k in pos_dict.keys():
        neg_num = len(neg_dict[k])
        np.random.shuffle(neg_dict[k])
        train_neg[k] = neg_dict[k][:int(neg_num*ratio)]
        val_neg[k] = neg_dict[k][int(neg_num*ratio):int(neg_num*(0.1+ratio))]
        test_neg[k] = neg_dict[k][int(neg_num*(0.1+ratio)):]
    
    train_pos_set = []
    for k in train_pos.keys():
        for val in train_pos[k]:
            train_pos_set.append([k, val])
    
    train_neg_set = []
    for k in train_neg.keys():
        for val in train_neg[k]:
            train_neg_set.append([k, val])
        
    train_set = train_pos_set + train_neg_set
    train_label = [1 for _ in range(len(train_pos_set))] + [0 for _ in range(len(train_neg_set))]
    
    train_sample = np.array(train_set)
    train = pd.DataFrame()
    train['TF'] = train_sample[:,0]
    train['target'] = train_sample[:,1]
    train['Lable'] = train_label
    train.to_csv(train_set_file)
    
    val_pos_set = []
    for k in val_pos.keys():
        for val in val_pos[k]:
            val_pos_set.append([k, val])
        
    val_neg_set = []
    for k in val_neg.keys():
        for val in val_neg[k]:
            val_neg_set.append([k, val])
    
    val_set = val_pos_set + val_neg_set
    val_label = [1 for _ in range(len(val_pos_set))] + [0 for _ in range(len(val_neg_set))]
    
    val_sample = np.array(val_set)
    val = pd.DataFrame()
    val['TF'] = val_sample[:,0]
    val['Target'] = val_sample[:,1]
    val['Label'] = val_label
    val.to_csv(val_set_file)
    
    test_pos_set = []
    for k in test_pos.keys():
        for val in test_pos[k]:
            test_pos_set.append([k, val])
    
    test_neg_set = []
    for k in test_neg.keys():
        for val in test_neg[k]:
            test_neg_set.append([k, val])
    
    test_set = test_pos_set + test_neg_set
    test_label = [1 for _ in range(len(test_pos_set))] + [0 for _ in range(len(test_neg_set))]
    
    test_sample = np.array(test_set)
    test = pd.DataFrame()
    test['TF'] = test_sample[:,0]
    test['Target'] = test_sample[:,1]
    test['Label'] = test_label
    test.to_csv(test_set_file)
    
    return 1


def train_val_test_set(label_file, Gene_file, TF_file, train_set_file, val_set_file, test_set_file, density, p_val):
    if not os.path.exists(os.path.dirname(train_set_file)):
        os.makedirs(os.path.dirname(train_set_file))
    
    gene_set = pd.read_csv(Gene_file, index_col = 0)['index'].values
    tf_set = pd.read_csv(TF_file, index_col = 0)['index'].values
    
    label = pd.read_csv(label_file, index_col = 0)
    tf = label['TF'].values
    
    tf_list = np.unique(tf)
    pos_dict = {}
    for i in tf_list:
        pos_dict[i] = []
    
    for i, j in label.values:
        pos_dict[i].append(j)
    
    train_pos = {}
    val_pos = {}
    test_pos = {}
    
    for k in pos_dict.keys():
        if len(pos_dict[k]) <= 1:
            p = np.random.uniform(0, 1)
            if p <= p_val:
                train_pos[k] = pos_dict[k]
            else:
                test_pos[k] = pos_dict[k]
        elif len(pos_dict[k]) == 2:
            train_pos[k] = [pos_dict[k][0]]
            test_pos[k] = [pos_dict[k][1]]
        else:
            np.random.shuffle(pos_dict[k])
            train_pos[k] = pos_dict[k][:len(pos_dict[k])*2//3]
            test_pos[k] = pos_dict[k][len(pos_dict[k])*2//3:]
            val_pos[k] = train_pos[k][:len(train_pos[k])//5]
            train_pos[k] = train_pos[k][len(train_pos[k])//5:]
        
    train_neg = {}
    for k in train_pos.keys():
        train_neg[k] = []
        for i in range(len(train_pos[k])):
            neg = np.random.choice(gene_set)
            while neg == k or neg in pos_dict[k] or neg in train_neg[k]:
                neg = np.random.choice(gene_set)
            train_neg[k].append(neg)
    
    train_pos_set = []
    train_neg_set = []
    for k in train_pos.keys():
        for j in train_pos[k]:
            train_pos_set.append([k,j])
    train_pos_label = [1 for _ in range(len(train_pos_set))]
    
    for k in train_neg.keys():
        for j in train_neg[k]:
            train_neg_set.append([k,j])
    train_neg_label = [0 for _ in range(len(train_neg_set))]
    
    train_set = train_pos_set + train_neg_set
    train_label = train_pos_label + train_neg_label
    
    train_sample = train_set.copy()
    for i, v in enumerate(train_sample):
        v.append(train_label[i])
    train = pd.DataFrame(train_sample, columns = ['TF', 'Target', 'Label'])
    train.to_csv(train_set_file)
    
    val_pos_set = []
    for k in val_pos.keys():
        for j in val_pos[k]:
            val_pos_set.append([k, j])
    val_pos_label = [1 for _ in range(len(val_pos_set))]
    
    val_neg = {}
    for k in val_pos.keys():
        val_neg[k] = []
        for i in range(len(val_pos[k])):
            neg = np.random.choice(gene_set)
            while neg == k or neg in pos_dict[k] or neg in train_neg[k] or neg in val_neg[k]:
                neg = np.random.choice(gene_set)
            val_neg[k].append(neg)
    
    val_neg_set = []
    for k in val_neg.keys():
        for j in val_neg[k]:
            val_neg_set.append([k,j])
    val_neg_label = [0 for _ in range(len(val_neg_set))]
    val_set = val_pos_set + val_neg_set
    val_label = val_pos_label + val_neg_label
    
    val_sample= val_set.copy()
    for i, v in enumerate(val_sample):
        v.append(val_label[i])
    val = pd.DataFrame(val_sample, columns = ['TF', 'Target', 'Label'])
    val.to_csv(val_set_file)
    
    test_pos_set = []
    for k in test_pos.keys():
        for j in test_pos[k]:
            test_pos_set.append([k, j])
    
    count = 0
    for k in test_pos.keys():
        count += len(test_pos[k])
    test_neg_num = int(count//density - count)
    
    test_neg_set = []
    for i in range(test_neg_num):
        t1 = np.random.choice(tf_set)
        t2 = np.random.choice(gene_set)
        while t1 == t2 or [t1, t2] in train_set or [t1, t2] in test_pos_set or [t1, t2] in val_set or [t1,t2] in test_neg_set:
            t2 = np.random.choice(gene_set)
        test_neg_set.append([t1,t2])
    
    test_pos_label = [1 for _ in range(len(test_pos_set))]
    test_neg_label = [0 for _ in range(len(test_neg_set))]
    
    test_set = test_pos_set + test_neg_set
    test_label = test_pos_label + test_neg_label
    for i, v in enumerate(test_set):
        v.append(test_label[i])
    
    test_sample = pd.DataFrame(test_set, columns=['TF', 'Target', 'Label'])
    test_sample.to_csv(test_set_file)
    
    return 1

#%%
parser = argparse.ArgumentParser()
parser.add_argument('--ratio', type=float, default=0.67, help='the ratio of the training set')
parser.add_argument('--p_val', type=float, default=0.5, help='the position of the target with degree equaling to one')
args = parser.parse_args()


#%%
dataname_list = ['hESC', 'hHEP', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']
gene_num_list = [500, 1000]
net_type_list = ['STRING', 'Non-Specific', 'Specific', 'Lofgof']
seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for s in seed_list:
    np.random.seed(s)
    for net_type in net_type_list:
        for data_type in dataname_list:
            for num in gene_num_list:
                if net_type == 'Lofgof' and data_type != 'mESC':
                    continue
                density = Network_Statistic(data_type = data_type, net_scale= num, net_type = net_type)
    
                path_r = './GENELink-main/Dataset/Benchmark Dataset'
                TF_file = path_r + '/' + net_type + ' Dataset/' + data_type + '/TFs+' + str(num) + '/TF.csv'
                Gene_file = path_r + '/' + net_type + ' Dataset/' + data_type + '/TFs+' + str(num) + '/Target.csv'
                label_file = path_r + '/' + net_type + ' Dataset/' + data_type + '/TFs+' + str(num) + '/Label.csv'
                
                
                path_w = './GRN/Train_validation_test_rep10/' + f'seed{s}' + '/'
                train_set_file = path_w  + net_type + '/' + data_type + ' ' + str(num) + '/Train_set.csv'
                test_set_file = path_w  + net_type + '/' + data_type + ' ' + str(num)  + '/Test_set.csv'
                val_set_file = path_w + net_type + '/' + data_type + ' ' + str(num) +  '/Validation_set.csv'
            
                print(path_w, train_set_file)
                if net_type == 'Specific':
                    Hard_Negative_Specific_train_val_test(label_file, Gene_file, TF_file, train_set_file, val_set_file, test_set_file, args.ratio, args.p_val)
                else:
                    train_val_test_set(label_file, Gene_file, TF_file, train_set_file, val_set_file, test_set_file, density, args.p_val)

#%%
