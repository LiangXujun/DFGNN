"""Train DFGNN on repeated, reciprocal-pair-disjoint benchmark splits."""

import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import APPNP, GCNConv

from dfgnn_utils import EdgeDataset, evaluate, expression_features


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def knn_graph(similarity, k):
    graph = np.zeros(similarity.shape)
    for i in range(len(similarity)):
        for j in np.argsort(similarity[i])[::-1][:k + 1]:
            if i != j:
                graph[i, j] = similarity[i, j]
                graph[j, i] = similarity[j, i]
    return graph


def consistency_loss(first, second):
    first = F.normalize(first - first.mean(0, keepdim=True), p=2, dim=1)
    second = F.normalize(second - second.mean(0, keepdim=True), p=2, dim=1)
    return ((first @ first.T - second @ second.T) ** 2).mean()


class SelfGate(nn.Module):
    """Preserve the ReLU gate used in the supplied implementation."""

    def __init__(self, dimension):
        super().__init__()
        self.lin = nn.Linear(dimension, dimension)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        return x * F.relu(self.lin(x))


class GraphEncoder(nn.Module):
    def __init__(self, dimension, dropout):
        super().__init__()
        self.gc2 = APPNP(K=2, alpha=0.3, add_self_loops=True)
        self.gc1 = GCNConv(dimension, dimension, add_self_loops=True, bias=True)
        self.dropout = dropout

    def forward(self, x, edges):
        x = F.dropout(F.relu(self.gc1(x, edges)), self.dropout, training=self.training)
        return self.gc2(x, edges)


class DFGNN(nn.Module):
    def __init__(self, n_nodes, dimension=64, dropout=0.0):
        super().__init__()
        self.emb_t = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_nodes, dimension)))
        self.sgu = SelfGate(dimension)
        self.gcn_f = GraphEncoder(dimension, dropout)
        self.gcn_t = GraphEncoder(dimension, dropout)
        self.gcn_c = GraphEncoder(dimension, dropout)

    def forward(self, prior_edges, expression_edges, pairs):
        expression_embedding = self.sgu(self.emb_t)
        prior = self.gcn_t(self.emb_t, prior_edges) + self.emb_t
        expression = self.gcn_f(expression_embedding, expression_edges) + expression_embedding
        common_prior = self.gcn_c(prior, prior_edges)
        common_expression = self.gcn_c(expression, expression_edges)
        embedding = torch.cat((prior, expression, common_prior + common_expression), dim=1)
        pairs = torch.as_tensor(pairs, dtype=torch.long, device=embedding.device)
        scores = (embedding[pairs[:, 0]] * embedding[pairs[:, 1]]).sum(1)
        return scores, prior, common_prior, common_expression, expression


def fit_dataset(data_dir, split_dir, output_dir, args, weight_decay):
    features = expression_features(pd.read_csv(data_dir / 'BL--ExpressionData.csv', index_col=0))
    train = pd.read_csv(split_dir / 'Train_set.csv', index_col=0).to_numpy()
    validation = pd.read_csv(split_dir / 'Validation_set.csv', index_col=0).to_numpy()
    test = pd.read_csv(split_dir / 'Test_set.csv', index_col=0).to_numpy()
    dataset = EdgeDataset(train, len(features))
    prior_edges = torch.as_tensor(np.array(dataset.adjacency().nonzero()), dtype=torch.long, device=args.device)
    similarity = 1 - squareform(pdist(features, 'cosine'))
    expression_edges = torch.as_tensor(np.array(np.where(knn_graph(similarity, args.neighbors))),
                                       dtype=torch.long, device=args.device)
    model = DFGNN(len(features), args.dimension).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    batches = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    best_auc, best_epoch, non_improving = 0.0, 0, 0
    for epoch in range(args.epochs):
        total_loss = 0.0
        for pairs, labels in batches:
            model.train()
            optimizer.zero_grad()
            scores, _, common_prior, common_expression, _ = model(prior_edges, expression_edges, pairs)
            loss = F.mse_loss(scores, labels.to(args.device))
            loss = loss + args.consistency_weight * consistency_loss(common_prior, common_expression)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        model.eval()
        with torch.no_grad():
            scores = model(prior_edges, expression_edges, validation[:, :2])[0].sigmoid()
        val_auc, val_ap, _ = evaluate(validation[:, -1], scores)
        print(f'Epoch {epoch + 1}: loss={total_loss / len(dataset):.6f}, '
              f'validation AUROC={val_auc:.6f}, AUPRC={val_ap:.6f}', flush=True)
        if val_auc >= best_auc:
            best_auc, best_epoch = val_auc, epoch + 1
        else:
            non_improving += 1
            if non_improving > args.patience:
                break
    model.eval()
    with torch.no_grad():
        scores = model(prior_edges, expression_edges, test[:, :2])[0].sigmoid()
    auc, ap, normalized_ap = evaluate(test[:, -1], scores)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'n_nodes': len(features),
                'dimension': args.dimension}, output_dir / 'dfgnn.pt')
    predictions = pd.DataFrame(test, columns=['TF', 'Target', 'Label'])
    predictions['score'] = scores.cpu().numpy()
    predictions.to_csv(output_dir / 'test_predictions.csv', index=False)
    return {'AUROC': auc, 'AUPRC': ap, 'normalized_AUPRC': normalized_ap,
            'epochs_completed': epoch + 1, 'best_validation_epoch': best_epoch,
            'best_validation_AUROC': best_auc, 'weight_decay': weight_decay}


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--split-root', type=Path, default=root / 'results' / 'splits')
    parser.add_argument('--output-dir', type=Path, default=root / 'results' / 'benchmark')
    parser.add_argument('--datasets', nargs='+', default=['hESC', 'hHEP', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L'])
    parser.add_argument('--network-types', nargs='+', choices=['STRING', 'Non-Specific', 'Specific', 'Lofgof'],
                        default=['STRING', 'Non-Specific', 'Specific', 'Lofgof'])
    parser.add_argument('--gene-counts', nargs='+', type=int, default=[500, 1000])
    parser.add_argument('--split-seeds', nargs='+', type=int, default=list(range(1, 11)))
    parser.add_argument('--seed', type=int, default=8, help='Model/training seed, separate from the split seed.')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--dimension', type=int, default=64)
    parser.add_argument('--neighbors', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--consistency-weight', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    results = []
    for split_seed in args.split_seeds:
        for network in args.network_types:
            for dataset in args.datasets:
                if network == 'Lofgof' and dataset != 'mESC':
                    continue
                for gene_count in args.gene_counts:
                    set_seed(args.seed)
                    decay = args.weight_decay
                    if (network == 'Non-Specific' and dataset != 'mHSC-E') or network == 'Lofgof':
                        decay = 1e-2
                    elif network == 'STRING' and dataset == 'mHSC-L':
                        decay = 5e-3
                    data_dir = args.data_root / f'{network} Dataset' / dataset / f'TFs+{gene_count}'
                    split_dir = args.split_root / f'seed{split_seed}' / network / f'{dataset} {gene_count}'
                    output = args.output_dir / f'seed{split_seed}' / network / f'{dataset}_{gene_count}'
                    print(f'{network}: {dataset}, TFs+{gene_count}, split seed {split_seed}', flush=True)
                    result = fit_dataset(data_dir, split_dir, output, args, decay)
                    results.append({'network': network, 'dataset': dataset, 'gene_count': gene_count,
                                    'split_seed': split_seed, 'model_seed': args.seed, **result})
                    args.output_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(results).to_csv(args.output_dir / 'metrics.csv', index=False)


if __name__ == '__main__':
    main()
