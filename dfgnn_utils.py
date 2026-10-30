"""Expression preprocessing, edge data and benchmark evaluation for DFGNN."""

import numpy as np
import scipy.sparse as sparse
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class EdgeDataset(Dataset):
    def __init__(self, edges, n_genes):
        self.edges = np.asarray(edges)
        self.n_genes = n_genes

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, index):
        row = self.edges[index]
        return row[:2].astype(np.int64), row[-1].astype(np.float32)

    def adjacency(self):
        graph = sparse.dok_matrix((self.n_genes, self.n_genes), dtype=np.float32)
        for source, target, label in self.edges:
            if label == 1:
                source, target = int(source), int(target)
                graph[source, target] = graph[target, source] = 1.0
        return graph


def expression_features(expression):
    return StandardScaler().fit_transform(expression.to_numpy().T).T.astype(np.float32)


def evaluate(labels, scores):
    labels = np.asarray(labels).flatten().astype(int)
    scores = scores.detach().cpu().numpy().flatten()
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)
    return auroc, auprc, auprc / labels.mean()


_DENSITIES = {
    'STRING': {
        'hESC': (0.024, 0.021), 'hHEP': (0.028, 0.024), 'mDC': (0.038, 0.032),
        'mESC': (0.024, 0.021), 'mHSC-E': (0.029, 0.027),
        'mHSC-GM': (0.040, 0.037), 'mHSC-L': (0.048, 0.045),
    },
    'Non-Specific': {
        'hESC': (0.016, 0.014), 'hHEP': (0.015, 0.013), 'mDC': (0.019, 0.016),
        'mESC': (0.015, 0.013), 'mHSC-E': (0.022, 0.020),
        'mHSC-GM': (0.030, 0.029), 'mHSC-L': (0.048, 0.043),
    },
    'Lofgof': {'mESC': (0.158, 0.154)},
}


def network_density(dataset, gene_count, network):
    return _DENSITIES[network][dataset][(500, 1000).index(gene_count)]
