"""Create TF-stratified splits while keeping reciprocal pairs together."""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import numpy as np
import pandas as pd
from dfgnn_utils import network_density

def pair_key(u, v):
    u, v = (int(u), int(v))
    return (u, v) if u <= v else (v, u)

def _read_inputs(label_file, Gene_file, TF_file):
    label = pd.read_csv(label_file, index_col=0)
    gene_set = pd.read_csv(Gene_file, index_col=0)['index'].values.astype(int)
    tf_set = pd.read_csv(TF_file, index_col=0)['index'].values.astype(int)
    return (label, gene_set, tf_set)

def _split_group_keys_by_tf(group_to_edges, specific, ratio=0.67, p_val=0.5):
    tf_to_keys = defaultdict(set)
    for key, directed_edges in group_to_edges.items():
        for tf, target in directed_edges:
            tf_to_keys[tf].add(key)
    assignments = {}
    for tf in sorted(tf_to_keys):
        keys = [k for k in sorted(tf_to_keys[tf]) if k not in assignments]
        np.random.shuffle(keys)
        n = len(keys)
        if n == 0:
            continue
        if n == 1:
            train_keys = keys if np.random.uniform() <= p_val else []
            val_keys = []
            test_keys = keys if not train_keys else []
        elif n == 2:
            train_keys, val_keys, test_keys = (keys[:1], [], keys[1:])
        elif specific:
            a = int(n * ratio)
            b = int(n * (ratio + 0.1))
            train_keys, val_keys, test_keys = (keys[:a], keys[a:b], keys[b:])
        else:
            a = n * 2 // 3
            b = a // 5
            train_keys, val_keys, test_keys = (keys[b:a], keys[:b], keys[a:])
        for split, split_keys in [('train', train_keys), ('val', val_keys), ('test', test_keys)]:
            for key in split_keys:
                assignments[key] = split
    splits = {'train': [], 'val': [], 'test': []}
    for key, edges in group_to_edges.items():
        splits[assignments[key]].extend(edges)
    return splits

def _positive_splits(label, specific, ratio, p_val):
    groups = defaultdict(list)
    seen = set()
    for tf, target in label[['TF', 'Target']].itertuples(index=False, name=None):
        edge = (int(tf), int(target))
        if edge in seen:
            continue
        seen.add(edge)
        groups[pair_key(*edge)].append(edge)
    splits = _split_group_keys_by_tf(groups, specific, ratio, p_val)
    return (splits, set(groups))

def _specific_negative_splits(gene_set, tf_set, known_positive_keys, ratio, p_val):
    groups = defaultdict(list)
    for tf in tf_set:
        for target in gene_set:
            tf, target = (int(tf), int(target))
            key = pair_key(tf, target)
            if tf != target and key not in known_positive_keys:
                groups[key].append((tf, target))
    return _split_group_keys_by_tf(groups, True, ratio, p_val)

def _other_negative_splits(gene_set, tf_set, known_positive_keys, positive_splits, density):
    tf_set = np.asarray(tf_set, dtype=int)
    gene_set = np.asarray(gene_set, dtype=int)
    tf_pool = set((int(x) for x in tf_set))
    neg_splits = {'train': [], 'val': [], 'test': []}
    used_keys = set(known_positive_keys)
    pools = {}
    cursors = {}
    for tf in tf_pool:
        pool = [int(g) for g in gene_set if int(g) != tf and pair_key(tf, int(g)) not in known_positive_keys]
        np.random.shuffle(pool)
        pools[tf] = pool
        cursors[tf] = 0
    for split in ('train', 'val'):
        for tf, _ in positive_splits[split]:
            tf = int(tf)
            if tf not in tf_pool:
                raise ValueError(f'Positive source {tf} is absent from TF.csv.')
            pool = pools[tf]
            index = cursors[tf]
            while index < len(pool) and pair_key(tf, pool[index]) in used_keys:
                index += 1
            if index >= len(pool):
                raise ValueError(f'Insufficient negative pairs for TF {tf} in {split}.')
            target = pool[index]
            cursors[tf] = index + 1
            used_keys.add(pair_key(tf, target))
            neg_splits[split].append((tf, target))
    test_pos_count = len(positive_splits['test'])
    if density <= 0 or density > 1:
        raise ValueError(f'Invalid reference-network density: {density}')
    test_neg_count = int(test_pos_count // density - test_pos_count)
    candidates, candidate_keys = ([], set())
    for tf in tf_set:
        for target in gene_set:
            tf, target = (int(tf), int(target))
            key = pair_key(tf, target)
            if tf != target and key not in used_keys and (key not in candidate_keys):
                candidates.append((tf, target))
                candidate_keys.add(key)
    if test_neg_count > len(candidates):
        raise ValueError(f'Requested {test_neg_count} test negatives, but only {len(candidates)} pairs remain.')
    chosen = np.random.choice(len(candidates), test_neg_count, replace=False)
    neg_splits['test'] = [candidates[int(i)] for i in chosen]
    return neg_splits

def _save_splits(positive_splits, negative_splits, train_set_file, val_set_file, test_set_file):
    paths = {'train': train_set_file, 'val': val_set_file, 'test': test_set_file}
    owned_keys = {}
    counts = {}
    output_frames = {}
    for split in ('train', 'val', 'test'):
        pos = positive_splits[split]
        neg = negative_splits[split]
        rows = [(u, v, 1) for u, v in pos] + [(u, v, 0) for u, v in neg]
        for u, v, label in rows:
            key = pair_key(u, v)
            if key in owned_keys:
                old_split, old_label = owned_keys[key]
                if old_split != split or old_label != label:
                    raise AssertionError(f'Pair {key} crosses splits or has conflicting labels: {old_split}/{old_label} versus {split}/{label}.')
            else:
                owned_keys[key] = (split, label)
        output_frames[split] = pd.DataFrame(rows, columns=['TF', 'Target', 'Label'])
        counts[split] = (len(pos), len(neg))
    for split in ('train', 'val', 'test'):
        os.makedirs(os.path.dirname(os.path.abspath(paths[split])), exist_ok=True)
        output_frames[split].to_csv(paths[split])
    print('Split counts (positive, negative):', counts)

def split_specific(label_file, Gene_file, TF_file, train_set_file, val_set_file, test_set_file, ratio, p_val):
    label, gene_set, tf_set = _read_inputs(label_file, Gene_file, TF_file)
    pos, positive_keys = _positive_splits(label, True, ratio, p_val)
    neg = _specific_negative_splits(gene_set, tf_set, positive_keys, ratio, p_val)
    _save_splits(pos, neg, train_set_file, val_set_file, test_set_file)
    return 1

def split_other(label_file, Gene_file, TF_file, train_set_file, val_set_file, test_set_file, density, p_val):
    label, gene_set, tf_set = _read_inputs(label_file, Gene_file, TF_file)
    pos, positive_keys = _positive_splits(label, False, 0.67, p_val)
    neg = _other_negative_splits(gene_set, tf_set, positive_keys, pos, density)
    _save_splits(pos, neg, train_set_file, val_set_file, test_set_file)
    return 1

def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=root / 'results' / 'splits')
    parser.add_argument('--datasets', nargs='+', default=['hESC', 'hHEP', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L'])
    parser.add_argument('--network-types', nargs='+', choices=['STRING', 'Non-Specific', 'Specific', 'Lofgof'], default=['STRING', 'Non-Specific', 'Specific', 'Lofgof'])
    parser.add_argument('--gene-counts', nargs='+', type=int, default=[500, 1000])
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(1, 11)))
    parser.add_argument('--ratio', type=float, default=0.67, help='Training fraction for Specific networks.')
    parser.add_argument('--single-edge-probability', type=float, default=0.5)
    args = parser.parse_args()
    for seed in args.seeds:
        np.random.seed(seed)
        for network in args.network_types:
            for dataset in args.datasets:
                if network == 'Lofgof' and dataset != 'mESC':
                    continue
                for count in args.gene_counts:
                    data = args.data_root / f'{network} Dataset' / dataset / f'TFs+{count}'
                    output = args.output_dir / f'seed{seed}' / network / f'{dataset} {count}'
                    files = (data / 'Label.csv', data / 'Target.csv', data / 'TF.csv',
                             output / 'Train_set.csv', output / 'Validation_set.csv', output / 'Test_set.csv')
                    print(f'{network}: {dataset}, TFs+{count}, seed {seed}', flush=True)
                    if network == 'Specific':
                        split_specific(*files, args.ratio, args.single_edge_probability)
                    else:
                        split_other(*files, network_density(dataset, count, network), args.single_edge_probability)


if __name__ == '__main__':
    main()
