"""Shared influence scoring, enrichment and case-study figures."""

import argparse
from pathlib import Path
import re
import textwrap
from urllib.parse import urlencode
from urllib.request import urlopen

import gseapy as gp
from gseapy.parser import DEFAULT_CACHE_PATH
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
LIBRARIES = ['GO_Biological_Process_2023', 'KEGG_2021_Human', 'Reactome_2022']
CASES = {
    'BCM': {'condition': 'MT', 'control': 'PT', 'figure': 'Fig7',
            'title': 'Lung metastasis versus primary tumour',
            'expression_label': r'Gene expression $\log_2$FC (MT $-$ PT)',
            'edge_label': r'Interaction-score change $|s_{MT}-s_{PT}|$',
            'legend_counts': [3, 7, 13]},
    'MM': {'condition': 'no_resp', 'control': 'resp', 'figure': 'Fig8',
           'title': 'Multiple myeloma: non-responders versus responders',
           'expression_label': r'Gene expression $\log_2$FC (NR $-$ R)',
           'edge_label': r'Interaction-score change $|s_{NR}-s_R|$',
           'legend_counts': [3, 6, 12]},
}


def score_networks(case_scores, control_scores, differential_expression):
    if (set(case_scores.index) != set(control_scores.index)
            or set(case_scores.columns) != set(control_scores.columns)):
        raise ValueError('The two prediction matrices must contain the same regulators and targets.')
    regulators = sorted(case_scores.index)
    targets = sorted(case_scores.columns)
    case_scores = case_scores.loc[regulators, targets]
    control_scores = control_scores.loc[regulators, targets]
    target_fc = differential_expression.loc[targets, 'logFC'].abs()
    regulator_fc = differential_expression['logFC'].reindex(regulators).abs()
    case_values = case_scores.to_numpy().copy()
    control_values = control_scores.to_numpy().copy()
    difference = pd.DataFrame(case_values - control_values, index=regulators, columns=targets)
    weighted = pd.DataFrame(np.abs(case_values - control_values) * target_fc.to_numpy(),
                            index=regulators, columns=targets)
    scores = (weighted.sum(axis=1) * regulator_fc).dropna().sort_values(ascending=False)
    return scores, difference, case_scores, control_scores, regulator_fc


def differential_edges(scores, difference, case_scores, control_scores, training_pairs, config):
    edges = difference.loc[scores.index].stack().reset_index()
    edges.columns = ['tf', 'gene', 'value']
    edges = edges.loc[edges['tf'] != edges['gene']].copy()
    edges['abs_score_change'] = edges['value'].abs()
    edges = edges.sort_values(['abs_score_change', 'tf', 'gene'], ascending=[False, True, True]).head(100).copy()
    pairs = list(edges[['tf', 'gene']].itertuples(index=False, name=None))
    edges['prior'] = [int(pair in training_pairs) for pair in pairs]
    edges['score_' + config['condition']] = [case_scores.loc[a, b] for a, b in pairs]
    edges['score_' + config['control']] = [control_scores.loc[a, b] for a, b in pairs]
    return edges


def load_library(name, cache_dir):
    library_file = cache_dir / f'Enrichr.{name}.gmt'
    if not library_file.is_file():
        url = 'https://maayanlab.cloud/Enrichr/geneSetLibrary?' + urlencode(
            {'mode': 'text', 'libraryName': name})
        with urlopen(url, timeout=120) as response:
            lines = response.read().decode('utf-8').splitlines()
        cache_dir.mkdir(parents=True, exist_ok=True)
        with library_file.open('w', encoding='utf-8') as stream:
            for line in lines:
                fields = line.strip().split('\t')
                genes = [gene.split(',')[0] for gene in fields[2:] if gene]
                stream.write('\t'.join([fields[0], name, *genes]) + '\n')
    return gp.get_library(name=str(library_file), organism='Human', max_size=1000000)


def enrich(scores, cache_dir, output):
    background = scores.index.tolist()
    query = background[:40]
    pd.Series(background).to_csv(output / 'enrichment_background.txt', index=False, header=False)
    pd.Series(query).to_csv(output / 'enrichment_query.txt', index=False, header=False)
    libraries, reports = {}, {}
    for name in LIBRARIES:
        libraries[name] = load_library(name, cache_dir)
        result = gp.enrichr(gene_list=query, organism='Human', gene_sets=libraries[name],
                            background=background, cutoff=0.05, outdir=None, no_plot=True).results.copy()
        result['Gene_set'] = name
        reports[name] = result.sort_values('P-value')
        reports[name].to_csv(output / f'{name}.enrichment.tsv', sep='\t', index=False)
    return libraries, reports


def make_figures(edges, differential_expression, regulators, reactome, config, output, formats):
    figure_dir = output / 'figures'
    figure_dir.mkdir(exist_ok=True)
    prefix = config['figure']
    top_pathways = reactome.sort_values(['P-value', 'Term']).head(15).copy()
    top_pathways['-log10(pvalue)'] = -np.log10(top_pathways['P-value'].clip(lower=np.finfo(float).tiny))
    top_pathways['gene_count'] = top_pathways['Genes'].str.split(';').str.len()
    top_pathways.to_csv(figure_dir / f'{prefix}A_Reactome_top15.csv', index=False)

    def draw_enrichment(ax):
        tab = top_pathways.iloc[::-1]
        scatter = ax.scatter(tab['-log10(pvalue)'], np.arange(len(tab)),
                             s=tab['gene_count'] * 24, c=tab['Combined Score'], cmap='RdYlGn',
                             edgecolors='#444444', linewidths=.5, zorder=3)
        labels = [textwrap.fill(re.sub(r'\s+R-HSA-\d+$', '', term), width=43) for term in tab['Term']]
        ax.set_yticks(np.arange(len(tab)), labels, fontsize=9)
        ax.set_ylim(-.8, len(tab) - .2)
        low, high = tab['-log10(pvalue)'].min(), tab['-log10(pvalue)'].max()
        padding = max(.08, (high - low) * .07)
        ax.set_xlim(max(0., low - padding), high + padding)
        ax.set_xlabel(r'$-\log_{10}$(raw $P$-value)', fontsize=11)
        if ax.get_xlim()[0] <= -np.log10(.05) <= ax.get_xlim()[1]:
            ax.axvline(-np.log10(.05), color='#888888', linestyle=':', linewidth=.8)
        ax.grid(axis='x', color='#dddddd', linestyle='--', linewidth=.6)
        ax.spines[['top', 'right']].set_visible(False)
        bar = ax.inset_axes([.54, -.11, .46, .022])
        colorbar = ax.figure.colorbar(scatter, cax=bar, orientation='horizontal')
        colorbar.set_label('Combined score', fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
        counts = config['legend_counts']
        handles = [ax.scatter([], [], s=count * 24, facecolor='white', edgecolor='#444444',
                               linewidth=.5) for count in counts]
        ax.legend(handles, [str(count) for count in counts], title='Contributing TFs',
                  frameon=False, ncol=3, loc='upper left', bbox_to_anchor=(0., -.075),
                  fontsize=9, title_fontsize=9, handletextpad=.5, columnspacing=.9, borderaxespad=0)

    graph = nx.DiGraph()
    for row in edges.itertuples(index=False):
        graph.add_edge(row.tf, row.gene, change=row.abs_score_change, known=bool(row.prior))
    nodes = sorted(graph.nodes())
    node_fc = differential_expression['logFC'].reindex(nodes)
    topology = graph.to_undirected()
    positions = nx.spring_layout(topology, seed=27, k=2.1 / np.sqrt(len(nodes)),
                                 iterations=1000, weight=None)
    distances = {a: {b: min(float(distance), 2.) for b, distance in values.items()}
                 for a, values in nx.all_pairs_shortest_path_length(topology)}
    positions = nx.kamada_kawai_layout(topology, dist=distances, pos=positions, weight=None)
    coordinates = np.array([positions[gene] for gene in nodes])
    for _ in range(300):
        displacement = coordinates[:, None, :] - coordinates[None, :, :]
        distance = np.linalg.norm(displacement, axis=2)
        np.fill_diagonal(distance, np.inf)
        overlap = np.maximum(.115 - distance, 0)
        if overlap.max() < 1e-6:
            break
        coordinates += (displacement * (overlap / distance)[:, :, None]).sum(axis=1) * .3
    positions = dict(zip(nodes, coordinates))
    direction = config['condition'] + '_minus_' + config['control']
    pd.DataFrame({'gene': nodes, 'logFC_' + direction: node_fc.to_numpy(),
                  'is_TF': [gene in regulators for gene in nodes],
                  'x': coordinates[:, 0], 'y': coordinates[:, 1]}).to_csv(
                      figure_dir / f'{prefix}B_nodes.csv', index=False)
    edges.to_csv(figure_dir / f'{prefix}B_edges_top100.csv', index=False)
    node_limit = float(node_fc.abs().max())
    node_norm = TwoSlopeNorm(vmin=-node_limit, vcenter=0, vmax=node_limit)
    edge_norm = Normalize(vmin=edges['abs_score_change'].min(), vmax=edges['abs_score_change'].max())
    edge_cmap = LinearSegmentedColormap.from_list('edge_purples', plt.cm.Purples(np.linspace(.4, .98, 256)))

    def draw_network(ax):
        for known, style in [(False, 'dashed'), (True, 'solid')]:
            selected = [(a, b) for a, b, data in graph.edges(data=True) if data['known'] == known]
            colors = [edge_cmap(edge_norm(graph[a][b]['change'])) for a, b in selected]
            nx.draw_networkx_edges(graph, positions, ax=ax, edgelist=selected, edge_color=colors,
                                    style=style, width=1.0, alpha=.8, arrows=False, node_size=240)
        nx.draw_networkx_nodes(graph, positions, ax=ax, nodelist=nodes, node_size=240,
                                node_color=node_fc.to_numpy(), cmap='RdBu_r', vmin=-node_limit,
                                vmax=node_limit, edgecolors='#777777', linewidths=.4)
        labels = {gene: (point[0], point[1] + .045) for gene, point in positions.items()}
        nx.draw_networkx_labels(graph, labels, ax=ax, font_size=7.5, font_family='DejaVu Sans',
                                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .72, 'pad': .2})
        ax.margins(.055)
        ax.set_axis_off()
        handles = [Line2D([0], [0], color='#555555', linewidth=1.1, linestyle='solid',
                          label='Known training interaction'),
                   Line2D([0], [0], color='#555555', linewidth=1.1, linestyle='dashed',
                          label='Novel predicted relationship')]
        ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .015),
                  ncol=2, frameon=False, fontsize=9, borderaxespad=0)
        bar = ax.inset_axes([.07, -.11, .39, .018])
        colorbar = ax.figure.colorbar(ScalarMappable(norm=node_norm, cmap='RdBu_r'),
                                      cax=bar, orientation='horizontal')
        colorbar.set_label(config['expression_label'], fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
        bar = ax.inset_axes([.57, -.11, .39, .018])
        colorbar = ax.figure.colorbar(ScalarMappable(norm=edge_norm, cmap=edge_cmap),
                                      cax=bar, orientation='horizontal')
        colorbar.set_label(config['edge_label'], fontsize=9)
        colorbar.ax.tick_params(labelsize=8)

    def save(fig, name):
        for extension in formats:
            options = {'dpi': 300, 'facecolor': 'white', 'bbox_inches': 'tight', 'pad_inches': .12}
            if extension == 'jpg':
                options['pil_kwargs'] = {'quality': 98, 'subsampling': 0}
            fig.savefig(figure_dir / f'{name}.{extension}', **options)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 9.2))
    fig.subplots_adjust(left=.53, right=.97, top=.94, bottom=.15)
    draw_enrichment(ax)
    save(fig, prefix + 'A_Reactome')
    fig, ax = plt.subplots(figsize=(11, 9.2))
    fig.subplots_adjust(left=.03, right=.98, top=.94, bottom=.15)
    draw_network(ax)
    save(fig, prefix + 'B_network')
    fig = plt.figure(figsize=(20, 9.1))
    grid = fig.add_gridspec(1, 2, width_ratios=[.75, 1.45], left=.235, right=.99,
                           top=.94, bottom=.14, wspace=.16)
    first, second = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    draw_enrichment(first)
    draw_network(second)
    for ax, offset, label in [(first, -.08, 'A'), (second, -.03, 'B')]:
        ax.text(offset, 1.01, label, transform=ax.transAxes, fontsize=18,
                fontweight='bold', ha='left', va='bottom')
    save(fig, prefix + '_combined')


def main(case):
    config = CASES[case]
    parser = argparse.ArgumentParser(description=f'DFGNN {case} case-study analysis.')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / case)
    parser.add_argument('--cache-dir', type=Path, default=Path(DEFAULT_CACHE_PATH),
                        help='Cache downloaded Enrichr libraries outside the release data directory.')
    parser.add_argument('--output-dir', type=Path, default=ROOT.parent / 'results' / case)
    parser.add_argument('--skip-gsea', action='store_true', help='Skip the separate preranked GSEA analysis.')
    parser.add_argument('--gsea-permutations', type=int, default=1000)
    parser.add_argument('--formats', nargs='+', choices=['png', 'jpg', 'pdf', 'svg'],
                        default=['png', 'jpg', 'pdf', 'svg'])
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none'})
    case_scores = pd.read_csv(args.data_dir / f'predictions_{config["condition"]}.csv', index_col=0)
    control_scores = pd.read_csv(args.data_dir / f'predictions_{config["control"]}.csv', index_col=0)
    de = pd.read_csv(args.data_dir / 'differential_expression.tsv', sep='\t', index_col='gene')
    positive = pd.read_csv(args.data_dir / 'positive_training_edges.csv')
    training_pairs = set(positive.itertuples(index=False, name=None))
    scores, difference, case_scores, control_scores, fc = score_networks(case_scores, control_scores, de)
    scores.rename_axis('TF').rename('influence_score').to_csv(output / 'regulator_scores.csv')
    pd.Series(fc.index[fc.isna()], name='TF').to_csv(output / 'excluded_regulators.csv', index=False)
    edges = differential_edges(scores, difference, case_scores, control_scores, training_pairs, config)
    edges.to_csv(output / 'differential_edges.csv', index=False)
    genes = sorted(set(edges['tf']) | set(edges['gene']))
    de.loc[genes].to_csv(output / 'differential_genes.csv')
    print(f'{case}: {len(case_scores)} predicted regulators; {len(scores)} eligible regulators; '
          f'{len(edges)} differential edges.', flush=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(np.arange(len(scores)), scores)
    ax.set(title=config['title'], xlabel='TFs', ylabel='Influence score')
    fig.tight_layout()
    fig.savefig(output / 'regulator_influence_scores.jpeg', dpi=300, bbox_inches='tight')
    plt.close(fig)
    libraries, reports = enrich(scores, args.cache_dir, output)
    make_figures(edges, de, case_scores.index, reports['Reactome_2022'], config, output, args.formats)
    if not args.skip_gsea:
        for name in LIBRARIES:
            result = gp.prerank(rnk=scores, gene_sets=libraries[name],
                                outdir=str(output / f'{name}_gsea'),
                                permutation_num=args.gsea_permutations, seed=123,
                                min_size=10, max_size=300, threads=1, no_plot=True)
            result.res2d.sort_values('NES', ascending=False).to_csv(output / f'{name}.gsea.csv', index=False)
    print(f'Saved {case} results to {output}', flush=True)
