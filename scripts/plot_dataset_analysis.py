#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    'train': '#1f77b4',
    'valid': '#ff7f0e',
    'test': '#2ca02c',
}


def parse():
    parser = argparse.ArgumentParser(
        description='Create publication-ready dataset analysis plots from train/valid/test JSON files.'
    )
    parser.add_argument('--split_dir', type=str, required=True,
                        help='Directory containing train.json, valid.json, test.json')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory to save generated plots')
    parser.add_argument('--prefix', type=str, default='dataset',
                        help='Prefix for output file names')
    parser.add_argument('--title_prefix', type=str, default='',
                        help='Optional prefix shown in figure titles')
    parser.add_argument('--split_dir2', type=str, default=None,
                        help='Second split dir for combined two-setting figure')
    parser.add_argument('--label1', type=str, default='Setting I',
                        help='Row label for first setting in combined figure')
    parser.add_argument('--label2', type=str, default='Setting II',
                        help='Row label for second setting in combined figure')
    return parser.parse_args()


def load_jsonl(path):
    items = []
    with open(path, 'r') as fin:
        for line in fin:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def parse_date(date_str):
    if not date_str:
        return None
    for fmt in ('%Y-%m-%d', '%m/%d/%y', '%m/%d/%Y'):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def get_cluster_date(item):
    return item.get('time_split_cluster_date') or item.get('sabdab_date') or item.get('date')


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_fig(fig, out_dir, stem):
    png_path = os.path.join(out_dir, stem + '.png')
    pdf_path = os.path.join(out_dir, stem + '.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def title_with_prefix(prefix, title):
    return f'{prefix}: {title}' if prefix else title


def summarize_split(items):
    return {
        'n_complexes': len(items),
        'n_pdbs': len({item['pdb'] for item in items}),
        'n_clusters': len({item['cluster'] for item in items if 'cluster' in item}),
    }


def boxplot_by_split(ax, arrays, labels, ylabel):
    bp = ax.boxplot(arrays, patch_artist=True, labels=labels, showfliers=False)
    for patch, label in zip(bp['boxes'], labels):
        patch.set_facecolor(COLORS[label.lower()])
        patch.set_alpha(0.55)
    for median in bp['medians']:
        median.set_color('#222222')
        median.set_linewidth(1.6)
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def draw_split_overview(ax, data):
    splits = ['train', 'valid', 'test']
    complex_counts = [len(data[s]) for s in splits]
    cluster_counts = [len({item['cluster'] for item in data[s] if 'cluster' in item}) for s in splits]
    x = np.arange(len(splits))
    width = 0.36

    ax.bar(x - width / 2, complex_counts, width=width,
           color=[COLORS[s] for s in splits], alpha=0.9, label='Complexes')
    ax.bar(x + width / 2, cluster_counts, width=width,
           color=[COLORS[s] for s in splits], alpha=0.35, label='Clusters')

    for i, val in enumerate(complex_counts):
        ax.text(i - width / 2, val, f'{val:,}', ha='center', va='bottom', fontsize=8)
    for i, val in enumerate(cluster_counts):
        ax.text(i + width / 2, val, f'{val:,}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in splits])
    ax.set_ylabel('Count')
    ax.legend(frameon=False, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def draw_timeline(ax, data):
    splits = ['train', 'valid', 'test']
    all_dates = []
    split_dates = {}
    for split in splits:
        dates = [parse_date(get_cluster_date(item)) for item in data[split]]
        dates = [d for d in dates if d is not None]
        split_dates[split] = dates
        all_dates.extend(dates)

    if not all_dates:
        ax.text(0.5, 0.5, 'No date metadata', ha='center', va='center',
                transform=ax.transAxes, fontsize=10)
        ax.set_axis_off()
        return

    date_nums = matplotlib.dates.date2num(all_dates)
    bins = np.linspace(date_nums.min(), date_nums.max(), 26)
    for split in splits:
        dates = split_dates[split]
        if not dates:
            continue
        ax.hist(matplotlib.dates.date2num(dates), bins=bins, alpha=0.65,
                color=COLORS[split], label=split.capitalize())

    ax.xaxis_date()
    ax.set_ylabel('Count')
    ax.legend(frameon=False, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_split_overview(data, out_dir, prefix, title_prefix):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    draw_split_overview(ax, data)
    ax.set_title(title_with_prefix(title_prefix, 'Split Size Overview'))
    save_fig(fig, out_dir, prefix + '_split_overview')


def plot_timeline(data, out_dir, prefix, title_prefix):
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    draw_timeline(ax, data)
    fig.autofmt_xdate(rotation=30, ha='right')
    ax.set_title(title_with_prefix(title_prefix, 'Temporal Distribution of Split Assignment'))
    save_fig(fig, out_dir, prefix + '_timeline')


def plot_cdr_lengths(data, out_dir, prefix, title_prefix):
    cdr_specs = [
        ('cdrh1_seq', 'H1'),
        ('cdrh2_seq', 'H2'),
        ('cdrh3_seq', 'H3'),
        ('cdrl1_seq', 'L1'),
        ('cdrl2_seq', 'L2'),
        ('cdrl3_seq', 'L3'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharex=True)
    axes = axes.flatten()
    labels = ['Train', 'Valid', 'Test']

    for ax, (key, name) in zip(axes, cdr_specs):
        arrays = [
            [len(item[key]) for item in data['train']],
            [len(item[key]) for item in data['valid']],
            [len(item[key]) for item in data['test']],
        ]
        boxplot_by_split(ax, arrays, labels, 'Length')
        ax.set_title(f'CDR-{name}')

    fig.suptitle(title_with_prefix(title_prefix, 'CDR Length Distributions'), y=1.02, fontsize=13)
    fig.tight_layout()
    save_fig(fig, out_dir, prefix + '_cdr_lengths')


def plot_chain_antigen_lengths(data, out_dir, prefix, title_prefix):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    labels = ['Train', 'Valid', 'Test']
    specs = [
        ('heavy_chain_seq', 'Heavy chain length'),
        ('light_chain_seq', 'Light chain length'),
        ('antigen', 'Total antigen length'),
    ]

    for ax, (key, ylabel) in zip(axes, specs):
        if key == 'antigen':
            arrays = [
                [sum(len(seq) for seq in item['antigen_seqs']) for item in data['train']],
                [sum(len(seq) for seq in item['antigen_seqs']) for item in data['valid']],
                [sum(len(seq) for seq in item['antigen_seqs']) for item in data['test']],
            ]
        else:
            arrays = [
                [len(item[key]) for item in data['train']],
                [len(item[key]) for item in data['valid']],
                [len(item[key]) for item in data['test']],
            ]
        boxplot_by_split(ax, arrays, labels, ylabel)

    fig.suptitle(title_with_prefix(title_prefix, 'Chain and Antigen Length Distributions'), y=1.04, fontsize=13)
    fig.tight_layout()
    save_fig(fig, out_dir, prefix + '_chain_antigen_lengths')


def draw_kde(ax, data, key_fn, xlabel, bw=0.35):
    from scipy.stats import gaussian_kde
    for split in ['train', 'valid', 'test']:
        vals = np.array([key_fn(item) for item in data[split]], dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals) < 5:
            continue
        lo, hi = vals.min(), vals.max()
        xs = np.linspace(lo, hi, 400)
        kde = gaussian_kde(vals, bw_method=bw)
        ys = kde(xs)
        ax.plot(xs, ys, color=COLORS[split], linewidth=2.2,
                label=split.capitalize())
        ax.fill_between(xs, ys, alpha=0.13, color=COLORS[split])
        median_val = float(np.median(vals))
        ax.axvline(median_val, color=COLORS[split], linewidth=1.0,
                   linestyle='--', alpha=0.75)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)


def _has_dates(data):
    for split in ['train', 'valid', 'test']:
        for item in data[split][:20]:
            if parse_date(get_cluster_date(item)):
                return True
    return False


def draw_split_size_bar(ax, data):
    splits = ['train', 'valid', 'test']
    counts = [len(data[s]) for s in splits]
    cluster_counts = [len({item['cluster'] for item in data[s] if 'cluster' in item})
                      for s in splits]
    total = max(sum(counts), 1)
    x = np.arange(len(splits))
    width = 0.38

    bars_c = ax.bar(x - width / 2, counts, width=width,
                    color=[COLORS[s] for s in splits], alpha=0.85,
                    edgecolor='white', linewidth=1.2, label='Complexes')
    bars_k = ax.bar(x + width / 2, cluster_counts, width=width,
                    color=[COLORS[s] for s in splits], alpha=0.38,
                    edgecolor='white', linewidth=1.2, label='Clusters')

    for bar, cnt in zip(bars_c, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, cnt,
                f'{cnt:,}\n({100 * cnt / total:.0f}%)',
                ha='center', va='bottom', fontsize=8.5)
    for bar, cnt in zip(bars_k, cluster_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, cnt,
                f'{cnt:,}',
                ha='center', va='bottom', fontsize=8.5)

    ax.set_ylabel('Count', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in splits])
    ax.legend(frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_all_in_one(data, out_dir, prefix, title_prefix):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # Panel (a): timeline if dates exist, otherwise split-size bar
    ax_a = axes[0]
    if _has_dates(data):
        draw_timeline(ax_a, data)
        for lbl in ax_a.get_xticklabels():
            lbl.set_rotation(28)
            lbl.set_ha('right')
        ax_a.legend(frameon=False, fontsize=9)
        panel_a_title = '(a) Temporal Distribution'
    else:
        draw_split_size_bar(ax_a, data)
        panel_a_title = '(a) Split Size'
    ax_a.set_title(panel_a_title, fontsize=11, pad=8)

    # Panel (b): CDR-H3 length — the variable being designed
    ax_b = axes[1]
    draw_kde(ax_b, data,
             key_fn=lambda item: len(item['cdrh3_seq']),
             xlabel='CDR-H3 length (residues)',
             bw=0.5)
    ax_b.set_title('(b) CDR-H3 Length Distribution', fontsize=11, pad=8)

    # Panel (c): antigen length — most informative distribution shift
    ax_c = axes[2]
    draw_kde(ax_c, data,
             key_fn=lambda item: sum(len(seq) for seq in item['antigen_seqs']),
             xlabel='Total antigen length (residues)',
             bw=0.18)
    ax_c.set_title('(c) Antigen Length Distribution', fontsize=11, pad=8)

    fig.suptitle(title_with_prefix(title_prefix, 'Dataset Analysis'),
                 fontsize=13, y=1.03)
    fig.tight_layout()
    save_fig(fig, out_dir, prefix + '_all_in_one')


def plot_combined_two_settings(data_a, label_a, data_b, label_b, out_path):
    """Two-row × 3-column figure combining two dataset settings side by side.

    Row 0 = setting A (e.g. SAbDab → RAbD external benchmark)
    Row 1 = setting B (e.g. SAbDab chronological time split)
    Columns: (0) size/timeline  (1) CDR-H3 KDE  (2) antigen KDE
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    fig.subplots_adjust(hspace=0.52, wspace=0.28)

    row_labels = [label_a, label_b]
    datasets   = [data_a, data_b]

    for row_idx, (data, row_label) in enumerate(zip(datasets, row_labels)):
        axes[row_idx, 0].set_ylabel(row_label, fontsize=11,
                                    fontweight='bold', labelpad=8)

        # Panel col 0: temporal histogram or split-size bar
        ax0 = axes[row_idx, 0]
        if _has_dates(data):
            draw_timeline(ax0, data)
            for lbl in ax0.get_xticklabels():
                lbl.set_rotation(26)
                lbl.set_ha('right')
            ax0.legend(frameon=False, fontsize=8)
            col0_title = 'Temporal Distribution'
        else:
            draw_split_size_bar(ax0, data)
            col0_title = 'Split Size'

        # Panel col 1: CDR-H3 KDE
        ax1 = axes[row_idx, 1]
        draw_kde(ax1, data,
                 key_fn=lambda item: len(item['cdrh3_seq']),
                 xlabel='CDR-H3 length (residues)',
                 bw=0.5)

        # Panel col 2: antigen length KDE
        ax2 = axes[row_idx, 2]
        draw_kde(ax2, data,
                 key_fn=lambda item: sum(len(seq) for seq in item['antigen_seqs']),
                 xlabel='Total antigen length (residues)',
                 bw=0.18)

        # Column titles only on top row
        if row_idx == 0:
            ax0.set_title(col0_title,      fontsize=11, pad=6)
            ax1.set_title('CDR-H3 Length', fontsize=11, pad=6)
            ax2.set_title('Antigen Length', fontsize=11, pad=6)

    png_path = out_path if out_path.endswith('.png') else out_path + '.png'
    pdf_path = png_path.replace('.png', '.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path,           bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved combined figure to {png_path}')


def save_summary_json(data, out_dir, prefix):
    splits = ['train', 'valid', 'test']
    total = sum(len(data[s]) for s in splits)
    summary = {}
    for split in splits:
        items = data[split]
        summary[split] = summarize_split(items)
        summary[split]['ratio_percent'] = round(100.0 * len(items) / total, 4)
        cluster_dates = [get_cluster_date(item) for item in items if get_cluster_date(item)]
        if cluster_dates:
            summary[split]['time_range'] = [min(cluster_dates), max(cluster_dates)]
    path = os.path.join(out_dir, prefix + '_summary.json')
    with open(path, 'w') as fout:
        json.dump(summary, fout, indent=2)


def main(args):
    ensure_dir(args.out_dir)
    data = {
        split: load_jsonl(os.path.join(args.split_dir, split + '.json'))
        for split in ['train', 'valid', 'test']
    }
    plot_split_overview(data, args.out_dir, args.prefix, args.title_prefix)
    plot_timeline(data, args.out_dir, args.prefix, args.title_prefix)
    plot_cdr_lengths(data, args.out_dir, args.prefix, args.title_prefix)
    plot_chain_antigen_lengths(data, args.out_dir, args.prefix, args.title_prefix)
    plot_all_in_one(data, args.out_dir, args.prefix, args.title_prefix)
    save_summary_json(data, args.out_dir, args.prefix)

    if args.split_dir2 is not None:
        data2 = {
            split: load_jsonl(os.path.join(args.split_dir2, split + '.json'))
            for split in ['train', 'valid', 'test']
        }
        combined_path = os.path.join(args.out_dir, args.prefix + '_combined_two_settings.png')
        plot_combined_two_settings(data, args.label1, data2, args.label2, combined_path)


if __name__ == '__main__':
    main(parse())
