#!/usr/bin/env python3
"""
Generate benchmark plots from saved result files.

Usage:
    python plot.py data/benchmark_results.json [--output plots/benchmark/]
    python plot.py data/synthetic/*_benchmark.json          # batch: plots alongside each file
"""

import argparse
import glob as glob_module
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from models import format_time


def plot_results(results_path: str, plots_dir: str):
    """Load a benchmark results JSON and generate all plots into plots_dir."""
    with open(results_path) as f:
        data = json.load(f)

    room_names: List[str] = data['room_names']
    results: List[Dict] = data['strategies']
    actual_time: Optional[float] = data.get('actual', {}).get('total_time')
    actual_attempts: Optional[Dict] = data.get('actual', {}).get('attempts_per_room')

    os.makedirs(plots_dir, exist_ok=True)
    _create_plots(results, actual_time, actual_attempts, room_names, plots_dir)
    _create_mastery_sweep_plot(results, plots_dir)
    print(f"Saved plots to {plots_dir}/")


def plot_batch(result_paths: List[str], plots_base_dir: str = 'plots/batch'):
    """
    Generate plots for each result file.
    Plots go into plots_base_dir/<stem>/ for each file.
    """
    for path in result_paths:
        stem = os.path.splitext(os.path.basename(path))[0].removesuffix('_benchmark')
        plots_dir = os.path.join(plots_base_dir, stem)
        print(f"Plotting {os.path.basename(path)}...")
        plot_results(path, plots_dir)


# ── Internal plot helpers ─────────────────────────────────────────────────────

def _create_mastery_sweep_plot(results: List[Dict], plots_dir: str):
    """Plot mean completion time as a function of K for Mastery strategy entries."""
    mastery_results = sorted(
        [r for r in results if r.get('entry', {}).get('type') == 'mastery'],
        key=lambda r: r['entry'].get('k', 5),
    )
    if not mastery_results:
        return

    ks = np.array([r['entry'].get('k', 5) for r in mastery_results])
    means = np.array([r['mean_time'] / 3600 for r in mastery_results])
    stds = np.array([r['std_time'] / 3600 for r in mastery_results])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ks, means, 'o-', color='steelblue', linewidth=2, markersize=8, label='Mean time')
    ax.fill_between(ks, means - stds, means + stds, alpha=0.2, color='steelblue', label='± 1 std')

    best_idx = np.argmin(means)
    ax.axvline(ks[best_idx], color='green', linestyle='--', alpha=0.7,
               label=f'Best K={ks[best_idx]} ({means[best_idx]:.1f}h)')
    ax.plot(ks[best_idx], means[best_idx], '*', color='green', markersize=15, zorder=5)

    for rtype, label, color in [('semiomniscient', 'Semiomniscient', 'red'),
                                  ('naive_grind', 'Naive Grind', 'gray')]:
        matching = [r for r in results if r.get('entry', {}).get('type') == rtype]
        if matching:
            ref_hours = matching[0]['mean_time'] / 3600
            ax.axhline(ref_hours, color=color, linestyle=':', linewidth=1.5, alpha=0.7,
                       label=f'{label} ({ref_hours:.1f}h)')

    ax.set_xlabel('K (consecutive successes required)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Time to Completion (hours)', fontsize=12, fontweight='bold')
    ax.set_title('Mastery: Effect of K on Completion Time', fontsize=14, fontweight='bold')
    ax.set_xticks(ks)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mastery_k_sweep.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _create_plots(
    results: List[Dict],
    actual_time: Optional[float],
    actual_attempts: Optional[Dict[str, int]],
    room_names: List[str],
    plots_dir: str,
):
    """Create all standard benchmark visualizations."""
    from scipy.stats import gaussian_kde

    strategy_names = [r['strategy'] for r in results]

    # 1. Ridgeline plot of time distributions
    sorted_results = sorted(results, key=lambda r: r['mean_time'], reverse=True)
    n_strategies = len(sorted_results)

    all_times_hours = [np.array(r['all_times']) / 3600 for r in sorted_results]
    x_min = min(t.min() for t in all_times_hours)
    x_max = max(t.max() for t in all_times_hours)
    x_pad = (x_max - x_min) * 0.05
    xs = np.linspace(x_min - x_pad, x_max + x_pad, 500)

    kdes = []
    for times_h in all_times_hours:
        if len(set(times_h)) > 1:
            kdes.append(gaussian_kde(times_h, bw_method='scott')(xs))
        else:
            kdes.append(np.zeros_like(xs))
    max_density = max(k.max() for k in kdes) if kdes else 1.0

    overlap = 0.6
    ridge_height = 1.0
    fig_height = max(6, 1.2 + n_strategies * ridge_height * (1 - overlap * 0.5))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    colors_ridge = plt.colormaps['viridis'](np.linspace(0.15, 0.85, n_strategies))

    for i, r in enumerate(sorted_results):
        times_h = all_times_hours[i]
        scaled = kdes[i] / max_density * ridge_height if max_density > 0 else kdes[i]
        baseline = i * ridge_height * (1 - overlap)
        color = colors_ridge[i]
        ax.fill_between(xs, baseline, baseline + scaled, alpha=0.7,
                        color=color, edgecolor='white', linewidth=0.8)
        ax.plot(xs, baseline + scaled, color=color, linewidth=1.2)
        ax.text(x_min - x_pad - (x_max - x_min) * 0.01, baseline + ridge_height * 0.15,
                f'{r["strategy"]}  ({np.mean(times_h):.1f}h)',
                ha='right', va='bottom', fontsize=9, fontweight='bold', color=color)

    y_top = n_strategies * ridge_height * (1 - overlap) + ridge_height
    if actual_time is not None:
        actual_hours = actual_time / 3600
        ax.plot([actual_hours, actual_hours], [0, y_top],
                color='black', linestyle='--', linewidth=2, zorder=10)
        ax.text(actual_hours, y_top * 1.01, f'Actual ({format_time(actual_time)})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Time to Completion (hours)', fontsize=12, fontweight='bold')
    ax.set_title('Time Distribution Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_yticks([])
    ax.set_xlim(x_min - x_pad - (x_max - x_min) * 0.35, x_max + x_pad)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'time_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Mean time bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_times = [r['mean_time'] / 3600 for r in results]
    std_times = [r['std_time'] / 3600 for r in results]
    x = np.arange(len(strategy_names))

    sorted_indices = np.argsort(mean_times)
    bar_colors = np.zeros((len(mean_times), 4))
    for rank, idx in enumerate(sorted_indices):
        bar_colors[idx] = plt.colormaps['RdYlGn_r'](rank / max(len(mean_times) - 1, 1) * 0.6 + 0.2)

    bars = ax.bar(x, mean_times, yerr=std_times, alpha=0.8, capsize=5, color=bar_colors)
    if actual_time is not None:
        actual_hours = actual_time / 3600
        ax.axhline(actual_hours, color='black', linestyle='--', linewidth=2,
                   label=f'Actual ({format_time(actual_time)})')

    for i, (bar, mean, std) in enumerate(zip(bars, mean_times, std_times)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std_times[i] * 1.04,
                f'{mean:.1f} ± {std:.1f}h', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(strategy_names, rotation=45, ha='right')
    ax.set_ylabel('Time (hours)', fontweight='bold')
    ax.set_title('Mean Time to Completion', fontweight='bold', fontsize=14)
    if actual_time is not None:
        ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mean_time_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Per-room attempt counts
    room_attempts_dir = os.path.join(plots_dir, 'room_attempts')
    os.makedirs(room_attempts_dir, exist_ok=True)
    for room in room_names:
        fig, ax = plt.subplots(figsize=(10, 6))
        means = [r['room_attempts'][room]['mean'] for r in results]
        stds = [r['room_attempts'][room]['std'] for r in results]
        x = np.arange(len(strategy_names))

        ax.bar(x, means, yerr=stds, alpha=0.7, capsize=5, color='steelblue')
        if actual_attempts is not None:
            ax.axhline(actual_attempts[room], color='black', linestyle='--', linewidth=2,
                       label=f'Actual ({actual_attempts[room]})')

        for i, (bar, mean, std) in enumerate(zip(ax.patches, means, stds)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 1,
                    f'{mean:.0f} ± {std:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(strategy_names, rotation=45, ha='right')
        ax.set_ylabel('Attempts', fontweight='bold')
        ax.set_title(f'Room {room}: Attempts per Strategy', fontweight='bold', fontsize=14)
        if actual_attempts is not None:
            ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(room_attempts_dir, f'{room}.png'), dpi=300, bbox_inches='tight')
        plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate benchmark plots from saved result files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'results',
        nargs='+',
        metavar='PATH',
        help='Benchmark result JSON file(s). Globs are supported.',
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        metavar='DIR',
        help='Explicit output directory (overrides --job-name).',
    )
    parser.add_argument(
        '--job-name',
        default=None,
        metavar='NAME',
        help='Job name: plots go to plots/<NAME>/ (single) or plots/<NAME>/<stem>/ (batch).',
    )
    args = parser.parse_args()

    # Expand globs
    result_paths = []
    for pattern in args.results:
        expanded = sorted(glob_module.glob(pattern))
        if expanded:
            result_paths.extend(expanded)
        elif os.path.isfile(pattern):
            result_paths.append(pattern)
        else:
            print(f"Warning: no files matched: {pattern}")

    if not result_paths:
        print("Error: no result files found.")
        raise SystemExit(1)

    if len(result_paths) == 1:
        if args.output:
            plots_dir = args.output
        elif args.job_name:
            plots_dir = os.path.join('plots', args.job_name)
        else:
            plots_dir = 'plots/benchmark'
        plot_results(result_paths[0], plots_dir)
    else:
        if args.output:
            plots_base_dir = args.output
        elif args.job_name:
            plots_base_dir = os.path.join('plots', args.job_name)
        else:
            plots_base_dir = 'plots/batch'
        plot_batch(result_paths, plots_base_dir=plots_base_dir)
