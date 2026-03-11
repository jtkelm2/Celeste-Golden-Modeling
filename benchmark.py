#!/usr/bin/env python3
"""
Benchmark strategies and generate comparison plots.
"""

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]  # pip install tomli for Python < 3.11
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Python 3.11+ is required, or install the tomli package: pip install tomli"
        )

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from models import RoomModels, format_time
from strategies import build_strategy, STRATEGY_TYPES
from simulator import benchmark


def _load_config(config_path: str) -> List[Dict]:
    """Load and validate benchmark_config.toml. Returns the list of strategy entries."""
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)

    entries = config.get('strategies', [])
    if not entries:
        raise ValueError(f"No [[strategies]] entries found in {config_path}")

    for i, entry in enumerate(entries):
        if 'type' not in entry:
            raise ValueError(f"Strategy entry {i} missing required field 'type'")
        if entry['type'] not in STRATEGY_TYPES:
            raise ValueError(
                f"Unknown strategy type '{entry['type']}' in entry {i}. "
                f"Valid types: {sorted(STRATEGY_TYPES)}"
            )
        if 'simulations' not in entry:
            raise ValueError(f"Strategy entry {i} ({entry['type']}) missing required field 'simulations'")
        if not isinstance(entry['simulations'], int) or entry['simulations'] < 1:
            raise ValueError(f"Strategy entry {i}: 'simulations' must be a positive integer")

    return entries


def run_benchmark(
    model_params: Dict,
    plots_dir: str,
    data_dir: str,
    config_path: str = 'benchmark_config.toml',
):
    """
    Run benchmarks on configured strategies and generate plots.

    Args:
        model_params: Dict of room model parameters
        plots_dir:    Output directory for plots
        data_dir:     Output directory for results JSON
        config_path:  Path to TOML benchmark config (default: benchmark_config.toml)
    """
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    entries = _load_config(config_path)
    models = RoomModels(model_params)
    room_names = models.room_names

    # Actual results from data
    actual_time = sum(model_params[r]['total_time'] for r in room_names)
    actual_attempts = {r: model_params[r]['total_attempts'] for r in room_names}

    total_sims = sum(e['simulations'] for e in entries)
    print(f"{'=' * 60}")
    print(f"RUNNING BENCHMARKS ({len(entries)} strategies, {total_sims} total simulations)")
    print(f"{'=' * 60}")
    print()

    results = []
    for entry in entries:
        cls, args = build_strategy(entry['type'], entry, room_names, models)
        n_sims = entry['simulations']
        print(f"Testing {cls(*args).name} ({n_sims} simulations)...")  # type: ignore[arg-type]
        r = benchmark(cls, args, models, n_sims)
        r['entry'] = entry
        results.append(r)

    # Print results
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print()

    for r in results:
        print(f"{r['strategy']}:")
        print(f"  Mean time: {format_time(r['mean_time'])} ± {format_time(r['std_time'])}")
        print()

    print(f"Actual results:")
    print(f"  Time: {format_time(actual_time)}")
    print()

    # Generate plots
    _create_plots(results, actual_time, actual_attempts, room_names, plots_dir)
    _create_mastery_sweep_plot(results, plots_dir)

    # Save results
    results_file = os.path.join(data_dir, 'benchmark_results.json')
    save_results = {
        'strategies': [
            {
                'strategy': r['strategy'],
                'mean_time': r['mean_time'],
                'std_time': r['std_time'],
                'room_attempts': r['room_attempts'],
            }
            for r in results
        ],
        'actual': {
            'total_time': actual_time,
            'attempts_per_room': actual_attempts,
        },
    }

    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2)

    # Print winner
    print("=" * 60)
    best = min(results, key=lambda r: r['mean_time'])
    print(f"🏆 WINNER: {best['strategy']}")
    print(f"   Expected: {format_time(best['mean_time'])}")
    print(f"   Actual:   {format_time(actual_time)}")
    print("=" * 60)

    return results


def _create_mastery_sweep_plot(
    results: List[Dict],
    plots_dir: str,
):
    """Create a plot showing mean completion time as a function of K for mastery practice."""
    mastery_results = [r for r in results if r['entry']['type'] == 'mastery']
    if not mastery_results:
        return

    ks = [r['entry'].get('k', 5) for r in mastery_results]
    means = np.array([r['mean_time'] / 3600 for r in mastery_results])
    stds = np.array([r['std_time'] / 3600 for r in mastery_results])
    ks = np.array(ks)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ks, means, 'o-', color='steelblue', linewidth=2, markersize=8, label='Mean time')
    ax.fill_between(ks, means - stds, means + stds, alpha=0.2, color='steelblue', label='± 1 std')

    # Mark the optimal K
    best_idx = np.argmin(means)
    ax.axvline(ks[best_idx], color='green', linestyle='--', alpha=0.7,
               label=f'Best K={ks[best_idx]} ({means[best_idx]:.1f}h)')
    ax.plot(ks[best_idx], means[best_idx], '*', color='green', markersize=15, zorder=5)

    # Reference lines for other strategies if available
    ref_strategies = [
        ('semiomniscient', 'Semiomniscient', 'red'),
        ('naive_grind', 'Naive Grind', 'gray'),
    ]
    for rtype, label, color in ref_strategies:
        matching = [r for r in results if r['entry']['type'] == rtype]
        if matching:
            ref_hours = matching[0]['mean_time'] / 3600
            ax.axhline(ref_hours, color=color, linestyle=':', linewidth=1.5, alpha=0.7,
                       label=f'{label} ({ref_hours:.1f}h)')

    ax.set_xlabel('K (consecutive successes required)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Time to Completion (hours)', fontsize=12, fontweight='bold')
    ax.set_title('Mastery: Effect of K on Completion Time',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(ks)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mastery_k_sweep.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved mastery K sweep plot to {plots_dir}/mastery_k_sweep.png")


def _create_plots(
    results: List[Dict],
    actual_time: float,
    actual_attempts: Dict[str, int],
    room_names: List[str],
    plots_dir: str
):
    """Create all benchmark visualizations."""

    strategy_names = [r['strategy'] for r in results]

    # 1. Ridgeline plot of time distributions
    from scipy.stats import gaussian_kde

    # Sort strategies by mean time (fastest at bottom so the eye reads upward)
    sorted_results = sorted(results, key=lambda r: r['mean_time'], reverse=True)
    n_strategies = len(sorted_results)

    # Global x range across all strategies
    all_times_hours = [np.array(r['all_times']) / 3600 for r in sorted_results]
    x_min = min(t.min() for t in all_times_hours)
    x_max = max(t.max() for t in all_times_hours)
    x_pad = (x_max - x_min) * 0.05
    xs = np.linspace(x_min - x_pad, x_max + x_pad, 500)

    actual_hours = actual_time / 3600

    # Compute KDEs and find global max density for uniform scaling
    kdes = []
    for times_h in all_times_hours:
        if len(set(times_h)) > 1:
            kde = gaussian_kde(times_h, bw_method='scott')
            kdes.append(kde(xs))
        else:
            kdes.append(np.zeros_like(xs))
    max_density = max(k.max() for k in kdes) if kdes else 1.0

    # Vertical spacing between ridges
    overlap = 0.6
    ridge_height = 1.0

    fig_height = max(6, 1.2 + n_strategies * ridge_height * (1 - overlap * 0.5))
    fig, ax = plt.subplots(figsize=(12, fig_height))

    colors_ridge = plt.colormaps['viridis'](np.linspace(0.15, 0.85, n_strategies))

    for i, r in enumerate(sorted_results):
        times_h = all_times_hours[i]
        density = kdes[i]

        scaled = density / max_density * ridge_height if max_density > 0 else density
        baseline = i * ridge_height * (1 - overlap)

        color = colors_ridge[i]
        ax.fill_between(xs, baseline, baseline + scaled, alpha=0.7,
                        color=color, edgecolor='white', linewidth=0.8)
        ax.plot(xs, baseline + scaled, color=color, linewidth=1.2)

        label = r['strategy']
        mean_h = np.mean(times_h)
        ax.text(x_min - x_pad - (x_max - x_min) * 0.01, baseline + ridge_height * 0.15,
                f'{label}  ({mean_h:.1f}h)',
                ha='right', va='bottom', fontsize=9, fontweight='bold',
                color=color)

    y_top = n_strategies * ridge_height * (1 - overlap) + ridge_height
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
        bar_colors[idx] = plt.colormaps['RdYlGn_r'](rank / (len(mean_times) - 1) * 0.6 + 0.2)

    bars = ax.bar(x, mean_times, yerr=std_times, alpha=0.8, capsize=5, color=bar_colors)
    ax.axhline(actual_hours, color='black', linestyle='--', linewidth=2, label=f'Actual ({format_time(actual_time)})')

    for i, (bar, mean, std) in enumerate(zip(bars, mean_times, std_times)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_times[i] * 1.04,
                f'{mean:.1f} ± {std:.1f}h', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(strategy_names, rotation=45, ha='right')
    ax.set_ylabel('Time (hours)', fontweight='bold')
    ax.set_title('Mean Time to Completion', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mean_time_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Per-room attempt counts
    for room in room_names:
        fig, ax = plt.subplots(figsize=(10, 6))

        means = [r['room_attempts'][room]['mean'] for r in results]
        stds = [r['room_attempts'][room]['std'] for r in results]
        x = np.arange(len(strategy_names))

        bars = ax.bar(x, means, yerr=stds, alpha=0.7, capsize=5, color='steelblue')
        ax.axhline(actual_attempts[room], color='black', linestyle='--', linewidth=2,
                   label=f'Actual ({actual_attempts[room]})')

        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                    f'{mean:.0f} ± {std:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(strategy_names, rotation=45, ha='right')
        ax.set_ylabel('Attempts', fontweight='bold')
        ax.set_title(f'Room {room}: Attempts per Strategy', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'room_{room}_attempts.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Saved plots to {plots_dir}/")
