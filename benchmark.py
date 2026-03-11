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
from strategies import NaiveGrind, CyclicGrind, BackwardLearning, Semiomniscient, Mastery, SemiomniscientOnline, Poisson, PoissonOnline
from simulator import benchmark


_STRATEGY_TYPES = {
    'naive_grind',
    'cyclic_grind',
    'backward_learning',
    'mastery',
    'semiomniscient',
    'semiomniscient_online',
    'poisson',
    'poisson_online',
}


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
        if entry['type'] not in _STRATEGY_TYPES:
            raise ValueError(
                f"Unknown strategy type '{entry['type']}' in entry {i}. "
                f"Valid types: {sorted(_STRATEGY_TYPES)}"
            )
        if 'simulations' not in entry:
            raise ValueError(f"Strategy entry {i} ({entry['type']}) missing required field 'simulations'")
        if not isinstance(entry['simulations'], int) or entry['simulations'] < 1:
            raise ValueError(f"Strategy entry {i}: 'simulations' must be a positive integer")

    return entries


def _make_key(entry: Dict, seen_keys: Dict[str, int]) -> str:
    """Generate a unique result-dict key for a strategy config entry."""
    t = entry['type']
    if t == 'naive_grind':
        base = 'naive'
    elif t == 'cyclic_grind':
        base = 'cyclic'
    elif t == 'backward_learning':
        chunk = entry.get('chunk_size', 1)
        base = 'backward' if chunk == 1 else f'backward_{chunk}'
    elif t == 'mastery':
        base = f"mastery_{entry.get('k', 5)}"
    elif t == 'semiomniscient':
        base = 'semiomniscient'
    elif t == 'semiomniscient_online':
        base = 'semiomniscient_online'
    elif t == 'poisson':
        base = 'poisson'
    elif t == 'poisson_online':
        base = 'poisson_online'
    else:
        base = t

    count = seen_keys.get(base, 0) + 1
    seen_keys[base] = count
    return base if count == 1 else f'{base}_{count}'


def _build_strategy_list(entries: List[Dict], room_names: List[str], models: RoomModels):
    """
    Convert config entries into a list of (key, strategy_class, args, n_sims) tuples.
    """
    strategies = []
    seen_keys: Dict[str, int] = {}

    for entry in entries:
        t = entry['type']
        n_sims = entry['simulations']
        key = _make_key(entry, seen_keys)

        if t == 'naive_grind':
            strategies.append((key, NaiveGrind, (room_names,), n_sims))
        elif t == 'cyclic_grind':
            strategies.append((key, CyclicGrind, (room_names,), n_sims))
        elif t == 'backward_learning':
            chunk = entry.get('chunk_size', 1)
            strategies.append((key, BackwardLearning, (room_names, chunk), n_sims))
        elif t == 'mastery':
            k = entry.get('k', 5)
            strategies.append((key, Mastery, (room_names, k), n_sims))
        elif t == 'semiomniscient':
            strategies.append((key, Semiomniscient, (room_names, models), n_sims))
        elif t == 'semiomniscient_online':
            neg_beta_threshold = entry.get('neg_beta_threshold', 0.5)
            stability_window = entry.get('stability_window', 5)
            stability_eps = entry.get('stability_eps', 0.1)
            strategies.append((key, SemiomniscientOnline, (room_names, models, neg_beta_threshold, stability_window, stability_eps), n_sims))
        elif t == 'poisson':
            strategies.append((key, Poisson, (room_names, models), n_sims))
        elif t == 'poisson_online':
            alpha = entry.get('alpha', 1.96)
            tau = entry.get('tau', 3.0)
            strategies.append((key, PoissonOnline, (room_names, models, alpha, tau), n_sims))

    return strategies


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

    strategy_list = _build_strategy_list(entries, room_names, models)

    # Actual results from data
    actual_time = sum(model_params[r]['total_time'] for r in room_names)
    actual_attempts = {r: model_params[r]['total_attempts'] for r in room_names}

    total_sims = sum(n for _, _, _, n in strategy_list)
    print(f"{'=' * 60}")
    print(f"RUNNING BENCHMARKS ({len(strategy_list)} strategies, {total_sims} total simulations)")
    print(f"{'=' * 60}")
    print()

    results = {}
    for key, strategy_class, args, n_sims in strategy_list:
        print(f"Testing {strategy_class(*args).name} ({n_sims} simulations)...")
        results[key] = benchmark(strategy_class, args, models, n_sims)

    # Print results
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print()

    for key, result in results.items():
        print(f"{result['strategy']}:")
        print(f"  Mean time: {format_time(result['mean_time'])} ± {format_time(result['std_time'])}")
        print()

    print(f"Actual results:")
    print(f"  Time: {format_time(actual_time)}")
    print()

    # Derive mastery K values from config for the sweep plot
    window_ks = [e.get('k', 5) for e in entries if e['type'] == 'mastery']

    # Generate plots
    _create_plots(results, actual_time, actual_attempts, room_names, plots_dir)
    _create_mastery_sweep_plot(results, window_ks, plots_dir)

    # Save results
    results_file = os.path.join(data_dir, 'benchmark_results.json')
    save_results = {
        key: {
            'strategy': r['strategy'],
            'mean_time': r['mean_time'],
            'std_time': r['std_time'],
            'room_attempts': r['room_attempts']
        }
        for key, r in results.items()
    }
    save_results['actual'] = {
        'total_time': actual_time,
        'attempts_per_room': actual_attempts
    }

    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2)

    # Print winner
    print("=" * 60)
    best = min(results.items(), key=lambda x: x[1]['mean_time'])
    print(f"🏆 WINNER: {best[1]['strategy']}")
    print(f"   Expected: {format_time(best[1]['mean_time'])}")
    print(f"   Actual:   {format_time(actual_time)}")
    print("=" * 60)

    return results


def _create_mastery_sweep_plot(
    results: Dict,
    window_ks: List[int],
    plots_dir: str
):
    """Create a plot showing mean completion time as a function of K for mastery practice."""
    ks = []
    means = []
    stds = []

    for k in window_ks:
        key = f'mastery_{k}'
        if key not in results:
            continue
        ks.append(k)
        means.append(results[key]['mean_time'] / 3600)
        stds.append(results[key]['std_time'] / 3600)

    if not ks:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    means = np.array(means)
    stds = np.array(stds)
    ks = np.array(ks)

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
        ('naive', 'Naive Grind', 'gray'),
    ]
    for key, label, color in ref_strategies:
        if key in results:
            ref_hours = results[key]['mean_time'] / 3600
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
    results: Dict,
    actual_time: float,
    actual_attempts: Dict[str, int],
    room_names: List[str],
    plots_dir: str
):
    """Create all benchmark visualizations."""

    strategy_names = [r['strategy'] for r in results.values()]

    # 1. Ridgeline plot of time distributions
    from scipy.stats import gaussian_kde

    # Sort strategies by mean time (fastest at bottom so the eye reads upward)
    sorted_keys = sorted(results.keys(), key=lambda k: results[k]['mean_time'], reverse=True)
    n_strategies = len(sorted_keys)

    # Global x range across all strategies
    all_times_hours = []
    for key in sorted_keys:
        all_times_hours.append(np.array(results[key]['all_times']) / 3600)
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

    for i, key in enumerate(sorted_keys):
        times_h = all_times_hours[i]
        density = kdes[i]

        scaled = density / max_density * ridge_height if max_density > 0 else density
        baseline = i * ridge_height * (1 - overlap)

        color = colors_ridge[i]
        ax.fill_between(xs, baseline, baseline + scaled, alpha=0.7,
                        color=color, edgecolor='white', linewidth=0.8)
        ax.plot(xs, baseline + scaled, color=color, linewidth=1.2)

        label = results[key]['strategy']
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

    mean_times = [r['mean_time'] / 3600 for r in results.values()]
    std_times = [r['std_time'] / 3600 for r in results.values()]
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

        means = [results[k]['room_attempts'][room]['mean'] for k in results.keys()]
        stds = [results[k]['room_attempts'][room]['std'] for k in results.keys()]
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
