#!/usr/bin/env python3
"""
Benchmark strategies and generate comparison plots.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from models import RoomModels, format_time
from strategies import NaiveGrind, CyclicGrind, BackwardLearning, Semiomniscient, WindowedPractice
from simulator import benchmark


def run_benchmark(
    model_params: Dict,
    plots_dir: str,
    data_dir: str,
    n_simulations: int = 1000,
    chunk_sizes: List[int] | None = None,
    window_sizes: List[int] | None = None
):
    """
    Run benchmarks on all strategies and generate plots.
    
    Args:
        model_params: Dict of room model parameters
        plots_dir: Output directory for plots
        data_dir: Output directory for results JSON
        n_simulations: Number of Monte Carlo simulations
        chunk_sizes: List of chunk sizes for backward learning variants
        window_sizes: List of K values for windowed practice sweep
    """
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    models = RoomModels(model_params)
    room_names = models.room_names

    if chunk_sizes is None:
        chunk_sizes = [len(room_names) // 5]
    if window_sizes is None:
        window_sizes = [3, 5, 7, 10, 15, 20]
    
    # Calculate actual results from data
    actual_time = sum(model_params[r]['total_time'] for r in room_names)
    actual_attempts = {r: model_params[r]['total_attempts'] for r in room_names}
    
    print(f"{'=' * 60}")
    print(f"RUNNING BENCHMARKS ({n_simulations} simulations each)")
    print(f"{'=' * 60}")
    print()
    
    results = {}
    
    # Define strategies to test
    strategies = [
        ('naive', NaiveGrind, (room_names,)),
        ('cyclic', CyclicGrind, (room_names,)),
        ('backward', BackwardLearning, (room_names, 1)),
    ]
    
    for chunk_size in chunk_sizes:
        strategies.append(
            (f'backward_{chunk_size}', BackwardLearning, (room_names, chunk_size))
        )
    
    for k in window_sizes:
        strategies.append(
            (f'windowed_{k}', WindowedPractice, (room_names, k))
        )
    
    strategies.append(
        ('semiomniscient', Semiomniscient, (room_names, models))
    )
    
    # Run benchmarks
    for key, strategy_class, args in strategies:
        print(f"Testing {strategy_class(*args).name}...")
        results[key] = benchmark(strategy_class, args, models, n_simulations)
    
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
    
    # Generate plots
    _create_plots(results, actual_time, actual_attempts, room_names, plots_dir)
    _create_windowed_sweep_plot(results, window_sizes, plots_dir)
    
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


def _create_windowed_sweep_plot(
    results: Dict,
    window_sizes: List[int],
    plots_dir: str
):
    """Create a plot showing mean completion time as a function of K for windowed practice."""
    ks = []
    means = []
    stds = []

    for k in window_sizes:
        key = f'windowed_{k}'
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
    ax.set_title('Windowed Practice: Effect of K on Completion Time',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(ks)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'windowed_k_sweep.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved windowed K sweep plot to {plots_dir}/windowed_k_sweep.png")


def _create_plots(
    results: Dict,
    actual_time: float,
    actual_attempts: Dict[str, int],
    room_names: List[str],
    plots_dir: str
):
    """Create all benchmark visualizations."""
    
    strategy_names = [r['strategy'] for r in results.values()]
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # 1. Time distribution histogram with mean lines
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, (key, result) in enumerate(results.items()):
        times_hours = np.array(result['all_times']) / 3600
        ax.hist(times_hours, alpha=0.8, label=result['strategy'],
                color=colors[i])
    
    # Actual result line
    actual_hours = actual_time / 3600
    ax.axvline(actual_hours, color='black', linestyle='--', linewidth=3,
               label=f'Actual ({format_time(actual_time)})')
    
    ax.set_xlabel('Time to Completion (hours)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Time Distribution Comparison', fontweight='bold', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'time_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Mean time bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mean_times = [r['mean_time'] / 3600 for r in results.values()]
    std_times = [r['std_time'] / 3600 for r in results.values()]
    x = np.arange(len(strategy_names))
    
    # Color by performance
    sorted_indices = np.argsort(mean_times)
    bar_colors = np.zeros((len(mean_times), 4))
    for rank, idx in enumerate(sorted_indices):
        bar_colors[idx] = plt.cm.RdYlGn_r(rank / (len(mean_times) - 1) * 0.6 + 0.2)
    
    bars = ax.bar(x, mean_times, yerr=std_times, alpha=0.8, capsize=5, color=bar_colors)
    ax.axhline(actual_hours, color='black', linestyle='--', linewidth=2, label=f'Actual ({format_time(actual_time)})')
    
    # Add numeric labels on bars
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
        
        # Add numeric labels on bars
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