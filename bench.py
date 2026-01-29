#!/usr/bin/env python3
"""
Benchmark different Celeste practice strategies
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from strategies import (
    NaiveGrind, CyclicGrind, BackwardLearning, BackwardChunkLearning, Semiomniscient,
    monte_carlo_benchmark, format_time
)
import os

data_dir = 'data'
plots_dir = 'plots'

# Load model parameters
with open(os.path.join(data_dir,'model_parameters.json'), 'r') as f:
    model_params = json.load(f)

# Get room names
room_names = sorted(model_params.keys())
print(f"Loaded {len(room_names)} rooms: {room_names[0]} to {room_names[-1]}")

# Create custom backward chunk learning with larger chunks
chunk_library = []
chunk_library_sizes = [7]
for chunk_size in chunk_library_sizes:
  custom_chunks = []
  # Build cumulative chunks from the end
  for i in range(chunk_size, len(room_names) + 1, chunk_size):
      chunk = room_names[-i:]  # Last i rooms
      custom_chunks.append(chunk)
  # Add final chunk with all rooms if not already covered
  if len(custom_chunks) == 0 or len(custom_chunks[-1]) < len(room_names):
      custom_chunks.append(room_names)

  print(f"\nCustom chunks (size {chunk_size}):")
  for i, chunk in enumerate(custom_chunks):
      print(f"  Chunk {i}: {chunk}")
  chunk_library.append(custom_chunks.copy())

# Set simulation parameters
N_SIMULATIONS = 10
print(f"\n{'='*80}")
print(f"RUNNING BENCHMARKS WITH {N_SIMULATIONS} SIMULATIONS EACH")
print(f"{'='*80}\n")

# Benchmark strategies
results = {}

# 1. Naive Grind
print("1. Testing Naive Grind Strategy...")
results['naive'] = monte_carlo_benchmark(
    NaiveGrind,
    (room_names,),
    model_params,
    N_SIMULATIONS
)

# 2. Cyclic Grind
print("\n2. Testing Cyclic Grind Strategy...")
results['cyclic'] = monte_carlo_benchmark(
    CyclicGrind,
    (room_names,),
    model_params,
    N_SIMULATIONS
)

# 3. Backward Learning
print("\n3. Testing Backward Learning Strategy...")
results['backward'] = monte_carlo_benchmark(
    BackwardLearning,
    (room_names,),
    model_params,
    N_SIMULATIONS
)

# # 4. Backward Chunk Learning
for chunk_size, chunks in zip(chunk_library_sizes, chunk_library):
  print(f"\n4. Testing Backward Chunk Learning Strategy ({chunk_size})...")
  results[f'backward_chunks_{chunk_size}'] = monte_carlo_benchmark(
      BackwardChunkLearning,
      (room_names, chunks, f"{chunk_size}"),
      model_params,
      N_SIMULATIONS
  )

# 5. Semiomniscient Learning
print("\n5. Testing Semiomniscient Learning Strategy...")
results['semiomniscient'] = monte_carlo_benchmark(
      Semiomniscient,
      (room_names, model_params),
      model_params,
      N_SIMULATIONS
)

# Print results
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80 + "\n")

for key, result in results.items():
    print(f"{result['strategy']}:")
    print(f"  Mean time to completion: {format_time(result['mean_time'])}")
    print(f"  Median time: {format_time(result['median_time'])}")
    print(f"  Std dev: {format_time(result['std_time'])}")
    print(f"  95th percentile: {format_time(result['percentiles_time']['p95'])}")
    print(f"  Mean attempts: {result['mean_attempts']:.0f}")
    print(f"  Median attempts: {result['median_attempts']:.0f}")
    print()

# Create comparison visualizations
print("Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Time distribution comparison
ax = axes[0, 0]
for key, result in results.items():
    times_hours = np.array(result['all_times']) / 3600  # Convert to hours
    ax.hist(times_hours, alpha=0.5, bins=50, label=result['strategy'], density=True)
ax.set_xlabel('Time to Completion (hours)', fontweight='bold')
ax.set_ylabel('Density', fontweight='bold')
ax.set_title('Time Distribution Comparison', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Box plot comparison
ax = axes[0, 1]
data_to_plot = [np.array(result['all_times']) / 3600 for result in results.values()]
labels = [result['strategy'] for result in results.values()]
bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.set_ylabel('Time to Completion (hours)', fontweight='bold')
ax.set_title('Time Distribution (Box Plot)', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 3. Attempt count comparison
ax = axes[1, 0]
strategy_names = [result['strategy'] for result in results.values()]
mean_attempts = [result['mean_attempts'] for result in results.values()]
std_attempts = [result['std_attempts'] for result in results.values()]
x_pos = np.arange(len(strategy_names))
ax.bar(x_pos, mean_attempts, yerr=std_attempts, alpha=0.7, capsize=5)
ax.set_xticks(x_pos)
ax.set_xticklabels(strategy_names, rotation=45, ha='right')
ax.set_ylabel('Total Attempts', fontweight='bold')
ax.set_title('Mean Attempts to Completion', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 4. Mean time comparison
ax = axes[1, 1]
mean_times = [result['mean_time'] / 3600 for result in results.values()]
std_times = [result['std_time'] / 3600 for result in results.values()]
x_pos = np.arange(len(strategy_names))
bars = ax.bar(x_pos, mean_times, yerr=std_times, alpha=0.7, capsize=5)
# Color bars by performance (green = best, red = worst)
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bars)))
for bar, color in zip(bars, colors):
    bar.set_color(color)
ax.set_xticks(x_pos)
ax.set_xticklabels(strategy_names, rotation=45, ha='right')
ax.set_ylabel('Time (hours)', fontweight='bold')
ax.set_title('Mean Time to Completion', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir,'strategy_comparison.png'), dpi=300, bbox_inches='tight')
print(f"Saved comparison plot to plots/strategy_comparison.png")

# Save results to JSON
results_for_json = {}
for key, result in results.items():
    # Remove large arrays for cleaner JSON
    result_copy = result.copy()
    result_copy.pop('all_times')
    result_copy.pop('all_attempts')
    results_for_json[key] = result_copy

with open(os.path.join(data_dir,'strategy_benchmark_results.json'), 'w') as f:
    json.dump(results_for_json, f, indent=2)
print(f"Saved results to {data_dir}/strategy_benchmark_results.json")

# Print winner
print("\n" + "="*80)
best_strategy = min(results.items(), key=lambda x: x[1]['mean_time'])
print(f"🏆 WINNER: {best_strategy[1]['strategy']}")
print(f"   Expected time: {format_time(best_strategy[1]['mean_time'])}")
print(f"   Expected attempts: {best_strategy[1]['mean_attempts']:.0f}")
print(f"   Actual time: {format_time(sum(model_params[room]['total_time'] for room in room_names))}")
print(f"   Actual attempts: {sum(model_params[room]['total_attempts'] for room in room_names)}")
print("="*80)