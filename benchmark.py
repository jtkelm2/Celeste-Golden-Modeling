#!/usr/bin/env python3
"""
Benchmark strategies and save results. Plotting is handled separately by plot.py.
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
import multiprocessing
import os
from collections import defaultdict
from typing import Dict, List, Optional
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


def _expand_entries(entries: List[Dict]) -> List[Dict]:
    """
    Expand list-valued strategy parameters into multiple entries.

    E.g. k = [3, 5, 7] becomes three separate entries with k=3, k=5, k=7.
    Only one parameter per entry may be a list; multiple list params raise an error.
    """
    expanded = []
    for entry in entries:
        list_params = {
            k: v for k, v in entry.items()
            if k not in ('type', 'simulations') and isinstance(v, list)
        }
        if len(list_params) > 1:
            raise ValueError(
                f"Strategy entry ({entry['type']}) has multiple list-valued params "
                f"{list(list_params.keys())}; only one sweep dimension per entry is supported."
            )
        if not list_params:
            expanded.append(entry)
        else:
            param, values = next(iter(list_params.items()))
            for v in values:
                expanded.append({**entry, param: v})
    return expanded


def run_benchmark(
    model_params: Dict,
    results_dir: str,
    config_path: str = 'benchmark_config.toml',
) -> Dict:
    """
    Run benchmarks on configured strategies and save results to JSON.

    Args:
        model_params: Dict of room model parameters
        results_dir:  Output directory for results JSON
        config_path:  Path to TOML benchmark config (default: benchmark_config.toml)

    Returns:
        The saved results dict (same structure as the JSON).
    """
    os.makedirs(results_dir, exist_ok=True)

    entries = _expand_entries(_load_config(config_path))
    models = RoomModels(model_params)
    room_names = models.room_names

    # Actual results from data — optional, absent for synthetic models
    has_actual = all(
        'total_time' in model_params[r] and 'total_attempts' in model_params[r]
        for r in room_names
    )
    actual_time = sum(model_params[r]['total_time'] for r in room_names) if has_actual else None
    actual_attempts = {r: model_params[r]['total_attempts'] for r in room_names} if has_actual else None

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

    # Print results summary
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print()

    for r in results:
        print(f"{r['strategy']}:")
        print(f"  Mean time: {format_time(r['mean_time'])} ± {format_time(r['std_time'])}")
        print()

    if actual_time is not None:
        print(f"Actual results:")
        print(f"  Time: {format_time(actual_time)}")
        print()

    best = min(results, key=lambda r: r['mean_time'])
    print("=" * 60)
    print(f"🏆 WINNER: {best['strategy']}")
    print(f"   Expected: {format_time(best['mean_time'])}")
    if actual_time is not None:
        print(f"   Actual:   {format_time(actual_time)}")
    print("=" * 60)

    # Save results
    save_results: Dict = {
        'room_names': room_names,
        'strategies': [
            {
                'strategy': r['strategy'],
                'mean_time': r['mean_time'],
                'std_time': r['std_time'],
                'room_attempts': r['room_attempts'],
                'all_times': r['all_times'],
                'entry': r['entry'],
            }
            for r in results
        ],
    }
    if actual_time is not None:
        save_results['actual'] = {
            'total_time': actual_time,
            'attempts_per_room': actual_attempts,
        }

    results_file = os.path.join(results_dir, 'benchmark_results.json')
    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"Saved results to {results_file}")

    return save_results


# ── Batch / parallel ──────────────────────────────────────────────────────────

def _run_batch_cell(job):
    """Worker function for parallel batch benchmarking (must be module-level for pickling)."""
    model_path, model_params, entry = job
    try:
        models = RoomModels(model_params)
        room_names = models.room_names
        cls, args = build_strategy(entry['type'], entry, room_names, models)
        result = benchmark(cls, args, models, entry['simulations'], verbose=False)
        result['entry'] = entry
        return model_path, result
    except Exception as e:
        return model_path, {'error': str(e), 'entry': entry}


def run_batch(
    model_paths: List[str],
    results_dir: str = 'results',
    config_path: str = 'benchmark_config.toml',
    workers: Optional[int] = None,
) -> List[str]:
    """
    Run benchmarks for multiple model files in parallel.
    Saves one <stem>_benchmark.json per model into results_dir.

    Args:
        model_paths:  List of paths to model_parameters JSON files.
        results_dir:  Output directory for result JSONs.
        config_path:  Path to TOML benchmark config.
        workers:      Worker process count (default: CPU count).

    Returns:
        List of paths to the saved result files.
    """
    os.makedirs(results_dir, exist_ok=True)
    entries = _expand_entries(_load_config(config_path))

    loaded = []
    for path in model_paths:
        with open(path) as f:
            params = json.load(f)
        loaded.append((path, params))

    jobs = [(path, params, entry) for path, params in loaded for entry in entries]
    total = len(jobs)
    n_workers = workers or os.cpu_count()

    print(f"Batch: {len(loaded)} models × {len(entries)} configs = {total} jobs on {n_workers} workers")

    results_by_path: Dict[str, List] = defaultdict(list)
    with multiprocessing.Pool(n_workers) as pool:
        checkpoint = max(1, total // 20)
        for done, (model_path, result) in enumerate(
            pool.imap_unordered(_run_batch_cell, jobs), 1
        ):
            results_by_path[model_path].append(result)
            if done % checkpoint == 0 or done == total:
                print(f"  {done}/{total} jobs complete")

    result_paths = []
    for path, params in loaded:
        room_names = sorted(params.keys())
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(results_dir, stem + '_benchmark.json')
        results = results_by_path[path]
        save: Dict = {
            'room_names': room_names,
            'strategies': [
                {
                    'strategy': r['strategy'],
                    'mean_time': r['mean_time'],
                    'std_time': r['std_time'],
                    'room_attempts': r['room_attempts'],
                    'all_times': r['all_times'],
                    'entry': r['entry'],
                }
                for r in results
                if 'error' not in r
            ],
        }
        errors = [r for r in results if 'error' in r]
        if errors:
            save['errors'] = errors
        with open(out_path, 'w') as f:
            json.dump(save, f, indent=2)
        result_paths.append(out_path)

    print(f"Saved {len(loaded)} result file(s) alongside model files.")
    return result_paths
