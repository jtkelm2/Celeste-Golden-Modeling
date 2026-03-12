#!/usr/bin/env python3
"""
Celeste Golden Run Analysis and Strategy Benchmarking

Usage:
    python main.py [--analyze] [--benchmark] [--no-plot] [--job-name NAME]
    python main.py --benchmark --models "parameters/synthetic/job_000/*.json" [--workers N]

If neither --analyze nor --benchmark is specified, both are run.
--plot is enabled by default; pass --no-plot to skip plot generation.

Each benchmark or batch job writes into a unique subdirectory:
  results/job_NNN/   (or results/<job-name>/)
  plots/job_NNN/     (or plots/<job-name>/)

Strategy selection and simulation counts are controlled via benchmark_config.toml.
List-valued strategy parameters in the config are automatically expanded into sweeps.
"""

import argparse
import glob as glob_module
import os
import sys


def _next_job_name(base_dir: str) -> str:
    """Return the lowest available job_NNN name within base_dir."""
    existing = set()
    if os.path.isdir(base_dir):
        for name in os.listdir(base_dir):
            if name.startswith('job_') and name[4:].isdigit():
                existing.add(int(name[4:]))
    n = 0
    while n in existing:
        n += 1
    return f'job_{n:03d}'


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Celeste gameplay data and benchmark practice strategies.'
    )
    parser.add_argument(
        '--attempts', '-a',
        default='inputs/attempts.json',
        help='Path to attempts JSON file (default: inputs/attempts.json)'
    )
    parser.add_argument(
        '--times', '-t',
        default='inputs/times.json',
        help='Path to times JSON file (default: inputs/times.json)'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Run analysis to fit models and generate learning curve plots'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run strategy benchmarks'
    )
    parser.add_argument(
        '--plot',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Generate plots after benchmarking (default: enabled)'
    )
    parser.add_argument(
        '--job-name',
        default=None,
        metavar='NAME',
        help='Job subdirectory name for results and plots (default: auto job_NNN)'
    )
    parser.add_argument(
        '--benchmark-config',
        default='benchmark_config.toml',
        metavar='PATH',
        help='Path to benchmark TOML config (default: benchmark_config.toml)'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        metavar='PATH',
        help='Model parameter files for batch benchmarking (globs supported). '
             'Results are saved into results/<job>/.'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        metavar='N',
        help='Worker processes for batch benchmarking (default: CPU count)'
    )

    args = parser.parse_args()

    # ── Batch benchmark mode ──────────────────────────────────────────────────
    if args.models:
        if args.analyze:
            print("Error: --analyze and --models cannot be combined.")
            sys.exit(1)

        model_paths = []
        for pattern in args.models:
            expanded = sorted(glob_module.glob(pattern))
            if expanded:
                model_paths.extend(expanded)
            elif os.path.isfile(pattern):
                model_paths.append(pattern)
            else:
                print(f"Warning: no files matched: {pattern}")

        if not model_paths:
            print("Error: no model files found.")
            sys.exit(1)

        job_name = args.job_name or _next_job_name('results')
        results_dir = os.path.join('results', job_name)
        plots_base_dir = os.path.join('plots', job_name)

        from benchmark import run_batch
        result_paths = run_batch(
            model_paths,
            results_dir=results_dir,
            config_path=args.benchmark_config,
            workers=args.workers,
        )

        if args.plot:
            from plot import plot_batch
            plot_batch(result_paths, plots_base_dir=plots_base_dir)

        return

    # ── Single-model mode ─────────────────────────────────────────────────────
    if not os.path.exists(args.attempts):
        print(f"Error: Attempts file not found: {args.attempts}")
        sys.exit(1)
    if not os.path.exists(args.times):
        print(f"Error: Times file not found: {args.times}")
        sys.exit(1)

    # If neither flag specified, run both
    run_analyze = args.analyze or (not args.analyze and not args.benchmark)
    do_benchmark = args.benchmark or (not args.analyze and not args.benchmark)

    parameters_dir = 'parameters'
    analysis_plots_dir = 'plots/analysis'

    model_params = None

    if run_analyze:
        from analysis import run_analysis
        model_params = run_analysis(
            args.attempts,
            args.times,
            parameters_dir,
            analysis_plots_dir
        )
        print()

    if do_benchmark:
        import json
        from benchmark import run_benchmark

        if model_params is None:
            params_file = os.path.join(parameters_dir, 'model_parameters.json')
            if not os.path.exists(params_file):
                print("Error: Model parameters not found. Run --analyze first.")
                sys.exit(1)
            with open(params_file, 'r') as f:
                model_params = json.load(f)

        job_name = args.job_name or _next_job_name('results')
        results_dir = os.path.join('results', job_name)
        benchmark_plots_dir = os.path.join('plots', 'benchmark', job_name)

        run_benchmark(
            model_params,
            results_dir,
            config_path=args.benchmark_config,
        )

        if args.plot:
            from plot import plot_results
            results_file = os.path.join(results_dir, 'benchmark_results.json')
            plot_results(results_file, benchmark_plots_dir)


if __name__ == '__main__':
    main()
