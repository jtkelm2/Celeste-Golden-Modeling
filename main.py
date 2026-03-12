#!/usr/bin/env python3
"""
Celeste Golden Run Analysis and Strategy Benchmarking

Usage:
    python main.py [--attempts PATH] [--times PATH] [--analyze] [--benchmark]
                   [--benchmark-config PATH]
    python main.py --benchmark --models "data/synthetic/*.json" [--workers N]

If neither --analyze nor --benchmark is specified, both are run.
Strategy selection and simulation counts are controlled via benchmark_config.toml.
List-valued strategy parameters in the config are automatically expanded into sweeps.
"""

import argparse
import glob as glob_module
import os
import sys


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
             'Results are saved alongside each input file as <stem>_benchmark.json.'
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

        from benchmark import run_batch
        run_batch(model_paths, config_path=args.benchmark_config, workers=args.workers)
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

    data_dir = 'data'
    analysis_plots_dir = 'plots/analysis'
    benchmark_plots_dir = 'plots/benchmark'

    model_params = None

    if run_analyze:
        from analysis import run_analysis
        model_params = run_analysis(
            args.attempts,
            args.times,
            data_dir,
            analysis_plots_dir
        )
        print()

    if do_benchmark:
        import json
        from benchmark import run_benchmark

        if model_params is None:
            params_file = os.path.join(data_dir, 'model_parameters.json')
            if not os.path.exists(params_file):
                print("Error: Model parameters not found. Run --analyze first.")
                sys.exit(1)
            with open(params_file, 'r') as f:
                model_params = json.load(f)

        run_benchmark(
            model_params,
            benchmark_plots_dir,
            data_dir,
            config_path=args.benchmark_config,
        )


if __name__ == '__main__':
    main()