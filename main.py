#!/usr/bin/env python3
"""
Celeste Golden Run Analysis and Strategy Benchmarking

Usage:
    python main.py [--attempts PATH] [--times PATH] [--analyze] [--benchmark] [--simulations N]

If neither --analyze nor --benchmark is specified, both are run.
"""

import argparse
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
        '--simulations', '-n',
        type=int,
        default=1000,
        help='Number of Monte Carlo simulations for benchmarking (default: 1000)'
    )
    parser.add_argument(
        '--chunks',
        type=int,
        nargs='+',
        default=[7],
        help='Chunk sizes for backward learning variants (default: 7)'
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not os.path.exists(args.attempts):
        print(f"Error: Attempts file not found: {args.attempts}")
        sys.exit(1)
    if not os.path.exists(args.times):
        print(f"Error: Times file not found: {args.times}")
        sys.exit(1)
    
    # If neither flag specified, run both
    run_analyze = args.analyze or (not args.analyze and not args.benchmark)
    run_benchmark = args.benchmark or (not args.analyze and not args.benchmark)
    
    # Directory setup
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
    
    if run_benchmark:
        import json
        from benchmark import run_benchmark
        
        # Load model params if not already computed
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
            n_simulations=args.simulations,
            chunk_sizes=args.chunks
        )


if __name__ == '__main__':
    main()