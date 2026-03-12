#!/usr/bin/env python3
"""
Generate synthetic room model parameters for benchmarking.

Each generated file has the same structure as parameters/model_parameters.json
but without total_attempts/total_time (which only exist for real gameplay data).

Usage:
    python synth.py --rooms 13 --p0-range 0.1 0.8 --time-range 5 35 --count 50
    python synth.py --rooms 13 --count 50 --job-name sweep1
"""

import json
import os
import argparse
import numpy as np
from scipy.special import logit


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


def generate_model(
    n_rooms: int,
    p0_range: tuple,
    beta1_range: tuple,
    time_range: tuple,
    rng: np.random.Generator,
) -> dict:
    """Generate a single synthetic model_params dict."""
    params = {}
    for i in range(n_rooms):
        p0 = float(rng.uniform(*p0_range))
        beta1 = float(rng.uniform(*beta1_range))
        time = float(rng.uniform(*time_range))
        params[f"r{i:02d}"] = {
            'beta_0': float(logit(p0)),
            'beta_1': beta1,
            'time': time,
        }
    return params


def generate_batch(
    n_rooms: int,
    p0_range: tuple,
    beta1_range: tuple,
    time_range: tuple,
    count: int,
    seed,
    output_dir: str,
) -> list:
    """
    Generate `count` synthetic models and save to output_dir.
    Returns list of saved file paths.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i in range(count):
        params = generate_model(n_rooms, p0_range, beta1_range, time_range, rng)
        fname = 'model_parameters.json' if count == 1 else f'model_{i:04d}.json'
        path = os.path.join(output_dir, fname)
        with open(path, 'w') as f:
            json.dump(params, f, indent=2)
        paths.append(path)
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate synthetic room model parameters.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--rooms', type=int, required=True,
                        help='Number of rooms per chapter')
    parser.add_argument('--p0-range', type=float, nargs=2, default=[0.2, 0.8],
                        metavar=('MIN', 'MAX'),
                        help='Initial success probability range')
    parser.add_argument('--beta1-range', type=float, nargs=2, default=[0.0, 0.15],
                        metavar=('MIN', 'MAX'),
                        help='Learning rate (beta_1) range')
    parser.add_argument('--time-range', type=float, nargs=2, default=[5.0, 30.0],
                        metavar=('MIN', 'MAX'),
                        help='Room duration range in seconds')
    parser.add_argument('--count', type=int, default=1,
                        help='Number of chapters to generate')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--output', default='parameters/synthetic',
                        help='Base output directory')
    parser.add_argument('--job-name', default=None, metavar='NAME',
                        help='Job subdirectory name (default: auto job_NNN)')
    args = parser.parse_args()

    job_name = args.job_name or _next_job_name(args.output)
    output_dir = os.path.join(args.output, job_name)

    paths = generate_batch(
        args.rooms,
        tuple(args.p0_range),
        tuple(args.beta1_range),
        tuple(args.time_range),
        args.count,
        args.seed,
        output_dir,
    )
    print(f"Generated {len(paths)} model file(s) in {output_dir}")
