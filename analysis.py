#!/usr/bin/env python3
"""
Fit logistic models to Celeste attempt data and generate visualizations.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from typing import Dict, List
from models import fit_logistic_model


def analyze_room(room_name: str, attempts: List[bool], time: float, plots_dir: str) -> dict:
    """
    Fit model for a single room and create visualization.
    
    Returns:
        Dict with beta_0, beta_1, time, total_attempts, total_time
    """
    n = len(attempts)
    t = np.arange(n)
    y = np.array([1 if a else 0 for a in attempts])
    
    # Fit model
    result = fit_logistic_model(attempts)
    beta_0, beta_1 = result['beta_0'], result['beta_1']
    
    # Predictions
    predicted_prob = expit(beta_0 + beta_1 * t)
    
    # Smoothed actual (rolling average)
    window = max(1, min(20, n // 5))
    smoothed = np.array([
        np.mean(y[max(0, i - window + 1):i + 1])
        for i in range(n)
    ])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.scatter(t + 1, y, alpha=0.3, color='steelblue', s=30, label='Raw Data')
    ax.plot(t + 1, smoothed, color='darkblue', linewidth=2, alpha=0.7,
            linestyle='--', label='Smoothed Actual')
    ax.plot(t + 1, predicted_prob, color='red', linewidth=2.5, label='Fitted Model')
    
    ax.set_xlabel('Attempt Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability of Survival', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    success_rate = np.mean(y)
    title = f'Room {room_name}: Survival Probability Over Time'
    subtitle = (f'β₀={beta_0:.3f}, β₁={beta_1:.6f} | '
                f'Attempts: {n} | Success rate: {success_rate*100:.1f}%')
    ax.set_title(f'{title}\n{subtitle}', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'room_{room_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate total time
    total_time = n * time * (1 + success_rate) / 2
    
    return {
        'beta_0': beta_0,
        'beta_1': beta_1,
        'time': time,
        'total_attempts': n,
        'total_time': total_time
    }


def run_analysis(attempts_path: str, times_path: str, data_dir: str, plots_dir: str):
    """
    Run full analysis on attempt data.
    
    Args:
        attempts_path: Path to attempts JSON
        times_path: Path to times JSON
        data_dir: Output directory for model parameters
        plots_dir: Output directory for plots
    """
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    with open(attempts_path, 'r') as f:
        attempts_data = json.load(f)
    
    with open(times_path, 'r') as f:
        times_data = json.load(f)
    
    print("=" * 60)
    print("CELESTE GAMEPLAY ANALYSIS")
    print("=" * 60)
    print()
    
    model_params = {}
    all_results = {}
    
    for room_name in sorted(attempts_data.keys()):
        print(f"Processing Room {room_name}...")
        result = analyze_room(
            room_name,
            attempts_data[room_name],
            times_data[room_name],
            plots_dir
        )
        model_params[room_name] = result
        all_results[room_name] = result
        
        if result['beta_1'] < 0:
            print(f"  WARNING: β₁ = {result['beta_1']:.6f} < 0 (no improvement)")
    
    # Summary plot
    print("\nCreating summary visualization...")
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(attempts_data)))
    
    for i, room_name in enumerate(sorted(attempts_data.keys())):
        n = len(attempts_data[room_name])
        t = np.arange(n)
        r = model_params[room_name]
        prob = expit(r['beta_0'] + r['beta_1'] * t)
        normalized_t = t / (n - 1) if n > 1 else t
        ax.plot(normalized_t, prob, linewidth=2, alpha=0.8,
                label=f'Room {room_name}', color=colors[i])
    
    ax.set_xlabel('Progress Through Room (normalized)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability of Survival', fontsize=12, fontweight='bold')
    ax.set_title('All Rooms: Learning Curves Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'summary_all_rooms.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summaries
    print()
    print("=" * 60)
    print("MODEL SUMMARIES")
    print("=" * 60)
    print()
    
    for room_name in sorted(model_params.keys()):
        r = model_params[room_name]
        print(f"ROOM {room_name}")
        print(f"  Attempts: {r['total_attempts']}")
        print(f"  Model: logit(P) = {r['beta_0']:.3f} + {r['beta_1']:.6f} × t")
        print()
    
    # Save parameters
    output_file = os.path.join(data_dir, 'model_parameters.json')
    with open(output_file, 'w') as f:
        json.dump(model_params, f, indent=2)
    
    print("=" * 60)
    print(f"Analysis complete! Plots saved to {plots_dir}/")
    print(f"Model parameters saved to {output_file}")
    print("=" * 60)
    
    return model_params