#!/usr/bin/env python3
"""
Celeste Gameplay Analysis - Probability of Survival by Room
Fits logistic regression models to estimate survival probability over time
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit  # logistic function
import os

input_dir = 'inputs'
data_dir = 'data'
plots_dir = 'plots'

# Create directories
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Read the data
with open(os.path.join(input_dir, 'attempts.json'), 'r') as f:
    data = json.load(f)

with open(os.path.join(input_dir, 'times.json'), 'r') as f:
    times = json.load(f)

def logistic_regression_mle(attempts):
    """
    Fit logistic regression using maximum likelihood estimation
    Model: P(survival) = 1 / (1 + exp(-(β₀ + β₁*t)))
    """
    # Prepare data
    n = len(attempts)
    t = np.arange(1, n + 1)
    y = np.array([1 if a else 0 for a in attempts])
    
    # Negative log-likelihood function
    def neg_log_likelihood(params):
        beta_0, beta_1 = params
        linear_pred = beta_0 + beta_1 * t
        p = expit(linear_pred)  # logistic function
        
        # Avoid log(0) by clipping probabilities
        p = np.clip(p, 1e-10, 1 - 1e-10)
        
        # Negative log-likelihood
        nll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        return nll
    
    # Initial guess
    initial_params = [0, 0]
    
    # Minimize negative log-likelihood
    result = minimize(neg_log_likelihood, initial_params, method='BFGS')
    
    beta_0, beta_1 = result.x
    
    # Calculate predictions
    linear_pred = beta_0 + beta_1 * t
    predicted_prob = expit(linear_pred)
    
    # Calculate smoothed actual success rate (rolling average)
    window_size = min(20, n // 5)
    if window_size < 1:
        window_size = 1
    smoothed_actual = np.array([
        np.mean(y[max(0, i - window_size + 1):i + 1]) 
        for i in range(n)
    ])
    
    return {
        'beta_0': beta_0,
        'beta_1': beta_1,
        't': t,
        'y': y,
        'predicted_prob': predicted_prob,
        'smoothed_actual': smoothed_actual,
        'initial_prob': expit(beta_0 + beta_1 * 1),
        'final_prob': expit(beta_0 + beta_1 * n)
    }

def analyze_room(room_name, attempts):
    """Analyze a single room and create visualization"""
    
    print(f"Processing Room {room_name}...")
    
    # Fit model
    results = logistic_regression_mle(attempts)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot raw data
    ax.scatter(results['t'], results['y'], alpha=0.3, color='steelblue', 
               s=30, label='Raw Data', zorder=1)
    
    # Plot smoothed actual
    ax.plot(results['t'], results['smoothed_actual'], color='darkblue', 
            linewidth=2, alpha=0.7, linestyle='--', label='Smoothed Actual', zorder=2)
    
    # Plot fitted probability
    ax.plot(results['t'], results['predicted_prob'], color='red', 
            linewidth=2.5, label='Fitted Model (MLE)', zorder=3)
    
    # Styling
    ax.set_xlabel('Attempt Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability of Survival', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    # Title with model info
    title = f'Room {room_name}: Survival Probability Over Time'
    subtitle = (f'Model: P(survival) = 1/(1 + exp(-(β₀ + β₁ × t))) where β₀ = {results["beta_0"]:.3f}, β₁ = {results["beta_1"]:.6f}\n'
                f'Initial: {results["initial_prob"]*100:.1f}% → Final: {results["final_prob"]*100:.1f}% | '
                f'Total attempts: {len(attempts)} | Success rate: {np.mean(results["y"])*100:.1f}%')
    
    ax.set_title(title + '\n' + subtitle, fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save plot
    filename = f'{plots_dir}/room_{room_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

# Analyze all rooms
print("="*80)
print("CELESTE GAMEPLAY ANALYSIS")
print("="*80)
print()

all_results = {}
for room_name in sorted(data.keys()):
    attempts = data[room_name]
    all_results[room_name] = analyze_room(room_name, attempts)

# Create summary plot
print("\nCreating summary visualization...")

fig, ax = plt.subplots(figsize=(14, 8))

colors = plt.cm.viridis(np.linspace(0, 1, len(data)))

for i, (room_name, results) in enumerate(sorted(all_results.items())):
    # Normalize attempt numbers to percentage of total attempts for that room
    normalized_t = results['t'] / len(results['t'])
    ax.plot(normalized_t, results['predicted_prob'], 
            linewidth=2, alpha=0.8, label=f'Room {room_name}', color=colors[i])

ax.set_xlabel('Progress Through Room (normalized)', fontsize=12, fontweight='bold')
ax.set_ylabel('Probability of Survival', fontsize=12, fontweight='bold')
ax.set_title('All Rooms: Learning Curves Comparison\nFitted probability models showing improvement over time', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)

plt.tight_layout()
plt.savefig(f'{plots_dir}/summary_all_rooms.png', dpi=300, bbox_inches='tight')
plt.close()

# Print model summaries
print()
print("="*80)
print("MODEL SUMMARIES")
print("="*80)
print()

# Extract model parameters for each room
model_parameters = {}

for room_name in sorted(data.keys()):
    results = all_results[room_name]
    n_attempts = len(data[room_name])
    success_rate = np.mean(results['y'])

    time = times[room_name]
    total_time = n_attempts * time * (1 + success_rate) / 2
    
    print(f"ROOM {room_name}")
    print(f"  Total attempts: {n_attempts}")
    print(f"  Total time: {total_time:.0f}s")
    print(f"  Success rate: {success_rate * 100:.1f}%")
    print(f"  Model: logit(P) = {results['beta_0']:.3f} + {results['beta_1']:.6f} × t")
    print(f"  Initial probability: {results['initial_prob']*100:.1f}%")
    print(f"  Final probability: {results['final_prob']*100:.1f}%")
    improvement = (results['final_prob'] - results['initial_prob']) * 100
    print(f"  Improvement: {improvement:+.1f} percentage points")
    print()

    model_parameters[room_name] = {
        'beta_0': float(results['beta_0']),
        'beta_1': float(results['beta_1']),
        'time': times[room_name],
        'total_attempts': len(data[room_name]),
        'total_time': total_time
    }

for room_name in all_results.keys():
    beta = all_results[room_name]['beta_1']
    if beta < 0:
        print(f"WARNING: {room_name} has beta_1 = {beta} < 0")

# Save to JSON file
output_file = os.path.join(data_dir, 'model_parameters.json')
with open(output_file, 'w') as f:
    json.dump(model_parameters, f, indent=2)

print("="*80)
print(f"Analysis complete! Generated {len(data) + 1} plots in {plots_dir}/")
print("="*80)