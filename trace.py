#!/usr/bin/env python3
"""
Trace and visualize a single simulation run of a given strategy.

Usage:
    python trace.py [--strategy TYPE] [--seed N] [--param KEY=VALUE ...]
                    [--attempts PATH] [--times PATH] [--output-dir DIR]

Strategy types: naive_grind, cyclic_grind, backward_learning,
                windowed_practice, semiomniscient, semiomniscient_online

Examples:
    python trace.py --strategy semiomniscient_online --seed 42
    python trace.py --strategy semiomniscient_online --param min_attempts_for_fit=5 --seed 42
    python trace.py --strategy semiomniscient --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.special import expit

from models import RoomModels
from strategies import (
    NaiveGrind, CyclicGrind, BackwardLearning,
    WindowedPractice, Semiomniscient, SemiomniscientOnline,
)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Step:
    step: int
    room: str
    room_idx: int
    success: bool
    # Mode the strategy was in when it picked this room (set by get_next_room)
    mode: str                   # 'training' | 'full_clear'
    # Why the strategy picked this room (online strategy only; else 'n/a')
    reason: str                 # 'insufficient_data' | 'negative_lr' | 'cost_benefit'
                                # | 'full_clear_mid' | 'full_clear_start'
    run_id: int                 # increments each time a new full_clear run begins
    strategy_prob: float        # strategy's probability estimate for this room
    true_prob: float            # true model probability at this attempt number
    # Full snapshot of all room probabilities at the moment of decision
    all_probs: Dict[str, float] = field(default_factory=dict)
    # Confidence class per room (online only): 0=insufficient, 1=neg_lr, 2=confident
    confidence: Dict[str, int] = field(default_factory=dict)
    # Fitted (β₀, β₁) per room from the online model (online only)
    fitted_params: Dict[str, tuple] = field(default_factory=dict)
    # Marginal net benefit of one more practice attempt per room (online only)
    net_benefits: Dict[str, float] = field(default_factory=dict)
    # Strategy's current E₀ estimate (online only)
    E0_estimate: float = float('nan')
    # True E₀ computed from ground-truth model + current attempt counts
    true_E0: float = float('nan')


# ── Decision-reason inspector ─────────────────────────────────────────────────

def _decision_reason(strategy) -> str:
    """
    Determine why SemiomniscientOnline would pick its next room.
    Replicates the logic in get_next_room without side effects.
    Returns a human-readable reason string.
    """
    if not isinstance(strategy, SemiomniscientOnline):
        return 'n/a'

    if strategy.mode == 'full_clear' and strategy.current_idx > 0:
        return 'full_clear_mid'

    if strategy._find_priority_room(SemiomniscientOnline.INSUFFICIENT_DATA) is not None:
        return 'insufficient_data'

    if strategy._find_priority_room(SemiomniscientOnline.NEGATIVE_LEARNING_RATE) is not None:
        return 'negative_lr'

    if strategy._best_net_benefit_room()[0] is not None:
        return 'cost_benefit'

    return 'full_clear_start'


def _confidence_snapshot(strategy) -> Dict[str, int]:
    """Extract per-room confidence class from SemiomniscientOnline."""
    if not isinstance(strategy, SemiomniscientOnline):
        return {}
    return {room: strategy.fitted[room][2] for room in strategy.room_names}


def _fitted_params_snapshot(strategy) -> Dict[str, tuple]:
    """Extract fitted (β₀, β₁) per room from SemiomniscientOnline."""
    if not isinstance(strategy, SemiomniscientOnline):
        return {}
    return {room: strategy.fitted[room][:2] for room in strategy.room_names}


def _net_benefits_snapshot(strategy) -> Dict[str, float]:
    """
    Compute marginal net benefit of one more practice attempt for every room,
    using the strategy's current fitted model and cached probabilities.
    Mirrors the loop inside SemiomniscientOnline._best_net_benefit_room but
    returns all rooms rather than just the best.
    """
    if not isinstance(strategy, SemiomniscientOnline):
        return {}
    E0 = strategy.current_E0
    if not np.isfinite(E0):
        return {room: float('nan') for room in strategy.room_names}
    result = {}
    for room in strategy.room_names:
        b0, b1, _ = strategy.fitted[room]
        n = len(strategy.history[room])
        p_new = expit(b0 + b1 * (n + 1))
        new_probs = strategy.current_probs.copy()
        new_probs[room] = p_new
        benefit = E0 - strategy._compute_E0(new_probs)
        cost = strategy.times[room] * (1 + strategy.current_probs[room]) / 2
        result[room] = benefit - cost
    return result


def _compute_true_E0(models: RoomModels, attempt_counts: Dict[str, int]) -> float:
    """
    Compute the ground-truth E₀ (expected time to golden run) using the true
    model parameters and the current per-room attempt counts.
    Uses the same formula as SemiomniscientOnline._compute_E0.
    """
    probs = {room: models.success_prob(room, attempt_counts[room])
             for room in models.room_names}
    P = 1.0
    for p in probs.values():
        P *= p
    if P < 1e-15:
        return float('inf')
    total = 0.0
    prod_prev = 1.0
    for room in models.room_names:
        p = probs[room]
        t = models.rooms[room].time
        total += t * (1 + p) / 2 * prod_prev
        prod_prev *= p
    return total / P


# ── Core tracer ───────────────────────────────────────────────────────────────

def run_trace(
    strategy_class,
    strategy_args: tuple,
    models: RoomModels,
    seed: Optional[int] = None,
    max_iterations: int = 2_000_000,
) -> List[Step]:
    """
    Run one simulation and record full per-step state.

    Returns a list of Step objects, one per room attempt, ending when
    the golden run is achieved.
    """
    if seed is not None:
        np.random.seed(seed)

    strategy = strategy_class(*strategy_args)
    attempt_counts = {room: 0 for room in models.room_names}
    room_to_idx = {room: i for i, room in enumerate(models.room_names)}

    consecutive_rooms: List[str] = []
    n_rooms = len(models.room_names)

    steps: List[Step] = []
    run_id = 0

    for step_num in range(max_iterations):
        # Capture state BEFORE get_next_room (reflects what we're choosing from)
        reason = _decision_reason(strategy)
        all_probs = dict(getattr(strategy, 'current_probs', {}))
        conf = _confidence_snapshot(strategy)
        fitted = _fitted_params_snapshot(strategy)
        net_bens = _net_benefits_snapshot(strategy)
        E0_est = float(getattr(strategy, 'current_E0', float('nan')))
        true_e0 = _compute_true_E0(models, attempt_counts)

        room = strategy.get_next_room()

        # Mode AFTER get_next_room (reflects the decision that was made)
        mode = getattr(strategy, 'mode', 'n/a')

        # Generic run-start detection: any room-0 attempt in full_clear mode is a new run.
        # Works for all strategies that use mode='full_clear' (WindowedPractice, Semiomniscient,
        # SemiomniscientOnline) since all of them reset current_idx=0 on failure.
        if mode == 'full_clear' and room_to_idx.get(room, -1) == 0:
            run_id += 1

        n = attempt_counts[room]
        attempt_counts[room] += 1
        true_prob = models.success_prob(room, n)
        success = np.random.random() < true_prob

        strategy.update(success)

        steps.append(Step(
            step=step_num,
            room=room,
            room_idx=room_to_idx[room],
            success=success,
            mode=mode,
            reason=reason,
            run_id=run_id,
            strategy_prob=all_probs.get(room, float('nan')),
            true_prob=true_prob,
            all_probs=all_probs,
            confidence=conf,
            fitted_params=fitted,
            net_benefits=net_bens,
            E0_estimate=E0_est,
            true_E0=true_e0,
        ))

        if success:
            consecutive_rooms.append(room)
            if (len(consecutive_rooms) >= n_rooms
                    and consecutive_rooms[-n_rooms:] == models.room_names):
                break
        else:
            consecutive_rooms = []

    return steps


# ── Plots ─────────────────────────────────────────────────────────────────────

_REASON_COLORS = {
    'insufficient_data': '#e67e22',
    'negative_lr':       '#8e44ad',
    'cost_benefit':      '#2980b9',
    'full_clear_mid':    '#27ae60',
    'full_clear_start':  '#1abc9c',
    'n/a':               '#7f8c8d',
}

_REASON_LABELS = {
    'insufficient_data': 'Insufficient data',
    'negative_lr':       'Negative LR',
    'cost_benefit':      'Cost-benefit',
    'full_clear_mid':    'Full clear (mid-run)',
    'full_clear_start':  'Full clear (new run)',
    'n/a':               'N/A',
}

_CONF_COLORS = {
    0: '#e67e22',   # INSUFFICIENT_DATA
    1: '#8e44ad',   # NEGATIVE_LEARNING_RATE
    2: '#27ae60',   # CONFIDENT
}
_CONF_LABELS = {0: 'Insufficient data', 1: 'Negative LR', 2: 'Confident'}


def plot_attempt_timeline(steps: List[Step], models: RoomModels,
                          output_path: str, title: str = "Attempt Timeline"):
    """
    Scatter plot: x = step, y = room index, color = reason for choice.
    Successful attempts are filled circles; failures are crosses.
    Background shading distinguishes full-clear runs from training.
    """
    room_names = models.room_names
    n_rooms = len(room_names)

    xs = [s.step for s in steps]
    ys = [s.room_idx for s in steps]
    reasons = [s.reason for s in steps]
    successes = [s.success for s in steps]

    colors = [_REASON_COLORS.get(r, '#7f8c8d') for r in reasons]
    markers_s = [('o' if ok else 'x') for ok in successes]

    fig, (ax, ax_mode) = plt.subplots(
        2, 1, figsize=(18, 9),
        gridspec_kw={'height_ratios': [5, 1]},
        sharex=True,
    )
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # ── shade full_clear run bands ──────────────────────────────────────────
    in_fc = False
    fc_start = 0
    for s in steps:
        is_fc = s.mode == 'full_clear'
        if is_fc and not in_fc:
            fc_start = s.step
            in_fc = True
        elif not is_fc and in_fc:
            ax.axvspan(fc_start, s.step, alpha=0.06, color='#2ecc71', zorder=0)
            in_fc = False
    if in_fc:
        ax.axvspan(fc_start, steps[-1].step, alpha=0.06, color='#2ecc71', zorder=0)

    # ── scatter ─────────────────────────────────────────────────────────────
    for marker in ('o', 'x'):
        mask = [i for i, m in enumerate(markers_s) if m == marker]
        ax.scatter(
            [xs[i] for i in mask],
            [ys[i] for i in mask],
            c=[colors[i] for i in mask],
            marker=marker,
            s=18 if marker == 'o' else 25,
            alpha=0.7,
            linewidths=1.0,
            zorder=3,
        )

    ax.set_yticks(range(n_rooms))
    ax.set_yticklabels(room_names, fontsize=8)
    ax.set_ylabel('Room')
    ax.set_ylim(-0.5, n_rooms - 0.5)
    ax.grid(True, alpha=0.15, axis='both')

    # ── legend ───────────────────────────────────────────────────────────────
    reason_seen = set(reasons)
    legend_patches = [
        mpatches.Patch(color=_REASON_COLORS[r], label=_REASON_LABELS[r])
        for r in _REASON_COLORS if r in reason_seen
    ]
    legend_patches += [
        plt.Line2D([0], [0], marker='o', color='gray', linestyle='', markersize=6, label='Success'),
        plt.Line2D([0], [0], marker='x', color='gray', linestyle='', markersize=6, label='Failure'),
        mpatches.Patch(color='#2ecc71', alpha=0.3, label='Full-clear mode'),
    ]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=8, ncol=2)

    # ── mode strip ───────────────────────────────────────────────────────────
    mode_vals = [1 if s.mode == 'full_clear' else 0 for s in steps]
    ax_mode.fill_between(xs, mode_vals, step='mid', alpha=0.5, color='#2ecc71')
    ax_mode.set_yticks([0, 1])
    ax_mode.set_yticklabels(['Train', 'FC'], fontsize=8)
    ax_mode.set_xlabel('Attempt #')
    ax_mode.set_ylim(-0.05, 1.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_room_attempt_counts(steps: List[Step], models: RoomModels,
                             output_path: str, title: str = "Cumulative Attempts per Room"):
    """
    Stacked area showing how many attempts have been made on each room
    as the simulation progresses. Helps identify which rooms dominate.
    """
    room_names = models.room_names
    n_rooms = len(room_names)
    n_steps = len(steps)

    counts = np.zeros((n_steps, n_rooms), dtype=int)
    running = np.zeros(n_rooms, dtype=int)

    for i, s in enumerate(steps):
        running[s.room_idx] += 1
        counts[i] = running.copy()

    xs = [s.step for s in steps]
    cmap = plt.colormaps['tab20'](np.linspace(0, 1, n_rooms))

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    ax.stackplot(xs, counts.T, labels=room_names, colors=cmap, alpha=0.8)
    ax.set_xlabel('Attempt #')
    ax.set_ylabel('Cumulative attempts')
    ax.legend(loc='upper left', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_probability_estimates(steps: List[Step], models: RoomModels,
                               output_path: str, title: str = "Probability Estimates vs True"):
    """
    For each room: plot the strategy's fitted probability estimate (sampled
    each time that room is visited) against the true model probability.
    Useful for spotting under/over-estimation.
    Online-strategy-only (requires all_probs snapshots).
    """
    if not steps or not steps[0].all_probs:
        print("No probability snapshot data; skipping probability plot.")
        return

    room_names = models.room_names
    n_rooms = len(room_names)

    # For each room, collect (attempt_index, estimated_prob, true_prob) at each visit
    visit_data: Dict[str, List] = {r: [] for r in room_names}
    attempt_counts = {r: 0 for r in room_names}

    for s in steps:
        attempt_counts[s.room] += 1
        n = attempt_counts[s.room]
        est = s.all_probs.get(s.room, float('nan'))
        visit_data[s.room].append((n, est, s.true_prob))

    ncols = min(4, n_rooms)
    nrows = (n_rooms + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight='bold')

    for idx, room in enumerate(room_names):
        ax = axes[idx // ncols][idx % ncols]
        data = visit_data[room]
        if not data:
            ax.set_title(room, fontsize=9)
            continue
        ns, ests, trues = zip(*data)
        ax.plot(ns, ests, color='steelblue', linewidth=1.2, label='Estimated')
        ax.plot(ns, trues, color='firebrick', linewidth=1.2, linestyle='--', label='True')
        ax.set_ylim(0, 1)
        ax.set_title(room, fontsize=9)
        ax.set_xlabel('Attempt #', fontsize=7)
        ax.set_ylabel('P(success)', fontsize=7)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2)

    # Hide unused subplots
    for idx in range(n_rooms, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_probability_estimates_global(steps: List[Step], models: RoomModels,
                                      output_path: str,
                                      title: str = "Probability Estimates vs True (Global Timeline)"):
    """
    Like plot_probability_estimates but x-axis is the global attempt count.
    Periods where a room is not being visited appear as horizontal segments,
    making training focus and idle time immediately visible.
    Works for all strategies; estimated probability is shown only when available.
    """
    room_names = models.room_names
    n_rooms = len(room_names)
    has_est = bool(steps and steps[0].all_probs)

    xs = [s.step for s in steps]

    # Build per-room series over the full global timeline.
    # true_at_step: true model prob at the moment of each global step (pre-attempt).
    # est_at_step:  strategy's fitted prob for each room at each global step.
    attempt_counts = {r: 0 for r in room_names}
    true_at_step: Dict[str, List[float]] = {r: [] for r in room_names}
    est_at_step:  Dict[str, List[float]] = {r: [] for r in room_names}

    for s in steps:
        for room in room_names:
            true_at_step[room].append(models.success_prob(room, attempt_counts[room]))
            if has_est:
                est_at_step[room].append(s.all_probs.get(room, float('nan')))
        attempt_counts[s.room] += 1

    ncols = min(4, n_rooms)
    nrows = (n_rooms + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3 * nrows),
        squeeze=False, sharex=True,
    )
    fig.suptitle(title, fontsize=13, fontweight='bold')

    for idx, room in enumerate(room_names):
        ax = axes[idx // ncols][idx % ncols]
        if has_est:
            ax.step(xs, est_at_step[room], where='post',
                    color='steelblue', linewidth=1.2, label='Estimated')
        ax.step(xs, true_at_step[room], where='post',
                color='firebrick', linewidth=1.2, linestyle='--', label='True')
        ax.set_ylim(0, 1)
        ax.set_title(room, fontsize=9)
        ax.set_xlabel('Step', fontsize=7)
        ax.set_ylabel('P(success)', fontsize=7)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2)

    for idx in range(n_rooms, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_confidence_timeline(steps: List[Step], models: RoomModels,
                             output_path: str, title: str = "Room Confidence Classification"):
    """
    For SemiomniscientOnline: heatmap showing the confidence class
    (InsufficientData / NegativeLR / Confident) of each room over time.
    """
    if not steps or not steps[0].confidence:
        print("No confidence data; skipping confidence timeline.")
        return

    room_names = models.room_names
    n_rooms = len(room_names)
    n_steps = len(steps)

    # Build matrix: rows = rooms, cols = steps
    mat = np.zeros((n_rooms, n_steps), dtype=float)
    for col, s in enumerate(steps):
        for ridx, room in enumerate(room_names):
            mat[ridx, col] = s.confidence.get(room, 0)

    xs = [s.step for s in steps]

    fig, ax = plt.subplots(figsize=(16, max(4, n_rooms * 0.5)))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    from matplotlib.colors import BoundaryNorm, ListedColormap
    cmap = ListedColormap(['#e67e22', '#8e44ad', '#27ae60'])  # 0=insuff, 1=neg_lr, 2=confident
    norm = BoundaryNorm([0, 1, 2, 3], cmap.N)

    ax.imshow(
        mat, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest',
        extent=[xs[0], xs[-1], n_rooms - 0.5, -0.5],
    )

    ax.set_yticks(range(n_rooms))
    ax.set_yticklabels(room_names, fontsize=8)
    ax.set_xlabel('Attempt #')
    ax.set_ylabel('Room')

    patches = [mpatches.Patch(color=c, label=l)
               for c, l in zip(['#e67e22', '#8e44ad', '#27ae60'],
                                ['Insufficient data', 'Negative LR', 'Confident'])]
    ax.legend(handles=patches, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_full_clear_runs(steps: List[Step], models: RoomModels,
                         output_path: str, title: str = "Full-Clear Run Progress"):
    """
    Horizontal bars showing how far each full-clear run progressed
    before failing (or succeeding on the last run).
    Each bar spans from room 0 to the room where the run ended.
    """
    room_names = models.room_names
    n_rooms = len(room_names)

    # Collect per-run outcomes: for each run, which rooms were attempted in full_clear mode
    # and where it ended.
    runs: Dict[int, List[Step]] = {}
    for s in steps:
        if s.mode == 'full_clear':
            if s.run_id not in runs:
                runs[s.run_id] = []
            runs[s.run_id].append(s)

    if not runs:
        print("No full-clear runs found; skipping run-progress plot.")
        return

    run_ids = sorted(runs.keys())
    fig, ax = plt.subplots(figsize=(12, max(4, len(run_ids) * 0.35 + 2)))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    for plot_row, rid in enumerate(run_ids):
        run_steps = runs[rid]
        reached_rooms = [s.room_idx for s in run_steps]
        last_step = run_steps[-1]
        # Run succeeded if the last step in the run was a success on the last room
        success = last_step.success and last_step.room_idx == n_rooms - 1

        color = '#27ae60' if success else '#e74c3c'
        max_room = max(reached_rooms) + 1  # rooms 0..max_room-1 were reached
        ax.barh(plot_row, max_room, color=color, alpha=0.6, height=0.8)

    ax.set_yticks(range(len(run_ids)))
    ax.set_yticklabels([f'Run {r}' for r in run_ids], fontsize=7)
    ax.set_xticks(range(n_rooms))
    ax.set_xticklabels(room_names, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Rooms reached')
    ax.axvline(n_rooms, color='black', linestyle='--', linewidth=1, alpha=0.5)

    patches = [
        mpatches.Patch(color='#27ae60', alpha=0.7, label='Success'),
        mpatches.Patch(color='#e74c3c', alpha=0.7, label='Failure'),
    ]
    ax.legend(handles=patches, loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.2, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_beta_fits(steps: List[Step], models: RoomModels,
                   output_path: str, title: str = "β Fits vs Ground Truth"):
    """
    For each room: fitted β₀ and β₁ as a function of per-room attempt count,
    overlaid with the true parameter values as horizontal dashed lines.
    X-axis = number of times that room has been visited (compresses inactive periods).
    Only meaningful for strategies that populate fitted_params.
    """
    if not steps or not steps[0].fitted_params:
        print("No fitted_params data; skipping beta fits plot.")
        return

    room_names = models.room_names
    n_rooms = len(room_names)

    visit_data: Dict[str, List] = {r: [] for r in room_names}
    attempt_counts = {r: 0 for r in room_names}
    for s in steps:
        attempt_counts[s.room] += 1
        n = attempt_counts[s.room]
        b0, b1 = s.fitted_params.get(s.room, (float('nan'), float('nan')))
        visit_data[s.room].append((n, b0, b1))

    ncols = min(4, n_rooms)
    nrows = (n_rooms + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight='bold')

    for idx, room in enumerate(room_names):
        ax = axes[idx // ncols][idx % ncols]
        ax2 = ax.twinx()

        true_b0 = models.rooms[room].beta_0
        true_b1 = models.rooms[room].beta_1

        data = visit_data[room]
        if data:
            ns, b0s, b1s = zip(*data)
            ax.step(ns, b0s, where='post', color='steelblue', linewidth=1.2)
            ax2.step(ns, b1s, where='post', color='firebrick', linewidth=1.2)

        ax.axhline(true_b0, color='steelblue', linestyle='--', linewidth=1.0, alpha=0.7)
        ax2.axhline(true_b1, color='firebrick', linestyle='--', linewidth=1.0, alpha=0.7)

        ax.set_title(room, fontsize=9)
        ax.set_xlabel('Attempt #', fontsize=7)
        ax.set_ylabel('β₀', fontsize=7, color='steelblue')
        ax2.set_ylabel('β₁', fontsize=7, color='firebrick')
        ax.tick_params(axis='y', labelcolor='steelblue', labelsize=6)
        ax2.tick_params(axis='y', labelcolor='firebrick', labelsize=6)
        ax.tick_params(axis='x', labelsize=6)
        ax.grid(True, alpha=0.15)

    handles = [
        Line2D([0], [0], color='steelblue', linewidth=1.2, label='β₀ fitted'),
        Line2D([0], [0], color='steelblue', linestyle='--', linewidth=1.0, label='β₀ true'),
        Line2D([0], [0], color='firebrick', linewidth=1.2, label='β₁ fitted'),
        Line2D([0], [0], color='firebrick', linestyle='--', linewidth=1.0, label='β₁ true'),
    ]
    axes[0][0].legend(handles=handles, fontsize=6, loc='upper left')

    for idx in range(n_rooms, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_beta_fits_global(steps: List[Step], models: RoomModels,
                          output_path: str, title: str = "β Fits vs Ground Truth (Global Timeline)"):
    """
    For each room: step-function of the online-fitted β₀ and β₁ over time,
    overlaid with the true parameter values as horizontal dashed lines.
    Left y-axis = β₀, right y-axis = β₁.
    Only meaningful for strategies that populate fitted_params.
    """
    if not steps or not steps[0].fitted_params:
        print("No fitted_params data; skipping beta fits plot.")
        return

    room_names = models.room_names
    n_rooms = len(room_names)
    ncols = min(4, n_rooms)
    nrows = (n_rooms + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight='bold')

    xs = [s.step for s in steps]

    for idx, room in enumerate(room_names):
        ax = axes[idx // ncols][idx % ncols]
        ax2 = ax.twinx()

        true_b0 = models.rooms[room].beta_0
        true_b1 = models.rooms[room].beta_1

        fitted_b0 = [s.fitted_params.get(room, (float('nan'), float('nan')))[0] for s in steps]
        fitted_b1 = [s.fitted_params.get(room, (float('nan'), float('nan')))[1] for s in steps]

        ax.step(xs, fitted_b0, where='post', color='steelblue', linewidth=1.2, label='β₀ fit')
        ax.axhline(true_b0, color='steelblue', linestyle='--', linewidth=1.0, alpha=0.7, label='β₀ true')

        ax2.step(xs, fitted_b1, where='post', color='firebrick', linewidth=1.2, label='β₁ fit')
        ax2.axhline(true_b1, color='firebrick', linestyle='--', linewidth=1.0, alpha=0.7, label='β₁ true')

        ax.set_title(room, fontsize=9)
        ax.set_xlabel('Step', fontsize=7)
        ax.set_ylabel('β₀', fontsize=7, color='steelblue')
        ax2.set_ylabel('β₁', fontsize=7, color='firebrick')
        ax.tick_params(axis='y', labelcolor='steelblue', labelsize=6)
        ax2.tick_params(axis='y', labelcolor='firebrick', labelsize=6)
        ax.tick_params(axis='x', labelsize=6)
        ax.grid(True, alpha=0.15)

    # Legend on first subplot only
    handles = [
        Line2D([0], [0], color='steelblue', linewidth=1.2, label='β₀ fitted'),
        Line2D([0], [0], color='steelblue', linestyle='--', linewidth=1.0, label='β₀ true'),
        Line2D([0], [0], color='firebrick', linewidth=1.2, label='β₁ fitted'),
        Line2D([0], [0], color='firebrick', linestyle='--', linewidth=1.0, label='β₁ true'),
    ]
    axes[0][0].legend(handles=handles, fontsize=6, loc='upper left')

    for idx in range(n_rooms, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_net_benefits(steps: List[Step], models: RoomModels,
                      output_path: str, title: str = "Marginal Training Net Benefit per Room"):
    """
    Heatmap: x = step, y = room, colour = marginal net benefit of one more
    practice attempt (positive → train this room; negative → don't bother).
    Uses a diverging colormap centred at zero.
    Only meaningful for strategies that populate net_benefits.
    """
    if not steps or not steps[0].net_benefits:
        print("No net_benefits data; skipping net benefits plot.")
        return

    room_names = models.room_names
    n_rooms = len(room_names)
    n_steps = len(steps)

    mat = np.full((n_rooms, n_steps), float('nan'))
    for col, s in enumerate(steps):
        for ridx, room in enumerate(room_names):
            mat[ridx, col] = s.net_benefits.get(room, float('nan'))

    xs = [s.step for s in steps]
    abs_max = np.nanmax(np.abs(mat))
    if abs_max == 0 or not np.isfinite(abs_max):
        abs_max = 1.0

    fig, ax = plt.subplots(figsize=(16, max(4, n_rooms * 0.45)))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    from matplotlib.colors import SymLogNorm
    # Negative side: max detriment is one room's expected attempt cost (≈ room time).
    # Positive side: cap at 10^4 s so early-run huge-E₀ spikes don't steal all resolution.
    max_room_time = max(models.rooms[r].time for r in models.room_names)
    vmin = -max_room_time * 1.1
    vmax = 1e4
    linthresh = 1.0  # linear region ±1 s; log scaling above that
    norm = SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
    img = ax.imshow(
        mat, aspect='auto', cmap='RdBu_r', interpolation='nearest',
        norm=norm,
        extent=[xs[0], xs[-1], n_rooms - 0.5, -0.5],
    )
    plt.colorbar(img, ax=ax, label='Net benefit (seconds, symlog)', shrink=0.8)

    # Overlay a line showing which room was actually chosen each step
    # (the room attempted = the one with highest net benefit, or priority room)
    chosen_xs = [s.step for s in steps if s.mode == 'training']
    chosen_ys = [s.room_idx for s in steps if s.mode == 'training']
    if chosen_xs:
        ax.scatter(chosen_xs, chosen_ys, color='black', s=2, alpha=0.4, zorder=5,
                   label='Chosen (training)')
        ax.legend(fontsize=8, loc='upper right')

    ax.set_yticks(range(n_rooms))
    ax.set_yticklabels(room_names, fontsize=8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Room')
    ax.axhline(-0.5, color='white', linewidth=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_E0_over_time(steps: List[Step], models: RoomModels,
                      output_path: str, title: str = "Expected Time to Completion"):
    """
    Strategy's E₀ estimate and the ground-truth E₀ over the course of the run.
    Background shading shows training vs full-clear mode.
    Y-axis in hours.
    Only meaningful for strategies that populate E0_estimate / true_E0.
    """
    has_est  = any(np.isfinite(s.E0_estimate) for s in steps)
    has_true = any(np.isfinite(s.true_E0)     for s in steps)
    if not has_est and not has_true:
        print("No E₀ data; skipping E₀ plot.")
        return

    xs        = [s.step       for s in steps]
    est_hours = [s.E0_estimate / 3600 if np.isfinite(s.E0_estimate) else float('nan')
                 for s in steps]
    true_hours = [s.true_E0 / 3600   if np.isfinite(s.true_E0)    else float('nan')
                  for s in steps]

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # Mode background
    in_fc, fc_start = False, 0
    for s in steps:
        if s.mode == 'full_clear' and not in_fc:
            fc_start, in_fc = s.step, True
        elif s.mode != 'full_clear' and in_fc:
            ax.axvspan(fc_start, s.step, alpha=0.07, color='#2ecc71', zorder=0)
            in_fc = False
    if in_fc:
        ax.axvspan(fc_start, steps[-1].step, alpha=0.07, color='#2ecc71', zorder=0)

    if has_est:
        ax.plot(xs, est_hours, color='steelblue', linewidth=1.0, alpha=0.85,
                label="Strategy's E₀ estimate")
    if has_true:
        ax.plot(xs, true_hours, color='firebrick', linewidth=1.2, linestyle='--', alpha=0.85,
                label='True E₀')

    ax.set_xlabel('Step')
    ax.set_ylabel('Expected time to golden run (hours)')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, which='both')

    # Cap y-axis at a sensible value to prevent inf spikes from dominating
    finite_vals = [v for v in est_hours + true_hours if np.isfinite(v) and v > 0]
    if finite_vals:
        ax.set_ylim(bottom=min(finite_vals) * 0.9,
                    top=min(np.percentile(finite_vals, 99) * 1.2, max(finite_vals)))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_trace(steps: List[Step], models: RoomModels,
               output_dir: str = 'plots/trace', title_prefix: str = ''):
    """Generate all trace diagnostic plots into output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    def path(name):
        return os.path.join(output_dir, name)

    pfx = f'{title_prefix}: ' if title_prefix else ''

    plot_attempt_timeline(
        steps, models, path('timeline.png'),
        title=f'{pfx}Attempt Timeline',
    )
    plot_room_attempt_counts(
        steps, models, path('room_counts.png'),
        title=f'{pfx}Cumulative Attempts per Room',
    )
    plot_probability_estimates(
        steps, models, path('probability_estimates.png'),
        title=f'{pfx}Probability Estimates vs True',
    )
    plot_probability_estimates_global(
        steps, models, path('probability_estimates_global.png'),
        title=f'{pfx}Probability Estimates vs True (Global Timeline)',
    )
    plot_confidence_timeline(
        steps, models, path('confidence.png'),
        title=f'{pfx}Room Confidence Classification',
    )
    plot_full_clear_runs(
        steps, models, path('full_clear_runs.png'),
        title=f'{pfx}Full-Clear Run Progress',
    )
    plot_beta_fits(
        steps, models, path('beta_fits.png'),
        title=f'{pfx}β Fits vs Ground Truth',
    )
    plot_beta_fits_global(
        steps, models, path('beta_fits_global.png'),
        title=f'{pfx}β Fits vs Ground Truth (Global Timeline)',
    )
    plot_net_benefits(
        steps, models, path('net_benefits.png'),
        title=f'{pfx}Marginal Training Net Benefit per Room',
    )
    plot_E0_over_time(
        steps, models, path('E0_over_time.png'),
        title=f'{pfx}Expected Time to Completion',
    )


# ── Strategy factory (mirrors benchmark.py) ──────────────────────────────────

def _build_strategy(strategy_type: str, params: Dict, room_names: List[str], models: RoomModels):
    """Return (strategy_class, args_tuple) for the given type + params."""
    t = strategy_type
    if t == 'naive_grind':
        return NaiveGrind, (room_names,)
    elif t == 'cyclic_grind':
        return CyclicGrind, (room_names,)
    elif t == 'backward_learning':
        chunk = int(params.get('chunk_size', 1))
        return BackwardLearning, (room_names, chunk)
    elif t == 'windowed_practice':
        k = int(params.get('k', 5))
        return WindowedPractice, (room_names, k)
    elif t == 'semiomniscient':
        return Semiomniscient, (room_names, models)
    elif t == 'semiomniscient_online':
        min_fit = int(params.get('min_attempts_for_fit', 15))
        neg_thresh = float(params.get('neg_beta_threshold', 0.5))
        return SemiomniscientOnline, (room_names, models, min_fit, neg_thresh)
    else:
        raise ValueError(f"Unknown strategy type: {t!r}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Trace and visualize a single simulation run.'
    )
    parser.add_argument('--strategy', '-s', default='semiomniscient_online',
                        help='Strategy type (default: semiomniscient_online)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--param', '-p', action='append', default=[],
                        metavar='KEY=VALUE',
                        help='Strategy parameter override, e.g. --param k=5')
    parser.add_argument('--attempts', '-a', default='inputs/attempts.json',
                        help='Path to attempts JSON (default: inputs/attempts.json)')
    parser.add_argument('--times', '-t', default='inputs/times.json',
                        help='Path to times JSON (default: inputs/times.json)')
    parser.add_argument('--output-dir', '-o', default='plots/trace',
                        help='Output directory for plots (default: plots/trace)')

    args = parser.parse_args()

    for path in (args.attempts, args.times):
        if not os.path.exists(path):
            print(f"Error: file not found: {path}")
            sys.exit(1)

    # Parse --param overrides
    params: Dict[str, str] = {}
    for kv in args.param:
        if '=' not in kv:
            print(f"Error: --param must be KEY=VALUE, got: {kv!r}")
            sys.exit(1)
        k, v = kv.split('=', 1)
        params[k.strip()] = v.strip()

    # Load model parameters (from data/ if available, else run analysis inline)
    model_params_file = os.path.join('data', 'model_parameters.json')
    if os.path.exists(model_params_file):
        with open(model_params_file) as f:
            model_params = json.load(f)
    else:
        print("Model parameters not found; run --analyze first.")
        sys.exit(1)

    models = RoomModels(model_params)
    strategy_class, strategy_args = _build_strategy(
        args.strategy, params, models.room_names, models
    )

    strategy_instance = strategy_class(*strategy_args)
    title_prefix = strategy_instance.name

    print(f"Tracing: {title_prefix}")
    print(f"Seed: {args.seed}")
    print()

    steps = run_trace(strategy_class, strategy_args, models, seed=args.seed)

    # Summary stats
    total = len(steps)
    room_counts = {r: 0 for r in models.room_names}
    for s in steps:
        room_counts[s.room] += 1

    print(f"Total attempts: {total}")
    print()
    print("Attempts per room:")
    for room, count in room_counts.items():
        bar = '#' * (count * 40 // max(room_counts.values()))
        print(f"  {room:>8}: {count:>5}  {bar}")
    print()

    from models import format_time
    total_time = sum(
        models.attempt_time(s.room, s.success) for s in steps
    )
    print(f"Total simulated time: {format_time(total_time)}")
    print()

    plot_trace(steps, models, output_dir=args.output_dir, title_prefix=title_prefix)


if __name__ == '__main__':
    main()
