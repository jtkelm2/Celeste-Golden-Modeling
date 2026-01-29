#!/usr/bin/env python3
"""
Monte Carlo simulation for Celeste practice strategies.
"""

import numpy as np
from typing import Dict, List, Type
from models import RoomModels
from strategies import Strategy


def simulate_once(
    strategy: Strategy,
    models: RoomModels,
    max_iterations: int = 1_000_000
) -> Dict:
    """
    Run a strategy until completion (full deathless run).
    
    Returns:
        Dict with total_time, attempts_per_room
    """
    attempt_counts = {room: 0 for room in models.room_names}
    total_time = 0.0
    
    # Track consecutive successes for completion detection
    consecutive_rooms = []
    target_sequence = models.room_names
    n_rooms = len(target_sequence)
    
    for _ in range(max_iterations):
        room = strategy.get_next_room()
        n = attempt_counts[room]
        attempt_counts[room] += 1
        
        prob = models.success_prob(room, n)
        success = np.random.random() < prob
        total_time += models.attempt_time(room, success)
        
        strategy.update(success)
        
        # Track completion
        if success:
            consecutive_rooms.append(room)
            if len(consecutive_rooms) >= n_rooms:
                if consecutive_rooms[-n_rooms:] == target_sequence:
                    return {
                        'total_time': total_time,
                        'attempts_per_room': attempt_counts
                    }
        else:
            consecutive_rooms = []
    
    raise RuntimeError(f"Failed to complete after {max_iterations} iterations")


def benchmark(
    strategy_class: Type[Strategy],
    strategy_args: tuple,
    models: RoomModels,
    n_simulations: int
) -> Dict:
    """
    Run Monte Carlo benchmark of a strategy.
    
    Returns:
        Dict with mean_time, std_time, room_attempts (mean/std per room)
    """
    times = []
    room_attempts = {room: [] for room in models.room_names}
    n_simulations_checkpoint = n_simulations // 10
    
    for i in range(n_simulations):
        if i > 0 and i % n_simulations_checkpoint == 0:
            print(f"  Completed {i}/{n_simulations} simulations...")

        strategy = strategy_class(*strategy_args)
        result = simulate_once(strategy, models)
        
        times.append(result['total_time'])
        for room, count in result['attempts_per_room'].items():
            room_attempts[room].append(count)
    
    times = np.array(times)
    
    return {
        'strategy': strategy_class(*strategy_args).name,
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'all_times': times.tolist(),
        'room_attempts': {
            room: {
                'mean': float(np.mean(counts)),
                'std': float(np.std(counts))
            }
            for room, counts in room_attempts.items()
        }
    }