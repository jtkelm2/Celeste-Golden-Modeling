#!/usr/bin/env python3
"""
Celeste Run Strategy Simulator
Tests different practice strategies using Monte Carlo simulation
"""

from abc import ABC, abstractmethod
import json
import numpy as np
from scipy.special import expit
from typing import Dict, List, Tuple
from collections import deque


class RunStrategy(ABC):
    """Abstract base class for practice strategies"""
    
    def __init__(self):
        self.state = self.initialize_state()

    @abstractmethod
    def name(self) -> str:
        """Name for the strategy"""
        pass
    
    @abstractmethod
    def initialize_state(self) -> dict:
        """Return initial internal state"""
        pass
    
    @abstractmethod
    def get_next_room(self) -> str:
        """Return room name to attempt next"""
        pass
    
    @abstractmethod
    def update_state(self, success: bool):
        """Update internal state after attempting get_next_room()"""
        pass


class NaiveGrind(RunStrategy):
    """
    Repeatedly attempt the full run from start; reset to room 0 on any death.
    Hypothesis: overtrains early levels, undertrains later levels.
    """
    
    def __init__(self, room_names: List[str]):
        self.room_names = room_names
        super().__init__()
    
    def name(self) -> str:
        return "Naive Grind"
    
    def initialize_state(self) -> dict:
        return {
            'current_room_idx': 0
        }
    
    def get_next_room(self) -> str:
        return self.room_names[self.state['current_room_idx']]
    
    def update_state(self, success: bool):
        if success:
            self.state['current_room_idx'] += 1
        else:
            self.state['current_room_idx'] = 0


class CyclicGrind(RunStrategy):
    """
    Repeatedly attempt levels in order; never reset on death.
    """
    
    def __init__(self, room_names: List[str]):
        self.room_names = room_names
        super().__init__()
    
    def name(self) -> str:
        return "Cyclic Grind"
    
    def initialize_state(self) -> dict:
        return {
            'current_room_idx': 0
        }
    
    def get_next_room(self) -> str:
        return self.room_names[self.state['current_room_idx']]
    
    def update_state(self, success: bool):
        # Always advance to next room (cycle)
        self.state['current_room_idx'] = (self.state['current_room_idx'] + 1) % len(self.room_names)


class BackwardLearning(RunStrategy):
    """
    Start with last room, master it, then add previous room, etc.
    - Start with room N-1, practice until one success
    - Then practice rooms N-2 to N-1 until one streak
    - Then practice rooms N-3 to N-1 until one streak
    - Continue until 0 to N-1 is achieved in a streak
    """
    
    def __init__(self, room_names: List[str]):
        self.room_names = room_names
        super().__init__()
    
    def name(self) -> str:
        return "Backward Learning"
    
    def initialize_state(self) -> dict:
        return {
            'target_start_room': len(self.room_names) - 1,  # Start with last room only
            'current_position': len(self.room_names) - 1,   # Currently at last room
        }
    
    def get_next_room(self) -> str:
        return self.room_names[self.state['current_position']]
    
    def update_state(self, success: bool):
        if success:
            # Advance to next room in current sequence
            self.state['current_position'] += 1
            
            # If we've completed the current sequence (reached end)
            if self.state['current_position'] >= len(self.room_names):
                # Expand the sequence to include one more room at the start
                self.state['target_start_room'] -= 1
                self.state['current_position'] = self.state['target_start_room']
        else:
            # Death: restart current sequence
            self.state['current_position'] = self.state['target_start_room']


class BackwardChunkLearning(RunStrategy):
    """
    Like backward learning but with customizable chunks.
    Practice chunks from end to beginning.
    """
    
    def __init__(self, room_names: List[str], chunks: List[List[str]], subtitle: str | None = None):
        """
        chunks: list of lists of room names, e.g.:
                [['19'], ['18', '19'], ['17', '18', '19'], ...]
                or custom chunks like [['17', '18', '19'], ['14', '15', '16'], ...]
        """
        self.room_names = room_names
        self.chunks = chunks
        self.subtitle = subtitle
        super().__init__()
    
    def name(self) -> str:
        if self.subtitle:
          return f"Backward Chunk Learning ({self.subtitle})"
        else:
          return "Backward Chunk Learning"
    
    def initialize_state(self) -> dict:
        return {
            'current_chunk_idx': 0,
            'position_in_chunk': 0
        }
    
    def get_next_room(self) -> str:
        current_chunk = self.chunks[self.state['current_chunk_idx']]
        return current_chunk[self.state['position_in_chunk']]
    
    def update_state(self, success: bool):
        if success:
            self.state['position_in_chunk'] += 1
            current_chunk = self.chunks[self.state['current_chunk_idx']]
            
            # If we've completed the current chunk
            if self.state['position_in_chunk'] >= len(current_chunk):
                # Move to next chunk
                self.state['current_chunk_idx'] += 1
                self.state['position_in_chunk'] = 0
        else:
            # Death: restart current chunk
            self.state['position_in_chunk'] = 0

class Semiomniscient(RunStrategy):
    """
    Practice strategy that uses statistical models to decide whether to practice
    individual rooms or attempt full clears.
    
    For each room, computes:
    - Benefit: reduction in E_0 from improving that room's probability
    - Cost: expected time for one practice attempt
    
    Practices the room with highest (benefit - cost) if any room has benefit > cost.
    Otherwise, attempts full clears like NaiveGrind.
    """
    
    def __init__(self, room_names, model_parameters):
        """
        Args:
            room_names: List of room names in order
            model_parameters: Dict mapping room name to dict with keys:
                - 'beta_0', 'beta_1': logistic regression parameters
                - 'time': expected time for successful attempt
        """
        self.room_names = room_names
        self.models = model_parameters
        super().__init__()
        self.current_probs = self._compute_current_probabilities()
        self.current_E0 = self._compute_E0(self.current_probs)
    
    def name(self) -> str:
        return "Semiomniscient"

    def initialize_state(self):
        return {
            'attempt_counts': {room: 0 for room in self.room_names},
            'mode': 'training',  # 'training' or 'full_clear'
            'current_room_idx': 0,  # for full_clear mode
            'last_room': None,  # track which room was just attempted
        }
    
    def _compute_current_probabilities(self):
        """Compute current probabilities for next attempt at each room"""
        probs = {}
        for room in self.room_names:
            n = self.state['attempt_counts'][room]
            beta_0 = self.models[room]['beta_0']
            beta_1 = self.models[room]['beta_1']
            linear_pred = beta_0 + beta_1 * n
            probs[room] = expit(linear_pred)
        return probs
    
    def _compute_E0(self, probs):
        """
        Compute expected time to full clear given probabilities.
        E_0 = (1/P) * sum_j [a_j * prod_{k<j} p_k]
        """
        P = 1.0
        for room in self.room_names:
            P *= probs[room]
        
        total = 0.0
        prod_prev = 1.0
        for room in self.room_names:
            p_i = probs[room]
            t_i = self.models[room]['time']
            a_i = t_i * (1 + p_i) / 2
            total += a_i * prod_prev
            prod_prev *= p_i
        
        return total / P
    
    def _compute_benefit_cost(self, room):
        """
        Compute benefit and cost of practicing given room once more.
        
        Returns:
            (benefit, cost) where:
            - benefit: reduction in E_0 from the practice
            - cost: expected time for the practice attempt
        """
        
        n = self.state['attempt_counts'][room]
        beta_0 = self.models[room]['beta_0']
        beta_1 = self.models[room]['beta_1']
        linear_pred_after = beta_0 + beta_1 * (n + 1)
        p_new = expit(linear_pred_after)
        
        new_probs = self.current_probs.copy()
        new_probs[room] = p_new
        new_E0 = self._compute_E0(new_probs)
        
        benefit = self.current_E0 - new_E0
        
        p_current = self.current_probs[room]
        t_i = self.models[room]['time']
        cost = t_i * (1 + p_current) / 2
        
        return benefit, cost
    
    def get_next_room(self):
        if self.state['mode'] == 'training':
            
            best_room = None
            best_net_value = 0
            
            for room in self.room_names:
                benefit, cost = self._compute_benefit_cost(room)
                net_value = benefit - cost
                
                if net_value > best_net_value:
                    best_net_value = net_value
                    best_room = room
            
            if best_room is not None:
                self.state['last_room'] = best_room
                return best_room
            else:
                self.state['mode'] = 'full_clear'
                self.state['current_room_idx'] = 0
                self.state['last_room'] = self.room_names[0]
                return self.room_names[0]
        else:
            room = self.room_names[self.state['current_room_idx']]
            self.state['last_room'] = room
            return room
    
    def update_state(self, success):
        room = self.state['last_room']
        self.state['attempt_counts'][room] += 1
        
        if self.state['mode'] == 'full_clear':
            if success:
                self.state['current_room_idx'] += 1
            else:
                self.state['current_room_idx'] = 0
        
        self.current_probs = self._compute_current_probabilities()
        self.current_E0 = self._compute_E0(self.current_probs)

class RunSimulator:
    """Simulates room attempts using fitted probability models"""
    
    def __init__(self, model_parameters: dict):
        """
        model_parameters: dict with room names as keys, each containing:
            - beta_0, beta_1: logistic regression coefficients
            - time: expected time for successful attempt (in seconds)
        """
        self.models = model_parameters
        self.room_names = sorted(model_parameters.keys())
        self.reset()
    
    def reset(self):
        """Reset attempt counters"""
        self.room_attempt_counts = {room: 0 for room in self.room_names}
        self.room_time_spent = {room: 0.0 for room in self.room_names}
        self.attempt_history = deque(maxlen=len(self.room_names))
    
    def attempt_room(self, room: str) -> Tuple[bool, float]:
        """
        Attempt a room using logistic model
        Returns: (success: bool, time_elapsed: float)
        """
        # Increment attempt counter
        attempt_num = self.room_attempt_counts[room]
        self.room_attempt_counts[room] += 1
        
        # Get model parameters
        model = self.models[room]
        beta_0 = model['beta_0']
        beta_1 = model['beta_1']
        time_full = model['time']
        
        # Calculate probability of success
        linear_pred = beta_0 + beta_1 * attempt_num
        prob_success = expit(linear_pred)
        
        # Monte Carlo: sample success/failure
        success = np.random.random() < prob_success
        
        # Calculate time elapsed
        time_elapsed = time_full if success else time_full / 2
        self.room_time_spent[room] += time_elapsed
        
        # Record attempt
        self.attempt_history.append((room, success))
        
        return success, time_elapsed
    
    def is_complete(self) -> bool:
        """
        Check if a full run (all rooms in order with zero deaths) was just completed
        """
        # Need exactly N successful attempts in history
        if len(self.attempt_history) < len(self.room_names):
            return False
        
        # Check last N attempts
        recent_attempts = list(self.attempt_history)[-len(self.room_names):]
        
        # All must be successful
        if not all(success for _, success in recent_attempts):
            return False
        
        # Must be in correct order
        expected_sequence = self.room_names
        actual_sequence = [room for room, _ in recent_attempts]
        
        return actual_sequence == expected_sequence
    
    def simulate_until_completion(self, strategy: RunStrategy) -> dict:
        """
        Run strategy until full run achieved
        Returns statistics about the run
        """
        self.reset()
        total_attempts = 0
        total_time = 0.0
        
        max_iterations = 1000000  # Safety limit
        
        for _ in range(max_iterations):
            # Get next room from strategy
            room = strategy.get_next_room()
            
            # Attempt the room
            success, time_elapsed = self.attempt_room(room)
            total_attempts += 1
            total_time += time_elapsed
            
            # Update strategy state
            strategy.update_state(success)
            
            # Check if full run is complete
            if self.is_complete():
                return {
                    'total_attempts': total_attempts,
                    'total_time': total_time,
                    'attempts_per_room': dict(self.room_attempt_counts),
                    'time_per_room': dict(self.room_time_spent)
                }
        
        # Should never reach here
        raise RuntimeError(f"Failed to complete after {max_iterations} iterations")


def monte_carlo_benchmark(
    strategy_class,
    strategy_args: tuple,
    model_parameters: dict,
    n_simulations: int = 1000
) -> dict:
    """
    Run Monte Carlo simulation of a strategy
    
    Args:
        strategy_class: The strategy class to test
        strategy_args: Arguments to pass to strategy constructor
        model_parameters: Model parameters for simulation
        n_simulations: Number of simulations to run
    
    Returns:
        Dictionary with statistics
    """
    times = []
    attempts = []
    name = strategy_class(*strategy_args).name()
    
    print(f"Running {n_simulations} simulations of {name}...")
    
    for i in range(n_simulations):
        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{n_simulations} simulations...")
        
        # Create fresh strategy and simulator
        strategy = strategy_class(*strategy_args)
        simulator = RunSimulator(model_parameters)
        
        # Run simulation
        result = simulator.simulate_until_completion(strategy)
        
        times.append(result['total_time'])
        attempts.append(result['total_attempts'])
    
    times = np.array(times)
    attempts = np.array(attempts)
    
    return {
        'strategy': name,
        'n_simulations': n_simulations,
        'mean_time': np.mean(times),
        'median_time': np.median(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'mean_attempts': np.mean(attempts),
        'median_attempts': np.median(attempts),
        'std_attempts': np.std(attempts),
        'percentiles_time': {
            'p25': np.percentile(times, 25),
            'p50': np.percentile(times, 50),
            'p75': np.percentile(times, 75),
            'p90': np.percentile(times, 90),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99)
        },
        'all_times': times.tolist(),
        'all_attempts': attempts.tolist()
    }


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


if __name__ == "__main__":
    # This would be run from a separate script
    print("Run strategy simulator module loaded successfully.")
    print("\nAvailable strategies:")
    print("  - NaiveGrind")
    print("  - CyclicGrind")
    print("  - BackwardLearning")
    print("  - BackwardChunkLearning")