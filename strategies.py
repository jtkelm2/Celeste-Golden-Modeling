#!/usr/bin/env python3
"""
Practice strategies for Celeste golden runs.
"""

from abc import ABC, abstractmethod
from typing import List
from scipy.special import expit
from models import RoomModels


class Strategy(ABC):
    """Abstract base class for practice strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def get_next_room(self) -> str:
        """Return room name to attempt next."""
        pass
    
    @abstractmethod
    def update(self, success: bool):
        """Update internal state after attempt."""
        pass


class NaiveGrind(Strategy):
    """
    Repeatedly attempt full runs from start; reset to room 0 on any death.
    """
    
    def __init__(self, room_names: List[str]):
        self.room_names = room_names
        self.current_idx = 0
    
    @property
    def name(self) -> str:
        return "Naive Grind"
    
    def get_next_room(self) -> str:
        return self.room_names[self.current_idx]
    
    def update(self, success: bool):
        if success:
            self.current_idx += 1
            if self.current_idx >= len(self.room_names):
                self.current_idx = 0
        else:
            self.current_idx = 0


class CyclicGrind(Strategy):
    """
    Practice rooms in order, cycling back to start after last room.
    Never reset on death.
    """
    
    def __init__(self, room_names: List[str]):
        self.room_names = room_names
        self.current_idx = 0
    
    @property
    def name(self) -> str:
        return "Cyclic Grind"
    
    def get_next_room(self) -> str:
        return self.room_names[self.current_idx]
    
    def update(self, success: bool):
        self.current_idx = (self.current_idx + 1) % len(self.room_names)


class BackwardLearning(Strategy):
    """
    Start with last room(s), master the sequence, then prepend more rooms.
    
    With chunk_size=1: practice room N, then N-1 to N, then N-2 to N, etc.
    With chunk_size=k: practice last k rooms, then last 2k, etc.
    """
    
    def __init__(self, room_names: List[str], chunk_size: int = 1):
        self.room_names = room_names
        self.chunk_size = chunk_size
        n = len(room_names)
        # Start index of current practice sequence
        self.sequence_start = max(0, n - chunk_size)
        # Current position within the sequence
        self.current_idx = self.sequence_start
    
    @property
    def name(self) -> str:
        if self.chunk_size == 1:
            return "Backward Learning"
        return f"Backward Learning (chunk={self.chunk_size})"
    
    def get_next_room(self) -> str:
        return self.room_names[self.current_idx]
    
    def update(self, success: bool):
        if success:
            self.current_idx += 1
            # Completed current sequence?
            if self.current_idx >= len(self.room_names):
                # Expand sequence by chunk_size rooms
                self.sequence_start = max(0, self.sequence_start - self.chunk_size)
                self.current_idx = self.sequence_start
        else:
            # Reset to start of current sequence
            self.current_idx = self.sequence_start


class Semiomniscient(Strategy):
    """
    Uses fitted models to decide whether to practice individual rooms
    or attempt full clears.
    
    Computes benefit (reduction in expected clear time) vs cost (practice time)
    for each room, practices room with highest net benefit, or attempts
    full clears when no room has positive net benefit.
    """
    
    def __init__(self, room_names: List[str], models: RoomModels):
        self.room_names = room_names
        self.models = models
        self.attempt_counts = {room: 0 for room in room_names}
        self.mode = 'training'
        self.current_idx = 0
        self._last_room = None
        
        # Cache current state
        self._update_cache()
    
    @property
    def name(self) -> str:
        return "Semiomniscient"
    
    def _update_cache(self):
        """Recompute cached probabilities and expected time."""
        self.current_probs = {
            room: self.models.success_prob(room, self.attempt_counts[room])
            for room in self.room_names
        }
        self.current_E0 = self._compute_E0(self.current_probs)
    
    def _compute_E0(self, probs: dict) -> float:
        """
        Expected time to full clear given probabilities.
        E_0 = (1/P) * sum_j [a_j * prod_{k<j} p_k]
        where a_j = t_j * (1 + p_j) / 2
        """
        P = 1.0
        for room in self.room_names:
            P *= probs[room]
        
        total = 0.0
        prod_prev = 1.0
        for room in self.room_names:
            p = probs[room]
            t = self.models.rooms[room].time
            a = t * (1 + p) / 2
            total += a * prod_prev
            prod_prev *= p
        
        return total / P
    
    def _compute_benefit_cost(self, room: str) -> tuple:
        """
        Compute benefit and cost of one more practice attempt on room.
        
        Returns: (benefit, cost)
        """
        n = self.attempt_counts[room]
        m = self.models.rooms[room]
        
        # Probability after one more attempt
        p_new = expit(m.beta_0 + m.beta_1 * (n + 1))
        
        # Expected time with improved probability
        new_probs = self.current_probs.copy()
        new_probs[room] = p_new
        new_E0 = self._compute_E0(new_probs)
        
        benefit = self.current_E0 - new_E0
        cost = m.time * (1 + self.current_probs[room]) / 2
        
        return benefit, cost
    
    def get_next_room(self) -> str:
        if self.mode == 'training':
            best_room = None
            best_net = 10
            
            for room in self.room_names:
                benefit, cost = self._compute_benefit_cost(room)
                net = benefit - cost
                if net > best_net:
                    best_net = net
                    best_room = room
            
            if best_room is not None:
                self._last_room = best_room
                return best_room
            else:
                # Switch to full clear mode
                self.mode = 'full_clear'
                self.current_idx = 0
                self._last_room = self.room_names[0]
                return self.room_names[0]
        else:
            room = self.room_names[self.current_idx]
            self._last_room = room
            return room
    
    def update(self, success: bool):
        assert self._last_room is not None
        self.attempt_counts[self._last_room] += 1
        
        if self.mode == 'full_clear':
            if success:
                self.current_idx += 1
                if self.current_idx >= len(self.room_names):
                    self.current_idx = 0
            else:
                self.current_idx = 0
        
        self._update_cache()