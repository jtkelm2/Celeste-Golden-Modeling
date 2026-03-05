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


class WindowedPractice(Strategy):
    """
    Practice each room in isolation until K consecutive successes are achieved,
    then move on. Once all rooms meet the threshold, switch to full clear attempts.

    Rooms are practiced in order of lowest current success probability.
    """

    def __init__(self, room_names: List[str], k: int = 5):
        self.room_names = room_names
        self.k = k
        self.consecutive = {room: 0 for room in room_names}
        self.mastered = {room: False for room in room_names}
        self.mode = 'training'
        self.current_idx = 0
        self._current_training_room = ''
        self._pick_next_training_room()

    @property
    def name(self) -> str:
        return f"Windowed Practice (K={self.k})"

    def _pick_next_training_room(self):
        """Pick the next unmastered room to train. Returns False if all mastered."""
        unmastered = [r for r in self.room_names if not self.mastered[r]]
        if not unmastered:
            self.mode = 'full_clear'
            self.current_idx = 0
            self._current_training_room = ''
            return False
        self._current_training_room = unmastered[0]
        return True

    def get_next_room(self) -> str:
        if self.mode == 'training':
            return self._current_training_room
        else:
            return self.room_names[self.current_idx]

    def update(self, success: bool):
        if self.mode == 'training':
            room = self._current_training_room
            if success:
                self.consecutive[room] += 1
                if self.consecutive[room] >= self.k:
                    self.mastered[room] = True
                    self._pick_next_training_room()
            else:
                self.consecutive[room] = 0
        else:
            if success:
                self.current_idx += 1
                if self.current_idx >= len(self.room_names):
                    self.current_idx = 0
            else:
                self.current_idx = 0


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
            best_net = 0
            
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


class SemiomniscientOnline(Strategy):
    """
    Simulates a human player using the Golden Compass mod in real time.

    Unlike the omniscient variant, this strategy does not know the true learning
    curves. Instead it maintains its own attempt history and fits logistic models
    online, mirroring the mod's ModelFitter logic:

      - Below min_attempts_for_fit, uses a constant-probability model and marks
        the room as InsufficientData (priority practice target).
      - Above that threshold, fits a logistic via MLE. If the fitted learning
        rate is negative AND the success rate is below neg_beta_threshold, the
        room is marked NegativeLearningRate (second-priority practice target).
      - Otherwise the room is Confident and enters normal cost/benefit ranking.

    Recommendation priority (matching the mod's Advisor):
      1. InsufficientData rooms (lowest success rate first)
      2. NegativeLearningRate rooms (lowest success rate first)
      3. Room with best marginal net benefit from one practice attempt
      4. If no room has positive net benefit, attempt a full clear

    Unlike the offline Semiomniscient, this strategy re-evaluates after every
    attempt outcome — including during a full clear sequence. A failed full-clear
    attempt adds data that may cause the advisor to recommend more practice,
    pulling the player back out of full-clear mode. This matches the mod's
    behavior where recommendations update live.

    Parameters:
        room_names:            Ordered list of room names.
        models:                RoomModels (used only for room times — the strategy
                               does NOT peek at the true beta parameters).
        min_attempts_for_fit:  Minimum attempts before fitting a logistic model.
                               (default 15).
        neg_beta_threshold:    Success rate below which a negative-beta room is
                               flagged NegativeLearningRate rather than Confident.
                               (default 0.50).
    """

    INSUFFICIENT_DATA = 0
    NEGATIVE_LEARNING_RATE = 1
    CONFIDENT = 2

    def __init__(
        self,
        room_names: List[str],
        models: RoomModels,
        min_attempts_for_fit: int = 15,
        neg_beta_threshold: float = 0.50,
    ):
        self.room_names = room_names
        self.min_attempts_for_fit = min_attempts_for_fit
        self.neg_beta_threshold = neg_beta_threshold

        # Online state: history of outcomes per room
        self.history = {room: [] for room in room_names}
        # Fitted parameters per room
        self.fitted = {room: (0.0, 0.0, self.INSUFFICIENT_DATA) for room in room_names}
        # Room times (known upfront, same as omniscient)
        self.times = {room: models.rooms[room].time for room in room_names}

        self.mode = 'training'
        self.current_idx = 0
        self._last_room = ''

        self._refit_all()

    @property
    def name(self) -> str:
        return f"Semiomniscient (online, b={self.neg_beta_threshold}, n≥{self.min_attempts_for_fit})"

    # ── Model fitting (mirrors ModelFitter.Fit) ──────────────────────

    def _fit_room(self, room: str):
        """Fit a logistic model to the room's observed history."""
        attempts = self.history[room]
        n = len(attempts)

        if n == 0:
            self.fitted[room] = (0.0, 0.0, self.INSUFFICIENT_DATA)
            return

        sr = sum(attempts) / n
        sr = max(0.01, min(0.99, sr))

        if n < self.min_attempts_for_fit:
            beta0 = _log_odds(sr)
            self.fitted[room] = (beta0, 0.0, self.INSUFFICIENT_DATA)
            return

        # Fit logistic via BFGS
        beta0, beta1 = _fit_logistic_mle(attempts)

        if beta1 < 0.00001:
            beta0 = _log_odds(sr)
            beta1 = 0.0
            confidence = (
                self.NEGATIVE_LEARNING_RATE if sr < self.neg_beta_threshold
                else self.CONFIDENT
            )
            self.fitted[room] = (beta0, beta1, confidence)
        else:
            self.fitted[room] = (beta0, beta1, self.CONFIDENT)

    def _refit_all(self):
        """Refit every room and recompute cached probabilities."""
        for room in self.room_names:
            self._fit_room(room)
        self._update_cache()

    # ── Probability and E0 computation ───────────────────────────────

    def _online_prob(self, room: str, attempt: float = -1) -> float:
        """Success probability from the online-fitted model."""
        b0, b1, _ = self.fitted[room]
        if attempt < 0:
            attempt = len(self.history[room])
        return expit(b0 + b1 * attempt)

    def _update_cache(self):
        self.current_probs = {
            room: self._online_prob(room) for room in self.room_names
        }
        self.current_E0 = self._compute_E0(self.current_probs)

    def _compute_E0(self, probs: dict) -> float:
        P = 1.0
        for room in self.room_names:
            P *= probs[room]
        if P < 1e-15:
            return float('inf')

        total = 0.0
        prod_prev = 1.0
        for room in self.room_names:
            p = probs[room]
            t = self.times[room]
            a = t * (1 + p) / 2
            total += a * prod_prev
            prod_prev *= p
        return total / P

    # ── Advisor logic (mirrors SemiomniscientAdvisor.GetRecommendation) ──

    def _find_priority_room(self, level: int) -> str | None:
        """Find the room with the given confidence level and lowest probability."""
        best = None
        best_prob = float('inf')
        for room in self.room_names:
            _, _, conf = self.fitted[room]
            if conf != level:
                continue
            p = self.current_probs[room]
            if p < best_prob:
                best_prob = p
                best = room
        return best

    def _best_net_benefit_room(self) -> tuple:
        """Find the room with the highest positive net benefit. Returns (room, net) or (None, 0)."""
        best_room = None
        best_net = 0.0

        for room in self.room_names:
            b0, b1, _ = self.fitted[room]
            n = len(self.history[room])
            p_new = expit(b0 + b1 * (n + 1))

            new_probs = self.current_probs.copy()
            new_probs[room] = p_new
            new_E0 = self._compute_E0(new_probs)

            benefit = self.current_E0 - new_E0
            cost = self.times[room] * (1 + self.current_probs[room]) / 2
            net = benefit - cost

            if net > best_net:
                best_net = net
                best_room = room

        return best_room, best_net

    def get_next_room(self) -> str:
        if self.mode == 'full_clear' and self.current_idx > 0:
            room = self.room_names[self.current_idx]
            self._last_room = room
            return room

        # Priority 1: InsufficientData
        priority = self._find_priority_room(self.INSUFFICIENT_DATA)
        if priority is None:
            # Priority 2: NegativeLearningRate
            priority = self._find_priority_room(self.NEGATIVE_LEARNING_RATE)

        if priority is not None:
            self.mode = 'training'
            self._last_room = priority
            return priority

        # Priority 3: cost/benefit optimization
        best_room, _ = self._best_net_benefit_room()
        if best_room is not None:
            self.mode = 'training'
            self._last_room = best_room
            return best_room

        # Priority 4: go for gold
        if self.mode != 'full_clear':
            self.mode = 'full_clear'
            self.current_idx = 0

        room = self.room_names[self.current_idx]
        self._last_room = room
        return room

    def update(self, success: bool):
        room = self._last_room
        self.history[room].append(success)

        if self.mode == 'full_clear':
            if success:
                self.current_idx += 1
            else:
                self.current_idx = 0

        # Refit only the affected room, then update cache
        self._fit_room(room)
        self._update_cache()


# ── Utility functions for online fitting ─────────────────────────────

def _log_odds(p: float) -> float:
    """Log-odds (logit) of a probability, clamped to avoid infinities."""
    import math
    p = max(1e-10, min(1 - 1e-10, p))
    return math.log(p / (1 - p))


def _fit_logistic_mle(attempts: List[bool]) -> tuple:
    """
    Fit logistic regression via MLE, matching the mod's BFGS approach.
    Returns (beta0, beta1).
    """
    import numpy as np
    from scipy.optimize import minimize as sp_minimize

    n = len(attempts)
    t = np.arange(n, dtype=float)
    y = np.array([1.0 if a else 0.0 for a in attempts])

    def neg_ll(params):
        b0, b1 = params
        p = expit(b0 + b1 * t)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    def grad(params):
        b0, b1 = params
        p = expit(b0 + b1 * t)
        residual = p - y
        return np.array([np.sum(residual), np.sum(residual * t)])

    result = sp_minimize(neg_ll, [0.0, 0.0], jac=grad, method='BFGS')
    return float(result.x[0]), float(result.x[1])