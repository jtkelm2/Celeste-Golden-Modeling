#!/usr/bin/env python3
"""
Practice strategies for Celeste golden runs.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
from scipy.special import expit
from models import RoomModels, expected_attempt_time
import numpy as np
from scipy.optimize import minimize as sp_minimize


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


class Mastery(Strategy):
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
        return f"Mastery (K={self.k})"

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
        where a_j is the average for a single room
        """
        P = 1.0
        for room in self.room_names:
            P *= probs[room]
        
        total = 0.0
        prod_prev = 1.0
        for room in self.room_names:
            p = probs[room]
            a = self.models.expected_attempt_time(room, p)
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
        cost = self.models.expected_attempt_time(room, self.current_probs[room])

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
        neg_beta_threshold: float = 0.50,
        stability_window: int = 5,
        stability_eps: float = 0.0005,
    ):
        self.room_names = room_names
        self.neg_beta_threshold = neg_beta_threshold
        self.stability_window = stability_window
        self.stability_eps = stability_eps

        # Online state: history of outcomes per room
        self.history = {room: [] for room in room_names}
        # Rolling history of raw MLE-fitted β₁ per room, used for stability gating
        self.beta1_history = {room: [] for room in room_names}
        # Fitted parameters per room: (β₀, β₁, confidence_class)
        self.fitted = {room: (0.0, 0.0, self.INSUFFICIENT_DATA) for room in room_names}
        # Room times (known upfront, same as omniscient)
        self.times = {room: models.rooms[room].time for room in room_names}

        self.mode = 'training'
        self.current_idx = 0
        self._last_room = ''

        self._refit_all()

    @property
    def name(self) -> str:
        return (f"Semiomniscient (online, b={self.neg_beta_threshold}, "
                f"w={self.stability_window}, ε={self.stability_eps})")

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
        beta0_constant = _log_odds(sr)

        if n < 3:
            # Too few points for a meaningful logistic fit
            self.fitted[room] = (beta0_constant, 0.0, self.INSUFFICIENT_DATA)
            return

        # Fit logistic via BFGS; track raw β₁ for stability gating
        beta0_fit, beta1_fit = _fit_logistic_mle(attempts)
        self.beta1_history[room].append(beta1_fit)

        recent = self.beta1_history[room][-self.stability_window:]
        if len(recent) < self.stability_window or np.std(recent) > self.stability_eps:
            # β₁ not yet stable: hold in INSUFFICIENT_DATA with constant model
            self.fitted[room] = (beta0_constant, 0.0, self.INSUFFICIENT_DATA)
            return

        # Stable enough — classify normally
        if beta1_fit < 0.00001:
            confidence = (
                self.NEGATIVE_LEARNING_RATE if sr < self.neg_beta_threshold
                else self.CONFIDENT
            )
            self.fitted[room] = (beta0_constant, 0.0, confidence)
        else:
            self.fitted[room] = (beta0_fit, beta1_fit, self.CONFIDENT)

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
            t = self.times[room]
            p = probs[room]
            a = expected_attempt_time(t, p)
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
            cost = expected_attempt_time(self.times[room], self.current_probs[room])
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

class Poisson(Strategy):
    """
    Models the chapter as a Poisson point process to decide between
    targeted practice and full-clear attempts.

    Each room i has death rate λ_i = -log(p_i). The chapter's aggregate
    death rate is Λ = Σλ_i, and the expected time to complete a golden
    run (assuming no further learning) is

        f(Λ) = T · (e^Λ - 1) / Λ

    where T = Σt_i is the total room time.

    The optimal room to practice maximises β₁_i · λ_i / t_i, and we
    should practice at all iff the marginal benefit exceeds the cost:

        f'(Λ) > t_best / (β₁_best · λ_best)

    This avoids recomputing E₀ per room entirely: the ranking is O(n)
    with no inner loops, and the train/clear decision is a single
    scalar comparison.

    Trade-off vs Semiomniscient: this ignores positional weighting
    (early rooms gate late rooms), treating all death risk as
    order-independent.
    """

    def __init__(self, room_names: List[str], models: RoomModels):
        self.room_names = room_names
        self.models = models
        self.attempt_counts = {room: 0 for room in room_names}
        self.mode = 'training'
        self.current_idx = 0
        self._last_room = None

        self._update_cache()

    @property
    def name(self) -> str:
        return "Poisson"

    # ── Cache ────────────────────────────────────────────────────────

    def _update_cache(self):
        """Recompute per-room probabilities, death rates, and aggregate quantities."""
        self.current_probs = {
            room: self.models.success_prob(room, self.attempt_counts[room])
            for room in self.room_names
        }
        eps = 1e-15
        self._lambdas = {
            room: -np.log(np.clip(self.current_probs[room], eps, 1 - eps))
            for room in self.room_names
        }
        self._big_lambda = sum(self._lambdas.values())
        self._big_T = sum(self.models.rooms[room].time for room in self.room_names)

    @staticmethod
    def _f(T: float, L: float) -> float:
        """Expected completion time: T(e^Λ - 1)/Λ."""
        if L < 1e-12:
            return T  # limΛ→0 f = T
        return T * (np.exp(L) - 1) / L

    @staticmethod
    def _df_dL(T: float, L: float) -> float:
        """df/dΛ = T[(Λ-1)e^Λ + 1] / Λ²."""
        if L < 1e-12:
            return T / 2  # limΛ→0 via Taylor: T·(1/2 + Λ/3 + ...)
        return T * ((L - 1) * np.exp(L) + 1) / (L * L)

    # ── Decision logic ───────────────────────────────────────────────

    def _best_training_room(self):
        """
        Find the room with highest priority score β₁·λ/t, and check
        whether training it beats attempting a full clear.

        Returns (room, net_benefit) or (None, 0) if no room is worth
        training.
        """
        df = self._df_dL(self._big_T, self._big_lambda)

        best_room = None
        best_score = 0.0

        for room in self.room_names:
            m = self.models.rooms[room]
            lam = self._lambdas[room]
            # β₁ ≤ 0 means no learning; skip
            if m.beta_1 <= 0:
                continue
            score = m.beta_1 * lam / m.time
            if score > best_score:
                best_score = score
                best_room = room

        if best_room is None:
            return None, 0.0

        m = self.models.rooms[best_room]
        lam = self._lambdas[best_room]
        threshold = m.time / (m.beta_1 * lam) if (m.beta_1 * lam) > 1e-15 else float('inf')

        if df > threshold:
            # Net benefit: first-order approximation
            p = self.current_probs[best_room]
            delta = m.beta_1 * (1 - p)
            benefit = df * delta
            cost = expected_attempt_time(m.time, p)
            return best_room, benefit - cost
        else:
            return None, 0.0

    def get_next_room(self) -> str:
        if self.mode == 'training':
            best_room, _ = self._best_training_room()

            if best_room is not None:
                self._last_room = best_room
                return best_room
            else:
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

class PoissonOnline(Strategy):
    """
    Online variant of the Poisson strategy using principled uncertainty
    quantification to balance exploration and exploitation.

    Like Poisson, models the chapter as a Poisson point process with
    aggregate death rate Λ = Σλ_i and expected completion time
    f(Λ) = T(e^Λ - 1)/Λ.  Unlike the offline variant, this strategy
    does not know the true learning curves.  It fits regularised
    logistic models online and uses the Laplace approximation (observed
    Fisher information) to quantify parametric uncertainty.

    Room selection uses a UCB (upper confidence bound) on the priority
    score s_i = β₁_i · λ_i / t_i.  Uncertainty on s_i is obtained via
    the delta method: Var(s) = ∇s^T Σ ∇s, where Σ = I_k⁻¹ is the
    posterior covariance from the Fisher information plus a Gaussian
    prior.  The UCB score is then

        s_UCB = ŝ + α · σ_s

    This naturally drives exploration of rooms with few attempts (high
    σ_s) and converges to the point-estimate ranking as data
    accumulates.  No ad-hoc stability windows or hard data-count
    thresholds are needed.

    The train-vs-clear decision checks df/dΛ · s_UCB > 1 for the best
    room; if no room passes this test, the strategy attempts a full
    clear.  Like SemiomniscientOnline, it re-evaluates at the start of
    each full-clear run (but not mid-run).

    Parameters:
        room_names: Ordered list of room names.
        models:     RoomModels (used ONLY for room times; true β
                    parameters are never accessed).
        alpha:      UCB exploration parameter.  Higher values drive more
                    exploration.
        tau:        Prior standard deviation on (β₀, β₁).  Controls
                    regularisation strength and initial uncertainty.
                    Larger τ = weaker prior = more initial exploration.
    """

    def __init__(
        self,
        room_names: List[str],
        models: RoomModels,
        alpha: float = 0.6,
        tau: float = 3.0,
    ):
        self.room_names = room_names
        self.alpha = alpha
        self.tau = tau
        self._tau2 = tau * tau

        # Only record room times — no peeking at true β parameters
        self.times = {room: models.rooms[room].time for room in room_names}
        self._big_T = sum(self.times.values())

        # Online state
        self.history: dict[str, list[bool]] = {room: [] for room in room_names}
        self.fitted: dict[str, tuple[float, float]] = {
            room: (0.0, 0.0) for room in room_names
        }
        # Posterior covariance per room (initialised to prior)
        self.covariance: dict[str, np.ndarray] = {
            room: np.eye(2) * self._tau2 for room in room_names
        }

        self.mode = 'training'
        self.current_idx = 0
        self._last_room = ''

        self._update_cache()

    @property
    def name(self) -> str:
        return f"Poisson (online, α={self.alpha}, τ={self.tau})"

    # ── Model fitting ────────────────────────────────────────────────

    def _fit_room(self, room: str):
        """
        Update the regularised logistic model for a room after a new
        observation.

        Performs a single Newton-Raphson step on the log-
        posterior — one gradient + Hessian evaluation instead of
        iterating BFGS to convergence.
        """
        attempts = self.history[room]
        n = len(attempts)

        if n == 0:
            self.fitted[room] = (0.0, 0.0)
            self.covariance[room] = np.eye(2) * self._tau2
            return

        t_arr = np.arange(n, dtype=float)
        y = np.array([1.0 if a else 0.0 for a in attempts])
        inv_tau2 = 1.0 / self._tau2

        
        # Single Newton-Raphson step: θ ← θ − H⁻¹∇
        b0, b1 = self.fitted[room]
        p_hat = expit(b0 + b1 * t_arr)
        r = p_hat - y

        grad = np.array([
            np.sum(r) + inv_tau2 * b0,
            np.sum(r * t_arr) + inv_tau2 * b1,
        ])

        w = p_hat * (1 - p_hat)
        H00 = np.sum(w) + inv_tau2
        H01 = np.sum(w * t_arr)
        H11 = np.sum(w * t_arr * t_arr) + inv_tau2
        det = H00 * H11 - H01 * H01

        if det > 1e-20:
            # H⁻¹ ∇  (inline 2×2 solve)
            step0 = ( H11 * grad[0] - H01 * grad[1]) / det
            step1 = (-H01 * grad[0] + H00 * grad[1]) / det
            b0 -= step0
            b1 = max(0.0, b1 - step1)  # projected Newton: β₁ ≥ 0

        self.fitted[room] = (b0, b1)

        # Posterior covariance: (H_lik + H_prior)⁻¹
        p_hat = expit(b0 + b1 * t_arr)
        w = p_hat * (1 - p_hat)

        I00 = np.sum(w) + inv_tau2
        I01 = np.sum(w * t_arr)
        I11 = np.sum(w * t_arr * t_arr) + inv_tau2
        det = I00 * I11 - I01 * I01

        if det > 1e-20:
            self.covariance[room] = np.array([
                [ I11, -I01],
                [-I01,  I00],
            ]) / det
        else:
            self.covariance[room] = np.eye(2) * self._tau2

    # ── Cached quantities ────────────────────────────────────────────

    def _update_cache(self):
        """Recompute per-room probabilities, death rates, and Λ."""
        eps = 1e-15
        self.current_probs: dict[str, float] = {}
        self._lambdas: dict[str, float] = {}

        for room in self.room_names:
            b0, b1 = self.fitted[room]
            n = len(self.history[room])
            p = float(expit(b0 + b1 * n))
            self.current_probs[room] = p
            self._lambdas[room] = -np.log(np.clip(p, eps, 1 - eps))

        self._big_lambda = sum(self._lambdas.values())

    # ── Poisson expected-time helpers ────────────────────────────────

    @staticmethod
    def _f(T: float, L: float) -> float:
        """Expected completion time: T(e^Λ − 1)/Λ."""
        if L < 1e-12:
            return T
        return T * (np.exp(L) - 1) / L

    @staticmethod
    def _df_dL(T: float, L: float) -> float:
        """df/dΛ = T[(Λ−1)e^Λ + 1] / Λ²."""
        if L < 1e-12:
            return T / 2
        return T * ((L - 1) * np.exp(L) + 1) / (L * L)

    # ── UCB on priority score ────────────────────────────────────────

    def _ucb_score(self, room: str) -> float:
        """
        Compute the upper confidence bound on the priority score

            s = β₁ · λ / t

        using the delta method to propagate parametric uncertainty.

        Returns s_UCB = ŝ + α · σ_s.
        """
        b0, b1 = self.fitted[room]
        n = len(self.history[room])
        t = self.times[room]
        Sigma = self.covariance[room]

        eta = b0 + b1 * n
        p = float(expit(eta))
        lam = -np.log(np.clip(p, 1e-15, 1 - 1e-15))

        s_hat = b1 * lam / t

        # ∂s/∂β₀ = −β₁(1−p) / t
        ds_db0 = -b1 * (1.0 - p) / t
        # ∂s/∂β₁ = [λ − n·β₁·(1−p)] / t
        ds_db1 = (lam - n * b1 * (1.0 - p)) / t
        grad = np.array([ds_db0, ds_db1])

        var_s = float(grad @ Sigma @ grad)
        sigma_s = np.sqrt(max(0.0, var_s))

        return s_hat + self.alpha * sigma_s

    # ── Decision logic ───────────────────────────────────────────────

    def _best_training_room(self):
        """
        Find the room with the highest UCB score and check whether
        training it beats attempting a full clear.

        Returns the room name, or None if no room is worth training.
        """
        df = self._df_dL(self._big_T, self._big_lambda)

        best_room = None
        best_ucb = -np.inf

        for room in self.room_names:
            ucb = self._ucb_score(room)
            if ucb > best_ucb:
                best_ucb = ucb
                best_room = room

        # Train iff df/dΛ · s_UCB > 1  (optimistic benefit > cost)
        if best_room is not None and best_ucb > 0 and df * best_ucb > 1.0:
            return best_room
        return None

    def get_next_room(self) -> str:
        # Mid full-clear run: don't interrupt
        if self.mode == 'full_clear' and self.current_idx > 0:
            room = self.room_names[self.current_idx]
            self._last_room = room
            return room

        # Re-evaluate: should we train?
        best = self._best_training_room()
        if best is not None:
            self.mode = 'training'
            self._last_room = best
            return best

        # Go for gold
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
                if self.current_idx >= len(self.room_names):
                    self.current_idx = 0
            else:
                self.current_idx = 0

        # Refit only the room that just got new data
        self._fit_room(room)
        self._update_cache()