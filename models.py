#!/usr/bin/env python3
"""
Room probability models for Celeste golden runs.
"""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize


@dataclass
class RoomModel:
    """Parameters for a single room's logistic success model."""
    beta_0: float
    beta_1: float
    time: float


class RoomModels:
    """
    Collection of room models with shared probability/time calculations.
    
    Model: P(success) = 1 / (1 + exp(-(β₀ + β₁ * attempt)))
    """
    
    def __init__(self, params: Dict[str, dict]):
        self.rooms = {
            name: RoomModel(
                beta_0=p['beta_0'],
                beta_1=p['beta_1'],
                time=p['time']
            )
            for name, p in params.items()
        }
        self.room_names = sorted(self.rooms.keys())
    
    def success_prob(self, room: str, attempt: int) -> float:
        """Probability of success on given attempt number (0-indexed)."""
        m = self.rooms[room]
        return expit(m.beta_0 + m.beta_1 * attempt)
    
    def attempt_time(self, room: str, success: bool) -> float:
        """Time for an attempt: full time if success, half if death."""
        t = self.rooms[room].time
        return t if success else t / 2


def fit_logistic_model(attempts: List[bool]) -> dict:
    """
    Fit logistic regression to attempt data using MLE.
    
    Args:
        attempts: List of success/failure outcomes
        
    Returns:
        Dict with beta_0, beta_1
    """
    n = len(attempts)
    t = np.arange(n)
    y = np.array([1 if a else 0 for a in attempts])
    
    def neg_log_likelihood(params):
        beta_0, beta_1 = params
        p = expit(beta_0 + beta_1 * t)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    
    result = minimize(neg_log_likelihood, [0, 0], method='BFGS')
    beta_0, beta_1 = result.x
    
    if beta_1 < 0:
        print("    WARNING: Negative learning curve detected; resorting to constant fit instead.")
        result = minimize(lambda x: neg_log_likelihood((x,0)), [0], method='BFGS')
        beta_0, beta_1 = result.x[0], 0
    
    return {'beta_0': float(beta_0), 'beta_1': float(beta_1)}


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"