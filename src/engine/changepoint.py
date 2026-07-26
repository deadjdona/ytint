"""Automatic penalty selection for ruptures.Pelt change-point detection.

A hardcoded penalty (e.g. change_point_penalty: 3.0) only suits one
dataset's noise level and scale -- it won't generalize across channels of
different comment volume or activity pattern. This selects a penalty via
the elbow method: sweep a penalty grid, track how many breakpoints each
value yields, and pick the first penalty where the breakpoint count
stabilizes for several consecutive grid steps. This is the heuristic
ruptures' own documentation recommends for penalty-based (Pelt) detectors
when the "correct" number of change points is unknown.
"""

from __future__ import annotations

import numpy as np
import ruptures as rpt


def select_pelt_penalty(
    signal: np.ndarray,
    model: str = "rbf",
    pen_grid: np.ndarray | None = None,
    plateau_len: int = 3,
) -> tuple[float, list[int]]:
    """Pick a Pelt penalty via the number-of-breakpoints elbow.

    Returns (chosen_penalty, breakpoint_count_per_grid_value) — the second
    value is returned too so callers can log/plot the full curve.
    """
    if pen_grid is None:
        pen_grid = np.geomspace(0.5, 500, 40)

    algo = rpt.Pelt(model=model).fit(signal)
    n_bkps = [len(algo.predict(pen=p)) - 1 for p in pen_grid]

    for i in range(len(n_bkps) - plateau_len):
        window = n_bkps[i : i + plateau_len + 1]
        if len(set(window)) == 1:
            return float(pen_grid[i]), n_bkps

    return float(pen_grid[-1]), n_bkps  # fallback: most conservative