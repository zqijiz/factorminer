"""Moving average / smoothing operators.

Input shape: ``(M, T)`` -> output shape ``(M, T)``.
All operate along the time axis (axis=1) per asset row.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


# ===========================================================================
# NumPy implementations
# ===========================================================================

def sma_np(x: np.ndarray, window: int = 10) -> np.ndarray:
    """Simple moving average (identical to Mean)."""
    window = int(window)
    M, T = x.shape
    out = np.full_like(x, np.nan, dtype=np.float64)
    # Cumsum trick for O(1) per element
    cs = np.nancumsum(x, axis=1)
    out[:, window - 1:] = cs[:, window - 1:]
    if window > 1:
        out[:, window - 1:] -= np.concatenate(
            [np.zeros((M, 1), dtype=np.float64), cs[:, :-window]], axis=1
        )[:, :T - window + 1]  # fix: just subtract shifted cumsum
        out[:, window - 1:] = (cs[:, window - 1:] - np.concatenate(
            [np.zeros((M, 1), dtype=np.float64), cs[:, :-1]], axis=1
        )[:, :T - window + 1])
    out[:, window - 1:] /= window
    out[:, :window - 1] = np.nan
    return out


def ema_np(x: np.ndarray, window: int = 10) -> np.ndarray:
    """Exponential moving average with span = window."""
    window = int(window)
    alpha = 2.0 / (window + 1.0)
    M, T = x.shape
    out = np.copy(x).astype(np.float64)
    for t in range(1, T):
        prev = out[:, t - 1]
        curr = x[:, t]
        both_valid = ~np.isnan(prev) & ~np.isnan(curr)
        only_prev = ~np.isnan(prev) & np.isnan(curr)

        # Branchless update to avoid expensive boolean slice assignments in a loop
        new_val = np.where(both_valid, alpha * curr + (1 - alpha) * prev,
                           np.where(only_prev, prev, out[:, t]))
        out[:, t] = new_val
    return out


def dema_np(x: np.ndarray, window: int = 10) -> np.ndarray:
    """Double EMA: 2 * EMA(x) - EMA(EMA(x))."""
    e1 = ema_np(x, window)
    e2 = ema_np(e1, window)
    return 2.0 * e1 - e2


def kama_np(x: np.ndarray, window: int = 10) -> np.ndarray:
    """Kaufman Adaptive Moving Average."""
    window = int(window)
    fast_sc = 2.0 / (2.0 + 1.0)
    slow_sc = 2.0 / (30.0 + 1.0)
    M, T = x.shape
    out = np.copy(x).astype(np.float64)

    # Pre-calculate rolling volatility and direction outside the loop for speed
    diffs = np.abs(np.diff(x, axis=1))
    np.nan_to_num(diffs, copy=False, nan=0.0)

    cumsum_diffs = np.zeros((M, T), dtype=np.float64)
    cumsum_diffs[:, 1:] = np.cumsum(diffs, axis=1)

    direction_all = np.abs(x[:, window:] - x[:, :-window])
    volatility_all = cumsum_diffs[:, window:] - cumsum_diffs[:, :-window]

    with np.errstate(invalid="ignore", divide="ignore"):
        er_all = np.where(volatility_all > 1e-10, direction_all / volatility_all, 0.0)
    sc_all = (er_all * (fast_sc - slow_sc) + slow_sc) ** 2

    for t in range(window, T):
        sc = sc_all[:, t - window]
        prev = out[:, t - 1]
        curr = x[:, t]
        valid = ~np.isnan(prev) & ~np.isnan(curr)

        # Branchless update avoids expensive slice assignment overhead
        out[:, t] = np.where(valid, prev + sc * (curr - prev), out[:, t])
    return out


def hma_np(x: np.ndarray, window: int = 10) -> np.ndarray:
    """Hull Moving Average: WMA(2*WMA(x, w/2) - WMA(x, w), sqrt(w))."""
    window = int(window)
    from factorminer.operators.timeseries import wma_np
    half = max(int(window / 2), 1)
    sqrt_w = max(int(np.sqrt(window)), 1)
    w1 = wma_np(x, half)
    w2 = wma_np(x, window)
    diff = 2.0 * w1 - w2
    return wma_np(diff, sqrt_w)


# ===========================================================================
# PyTorch implementations
# ===========================================================================

def sma_torch(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    """Simple moving average using conv1d for GPU efficiency."""
    window = int(window)
    M, T = x.shape
    # Use unfold-based approach
    from factorminer.operators.statistical import _pad_front_torch, _unfold_torch
    w = _unfold_torch(x, window)
    result = w.nanmean(dim=2)
    return _pad_front_torch(result, window, T)


def ema_torch(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    """EMA -- sequential by nature, but batch across assets."""
    window = int(window)
    alpha = 2.0 / (window + 1.0)
    M, T = x.shape
    out = x.clone()
    for t in range(1, T):
        prev = out[:, t - 1]
        curr = x[:, t]
        both = ~torch.isnan(prev) & ~torch.isnan(curr)
        only_prev = ~torch.isnan(prev) & torch.isnan(curr)

        # Branchless update eliminates host-device sync overhead in PyTorch loops
        new_val = torch.where(both, alpha * curr + (1 - alpha) * prev,
                           torch.where(only_prev, prev, out[:, t]))
        out[:, t] = new_val
    return out


def dema_torch(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    e1 = ema_torch(x, window)
    e2 = ema_torch(e1, window)
    return 2.0 * e1 - e2


def kama_torch(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    window = int(window)
    fast_sc = 2.0 / (2.0 + 1.0)
    slow_sc = 2.0 / (30.0 + 1.0)
    M, T = x.shape
    out = x.clone()

    # Pre-calculate volatility and direction outside the loop
    diffs = x.diff(dim=1).abs()
    volatility_all = diffs.unfold(1, window, 1).nansum(dim=2)
    direction_all = (x[:, window:] - x[:, :-window]).abs()

    er_all = torch.where(volatility_all > 1e-10, direction_all / volatility_all, torch.zeros_like(direction_all))
    sc_all = (er_all * (fast_sc - slow_sc) + slow_sc) ** 2

    for t in range(window, T):
        sc = sc_all[:, t - window]
        prev = out[:, t - 1]
        curr = x[:, t]
        valid = ~torch.isnan(prev) & ~torch.isnan(curr)

        # Branchless sequential update
        out[:, t] = torch.where(valid, prev + sc * (curr - prev), out[:, t])
    return out


def hma_torch(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    window = int(window)
    from factorminer.operators.timeseries import wma_torch
    half = max(int(window / 2), 1)
    sqrt_w = max(int(window ** 0.5), 1)
    w1 = wma_torch(x, half)
    w2 = wma_torch(x, window)
    diff = 2.0 * w1 - w2
    return wma_torch(diff, sqrt_w)


# ===========================================================================
# Registration table
# ===========================================================================

SMOOTHING_OPS = {
    "EMA": (ema_np, ema_torch),
    "DEMA": (dema_np, dema_torch),
    "SMA": (sma_np, sma_torch),
    "KAMA": (kama_np, kama_torch),
    "HMA": (hma_np, hma_torch),
}
