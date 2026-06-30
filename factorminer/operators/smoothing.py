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
    out[:, window - 1 :] = cs[:, window - 1 :]
    if window > 1:
        out[:, window - 1 :] -= np.concatenate(
            [np.zeros((M, 1), dtype=np.float64), cs[:, :-window]], axis=1
        )[:, : T - window + 1]  # fix: just subtract shifted cumsum
        out[:, window - 1 :] = (
            cs[:, window - 1 :]
            - np.concatenate([np.zeros((M, 1), dtype=np.float64), cs[:, :-1]], axis=1)[
                :, : T - window + 1
            ]
        )
    out[:, window - 1 :] /= window
    out[:, : window - 1] = np.nan
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
        out[both_valid, t] = alpha * curr[both_valid] + (1 - alpha) * prev[both_valid]
        out[only_prev, t] = prev[only_prev]
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

    if T <= window:
        return out

    # ⚡ Bolt: Precalculate diffs and cumsum outside the loop to avoid
    # recomputing diff and nansum slices for every time step.
    # diff(axis=1) replaces NaN with 0 for cumulative sum to avoid NaN propagation
    abs_diffs = np.abs(np.diff(x, axis=1))
    abs_diffs = np.nan_to_num(abs_diffs, nan=0.0)

    # ⚡ Bolt: Cumsum trick for O(1) rolling sum
    cs = np.concatenate([np.zeros((M, 1), dtype=np.float64), np.cumsum(abs_diffs, axis=1)], axis=1)

    for t in range(window, T):
        direction = np.abs(x[:, t] - x[:, t - window])
        # ⚡ Bolt: Use precomputed rolling sum instead of np.nansum on a slice
        volatility = cs[:, t] - cs[:, t - window]

        with np.errstate(invalid="ignore", divide="ignore"):
            er = np.where(volatility > 1e-10, direction / volatility, 0.0)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        prev = out[:, t - 1]
        curr = x[:, t]
        valid = ~np.isnan(prev) & ~np.isnan(curr)
        # ⚡ Bolt: Use np.where to avoid expensive boolean slice indexing assignment
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
        out[both, t] = alpha * curr[both] + (1 - alpha) * prev[both]
        out[only_prev, t] = prev[only_prev]
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

    if T <= window:
        return out

    # ⚡ Bolt: Precalculate abs diffs to avoid computing it per time step.
    abs_diffs = x.diff(dim=1).abs()

    # ⚡ Bolt: Calculate rolling sum in one go using unfold to avoid small kernel
    # launches and host-device synchronization inside the loop.
    w = abs_diffs.unfold(dimension=1, size=window, step=1)
    vol_all = w.nansum(dim=2)

    for t in range(window, T):
        direction = (x[:, t] - x[:, t - window]).abs()
        # ⚡ Bolt: Fetch precalculated volatility from the unfolded rolling sum
        vol = vol_all[:, t - window]

        er = torch.where(vol > 1e-10, direction / vol, torch.zeros_like(direction))
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        prev = out[:, t - 1]
        curr = x[:, t]
        valid = ~torch.isnan(prev) & ~torch.isnan(curr)
        # ⚡ Bolt: Replace boolean indexing assignment with torch.where
        out[:, t] = torch.where(valid, prev + sc * (curr - prev), out[:, t])

    return out


def hma_torch(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    window = int(window)
    from factorminer.operators.timeseries import wma_torch

    half = max(int(window / 2), 1)
    sqrt_w = max(int(window**0.5), 1)
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
