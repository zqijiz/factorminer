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
        valid = ~np.isnan(prev) & ~np.isnan(curr)
        out[:, t] = np.where(
            valid,
            alpha * curr + (1 - alpha) * prev,
            np.where(~np.isnan(prev) & np.isnan(curr), prev, out[:, t]),
        )
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

    # Pre-calculate volatility and direction for all t using vectorized operations
    direction = np.zeros((M, T), dtype=np.float64)
    direction[:, window:] = np.abs(x[:, window:] - x[:, :-window])

    abs_diff = np.abs(np.diff(x, axis=1))
    cs = np.zeros((M, T), dtype=np.float64)
    cs[:, 1:] = np.nancumsum(abs_diff, axis=1)

    volatility = np.zeros((M, T), dtype=np.float64)
    volatility[:, window:] = cs[:, window:] - cs[:, :-window]

    with np.errstate(invalid="ignore", divide="ignore"):
        er = np.where(volatility > 1e-10, direction / volatility, 0.0)

    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    # Vectorized loop for the sequential component
    for t in range(window, T):
        prev = out[:, t - 1]
        curr = x[:, t]
        valid = ~np.isnan(prev) & ~np.isnan(curr)
        out[:, t] = np.where(valid, prev + sc[:, t] * (curr - prev), out[:, t])
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
        valid = ~torch.isnan(prev) & ~torch.isnan(curr)
        out[:, t] = torch.where(
            valid,
            alpha * curr + (1 - alpha) * prev,
            torch.where(~torch.isnan(prev) & torch.isnan(curr), prev, out[:, t]),
        )
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

    # Pre-calculate direction and volatility
    direction = torch.zeros((M, T), dtype=x.dtype, device=x.device)
    direction[:, window:] = (x[:, window:] - x[:, :-window]).abs()

    abs_diff = x.diff(dim=1).abs()
    # Need cumulative sum to get rolling sum, treat nans as zero
    abs_diff_no_nan = torch.nan_to_num(abs_diff, nan=0.0)
    cs = torch.zeros((M, T), dtype=x.dtype, device=x.device)
    cs[:, 1:] = torch.cumsum(abs_diff_no_nan, dim=1)

    vol = torch.zeros((M, T), dtype=x.dtype, device=x.device)
    vol[:, window:] = cs[:, window:] - cs[:, :-window]

    er = torch.where(vol > 1e-10, direction / vol, torch.zeros_like(direction))
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    # Vectorized loop for the sequential component with branchless torch.where
    for t in range(window, T):
        prev = out[:, t - 1]
        curr = x[:, t]
        valid = ~torch.isnan(prev) & ~torch.isnan(curr)
        out[:, t] = torch.where(valid, prev + sc[:, t] * (curr - prev), out[:, t])
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
