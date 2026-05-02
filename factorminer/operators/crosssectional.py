"""Cross-sectional operators (across M assets at each time step t).

Input shape: ``(M, T)`` -> output shape ``(M, T)``.
Operations are performed along axis=0 (the asset dimension) for every column.
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

def cs_rank_np(x: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank -- key GPU target (26x speedup).

    For each time step, rank assets from 0 to 1.  NaN inputs get NaN rank.
    """
    # ⚡ Bolt Optimization: Vectorize using argsort(axis=0).argsort(axis=0)
    # This avoids python-level iteration over time steps (T) and is significantly faster,
    # natively replicates required tie-breaking behavior, correctly sorts NaNs to the end,
    # and avoids introducing unnecessary dependencies like scipy.stats.rankdata.
    valid = ~np.isnan(x)
    valid_counts = valid.sum(axis=0)

    order = x.argsort(axis=0).argsort(axis=0).astype(np.float64)

    with np.errstate(divide='ignore', invalid='ignore'):
        out = order / (valid_counts - 1)

    out[~valid] = np.nan
    out[:, valid_counts < 2] = np.nan
    return out


def cs_zscore_np(x: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score."""
    m = np.nanmean(x, axis=0, keepdims=True)
    s = np.nanstd(x, axis=0, keepdims=True, ddof=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 1e-10, (x - m) / s, np.nan)


def cs_demean_np(x: np.ndarray) -> np.ndarray:
    """Subtract cross-sectional mean."""
    return x - np.nanmean(x, axis=0, keepdims=True)


def cs_scale_np(x: np.ndarray) -> np.ndarray:
    """Scale to unit L1 norm cross-sectionally."""
    l1 = np.nansum(np.abs(x), axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(l1 > 1e-10, x / l1, np.nan)


def cs_neutralize_np(x: np.ndarray) -> np.ndarray:
    """Industry-neutralize (simplified: demean)."""
    return cs_demean_np(x)


def cs_quantile_np(x: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Assign each asset to a quantile bin (0 .. n_bins-1) cross-sectionally."""
    n_bins = int(n_bins)

    # ⚡ Bolt Optimization: Vectorize using argsort(axis=0).argsort(axis=0)
    # Similar to cs_rank_np, avoiding Python loops across columns is significantly
    # faster and leverages pure NumPy vectorization perfectly for ordinal ranking.
    valid = ~np.isnan(x)
    valid_counts = valid.sum(axis=0)

    order = x.argsort(axis=0).argsort(axis=0).astype(np.float64)

    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.floor(order / valid_counts * n_bins).clip(0, n_bins - 1)

    out[~valid] = np.nan
    out[:, valid_counts < 2] = np.nan
    return out


# ===========================================================================
# PyTorch implementations
# ===========================================================================

def cs_rank_torch(x: torch.Tensor) -> torch.Tensor:
    """Cross-sectional percentile rank -- fully vectorized for GPU."""
    M, T = x.shape
    not_nan = ~torch.isnan(x)
    # Replace NaN with very large value so they sort last
    filled = x.clone()
    filled[~not_nan] = float("inf")
    # argsort twice gives rank
    ranks = filled.argsort(dim=0).argsort(dim=0).float()
    # Count valid per column
    n_valid = not_nan.sum(dim=0, keepdim=True).float()
    result = ranks / (n_valid - 1).clamp(min=1)
    result[~not_nan] = float("nan")
    # Clamp ranks for entries that got inf-sorted
    result = result.clamp(0.0, 1.0)
    result[~not_nan] = float("nan")
    return result


def cs_zscore_torch(x: torch.Tensor) -> torch.Tensor:
    m = x.nanmean(dim=0, keepdim=True)
    d = x - m
    not_nan = ~torch.isnan(x)
    n = not_nan.sum(dim=0, keepdim=True).float()
    s = (d.nan_to_num(0.0).pow(2).sum(dim=0, keepdim=True) / n.clamp(min=1)).sqrt()
    result = torch.where(s > 1e-10, d / s, torch.tensor(float("nan"), device=x.device))
    result[~not_nan] = float("nan")
    return result


def cs_demean_torch(x: torch.Tensor) -> torch.Tensor:
    return x - x.nanmean(dim=0, keepdim=True)


def cs_scale_torch(x: torch.Tensor) -> torch.Tensor:
    l1 = x.abs().nansum(dim=0, keepdim=True)
    return torch.where(l1 > 1e-10, x / l1, torch.tensor(float("nan"), device=x.device))


def cs_neutralize_torch(x: torch.Tensor) -> torch.Tensor:
    return cs_demean_torch(x)


def cs_quantile_torch(x: torch.Tensor, n_bins: int = 5) -> torch.Tensor:
    n_bins = int(n_bins)
    M, T = x.shape
    not_nan = ~torch.isnan(x)
    filled = x.clone()
    filled[~not_nan] = float("inf")
    ranks = filled.argsort(dim=0).argsort(dim=0).float()
    n_valid = not_nan.sum(dim=0, keepdim=True).float()
    result = (ranks / n_valid * n_bins).floor().clamp(0, n_bins - 1)
    result[~not_nan] = float("nan")
    return result


# ===========================================================================
# Registration table
# ===========================================================================

CROSSSECTIONAL_OPS = {
    "CsRank": (cs_rank_np, cs_rank_torch),
    "CsZScore": (cs_zscore_np, cs_zscore_torch),
    "CsDemean": (cs_demean_np, cs_demean_torch),
    "CsScale": (cs_scale_np, cs_scale_torch),
    "CsNeutralize": (cs_neutralize_np, cs_neutralize_torch),
    "CsQuantile": (cs_quantile_np, cs_quantile_torch),
}
