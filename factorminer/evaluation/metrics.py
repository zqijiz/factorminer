"""Core evaluation metrics for alpha factors.

Provides vectorized, production-quality implementations of Information
Coefficient (IC), ICIR, quintile analysis, turnover, and comprehensive
factor statistics used by the validation pipeline.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

# ---------------------------------------------------------------------------
# Information Coefficient
# ---------------------------------------------------------------------------

def compute_ic(signals: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Compute IC_t = Corr_rank(s_t, r_{t+1}) for each time period.

    Uses Spearman rank correlation computed cross-sectionally at each t.
    Fully vectorized across the time dimension for optimal performance.

    Parameters
    ----------
    signals : np.ndarray, shape (M, T)
        Factor signals for M assets over T periods.
    returns : np.ndarray, shape (M, T)
        Forward returns for M assets over T periods.

    Returns
    -------
    np.ndarray, shape (T,)
        Spearman rank correlation per period.  NaN where fewer than 5
        valid (non-NaN) asset pairs exist.
    """
    invalid = np.isnan(signals) | np.isnan(returns)
    valid_count = (~invalid).sum(axis=0)

    s_masked = np.where(invalid, np.nan, signals)
    r_masked = np.where(invalid, np.nan, returns)

    rs = rankdata(s_masked, axis=0, nan_policy='omit')
    rr = rankdata(r_masked, axis=0, nan_policy='omit')

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        rs_mean = np.nanmean(rs, axis=0)
        rr_mean = np.nanmean(rr, axis=0)

    rs_m = rs - rs_mean
    rr_m = rr - rr_mean

    denom = np.sqrt(np.nansum(rs_m ** 2, axis=0) * np.nansum(rr_m ** 2, axis=0))

    ic_series = np.divide(
        np.nansum(rs_m * rr_m, axis=0),
        denom,
        out=np.zeros_like(denom),
        where=denom > 1e-12
    )

    # Safe conditional fill: scalar numeric fills before NaN masking
    ic_series[valid_count < 5] = np.nan

    return ic_series


def compute_ic_vectorized(signals: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Fully vectorized IC computation (faster for large M, T).

    Ranks are computed per-column, then Pearson correlation on ranks
    is computed without Python-level loops over T.

    Parameters
    ----------
    signals : np.ndarray, shape (M, T)
    returns : np.ndarray, shape (M, T)

    Returns
    -------
    np.ndarray, shape (T,)
    """
    M, T = signals.shape
    ic_series = np.full(T, np.nan, dtype=np.float64)

    # Mask invalid entries
    invalid = np.isnan(signals) | np.isnan(returns)

    # Rank each column independently (replace NaN with very large value to push to end)
    big = 1e18
    sig_filled = np.where(invalid, big, signals)
    ret_filled = np.where(invalid, big, returns)

    for t in range(T):
        valid = ~invalid[:, t]
        n = valid.sum()
        if n < 5:
            continue
        rs = rankdata(sig_filled[valid, t])
        rr = rankdata(ret_filled[valid, t])
        rs_m = rs - rs.mean()
        rr_m = rr - rr.mean()
        denom = np.sqrt((rs_m ** 2).sum() * (rr_m ** 2).sum())
        ic_series[t] = (rs_m * rr_m).sum() / denom if denom > 1e-12 else 0.0

    return ic_series


# ---------------------------------------------------------------------------
# IC-derived statistics
# ---------------------------------------------------------------------------

def compute_icir(ic_series: np.ndarray) -> float:
    """Compute ICIR = mean(IC) / std(IC).

    Parameters
    ----------
    ic_series : np.ndarray
        IC time series (may contain NaN).

    Returns
    -------
    float
        ICIR value.  Returns 0.0 if std is near zero or too few valid points.
    """
    valid = ic_series[~np.isnan(ic_series)]
    if len(valid) < 3:
        return 0.0
    std = float(np.std(valid, ddof=1))
    if std < 1e-12:
        return 0.0
    return float(np.mean(valid)) / std


def compute_ic_mean(ic_series: np.ndarray) -> float:
    """Compute mean absolute IC.

    Parameters
    ----------
    ic_series : np.ndarray

    Returns
    -------
    float
    """
    valid = ic_series[~np.isnan(ic_series)]
    if len(valid) == 0:
        return 0.0
    return float(np.mean(np.abs(valid)))


def compute_ic_win_rate(ic_series: np.ndarray) -> float:
    """Fraction of periods with positive IC.

    Parameters
    ----------
    ic_series : np.ndarray

    Returns
    -------
    float
        Win rate in [0, 1].
    """
    valid = ic_series[~np.isnan(ic_series)]
    if len(valid) == 0:
        return 0.0
    return float(np.mean(valid > 0))


# ---------------------------------------------------------------------------
# Cross-factor correlation
# ---------------------------------------------------------------------------

def compute_pairwise_correlation(
    signals_a: np.ndarray,
    signals_b: np.ndarray,
) -> float:
    """Time-averaged cross-sectional Spearman correlation between two factors.

    rho(a, b) = (1/|T|) * sum_t Corr_rank(s_t^a, s_t^b)
    Fully vectorized across the time dimension for optimal performance.

    Parameters
    ----------
    signals_a : np.ndarray, shape (M, T)
    signals_b : np.ndarray, shape (M, T)

    Returns
    -------
    float
        Average cross-sectional Spearman correlation.
    """
    invalid = np.isnan(signals_a) | np.isnan(signals_b)
    valid_count = (~invalid).sum(axis=0)

    a_masked = np.where(invalid, np.nan, signals_a)
    b_masked = np.where(invalid, np.nan, signals_b)

    ra = rankdata(a_masked, axis=0, nan_policy='omit')
    rb = rankdata(b_masked, axis=0, nan_policy='omit')

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ra_mean = np.nanmean(ra, axis=0)
        rb_mean = np.nanmean(rb, axis=0)

    ra_m = ra - ra_mean
    rb_m = rb - rb_mean

    denom = np.sqrt(np.nansum(ra_m ** 2, axis=0) * np.nansum(rb_m ** 2, axis=0))

    corrs = np.divide(
        np.nansum(ra_m * rb_m, axis=0),
        denom,
        out=np.zeros_like(denom),
        where=denom > 1e-12
    )

    valid_mask = valid_count >= 5
    if not np.any(valid_mask):
        return 0.0

    return float(np.mean(corrs[valid_mask]))


# ---------------------------------------------------------------------------
# Quintile analysis
# ---------------------------------------------------------------------------

def compute_quintile_returns(
    signals: np.ndarray,
    returns: np.ndarray,
    n_quantiles: int = 5,
) -> dict:
    """Sort assets into quintiles by factor signal, compute average returns.
    Fully vectorized across the time dimension for optimal performance.

    Parameters
    ----------
    signals : np.ndarray, shape (M, T)
    returns : np.ndarray, shape (M, T)
    n_quantiles : int
        Number of quantile buckets (default 5 for quintiles).

    Returns
    -------
    dict
        Keys: Q1..Q{n}, long_short, monotonicity.
        Q1 is lowest signal quintile, Q{n} is highest.
    """
    invalid = np.isnan(signals) | np.isnan(returns)
    valid_count = (~invalid).sum(axis=0)

    s_masked = np.where(invalid, np.nan, signals)
    ranks = rankdata(s_masked, axis=0, nan_policy='omit')

    with np.errstate(divide='ignore', invalid='ignore'):
        q_labels = np.ceil(ranks / valid_count * n_quantiles)

    q_labels = np.clip(q_labels, 1, n_quantiles)

    period_mask = valid_count >= n_quantiles

    means = {}
    result = {}

    import warnings
    for q in range(1, n_quantiles + 1):
        mask = (q_labels == q) & ~invalid
        with np.errstate(divide='ignore', invalid='ignore'), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            period_means = np.nansum(np.where(mask, returns, np.nan), axis=0) / mask.sum(axis=0)

        valid_period_means = period_means[period_mask & ~np.isnan(period_means)]
        if len(valid_period_means) > 0:
            means[q] = float(np.mean(valid_period_means))
        else:
            means[q] = 0.0

        result[f"Q{q}"] = means[q]

    # Long-short: top quintile minus bottom quintile
    result["long_short"] = means[n_quantiles] - means[1]

    # Monotonicity: Spearman corr between quintile index and mean return
    q_indices = np.arange(1, n_quantiles + 1, dtype=np.float64)
    q_returns = np.array([means[q] for q in range(1, n_quantiles + 1)])
    if np.std(q_returns) < 1e-12:
        result["monotonicity"] = 0.0
    else:
        rq = rankdata(q_indices)
        rr = rankdata(q_returns)
        rq_m = rq - rq.mean()
        rr_m = rr - rr.mean()
        denom = np.sqrt((rq_m ** 2).sum() * (rr_m ** 2).sum())
        result["monotonicity"] = float((rq_m * rr_m).sum() / denom) if denom > 1e-12 else 0.0

    return result


# ---------------------------------------------------------------------------
# Turnover
# ---------------------------------------------------------------------------

def compute_turnover(signals: np.ndarray, top_fraction: float = 0.2) -> float:
    """Compute average portfolio turnover rate.

    Turnover measures the fraction of top-ranked assets that change
    between consecutive periods.

    Parameters
    ----------
    signals : np.ndarray, shape (M, T)
    top_fraction : float
        Fraction of assets in the "top" bucket (default 0.2 = top quintile).

    Returns
    -------
    float
        Average turnover rate in [0, 1].
    """
    M, T = signals.shape
    k = max(int(M * top_fraction), 1)
    turnovers = []

    prev_top = None
    for t in range(T):
        col = signals[:, t]
        valid = ~np.isnan(col)
        if valid.sum() < k:
            prev_top = None
            continue
        # Get indices of top-k assets
        # Use argpartition for efficiency
        col_filled = np.where(valid, col, -np.inf)
        top_idx = set(np.argpartition(col_filled, -k)[-k:])

        if prev_top is not None:
            overlap = len(top_idx & prev_top)
            turnover = 1.0 - overlap / k
            turnovers.append(turnover)
        prev_top = top_idx

    if not turnovers:
        return 0.0
    return float(np.mean(turnovers))


# ---------------------------------------------------------------------------
# Comprehensive factor statistics
# ---------------------------------------------------------------------------

def compute_factor_stats(
    signals: np.ndarray,
    returns: np.ndarray,
) -> dict:
    """Compute comprehensive factor statistics.

    Parameters
    ----------
    signals : np.ndarray, shape (M, T)
    returns : np.ndarray, shape (M, T)

    Returns
    -------
    dict
        Keys: ic_mean, ic_abs_mean, icir, ic_win_rate,
              Q1..Q5, long_short, monotonicity, turnover
    """
    ic_series = compute_ic(signals, returns)
    valid_ic = ic_series[~np.isnan(ic_series)]

    stats: dict = {
        "ic_series": ic_series,
        "ic_mean": float(np.mean(valid_ic)) if len(valid_ic) > 0 else 0.0,
        "ic_abs_mean": compute_ic_mean(ic_series),
        "icir": compute_icir(ic_series),
        "ic_win_rate": compute_ic_win_rate(ic_series),
        "ic_std": float(np.std(valid_ic, ddof=1)) if len(valid_ic) > 2 else 0.0,
        "n_periods": int((~np.isnan(ic_series)).sum()),
    }

    # Quintile analysis
    quintile = compute_quintile_returns(signals, returns)
    stats.update(quintile)

    # Turnover
    stats["turnover"] = compute_turnover(signals)

    return stats
