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
    valid = ~(np.isnan(signals) | np.isnan(returns))

    # Fill invalid with np.nan for rankdata omit policy
    s_masked = np.where(valid, signals, np.nan)
    r_masked = np.where(valid, returns, np.nan)

    # rankdata with omit preserves NaNs, computes rank on valid items
    rs = rankdata(s_masked, axis=0, nan_policy='omit')
    rr = rankdata(r_masked, axis=0, nan_policy='omit')

    rs_mean = np.nanmean(rs, axis=0)
    rr_mean = np.nanmean(rr, axis=0)

    rs_m = rs - rs_mean
    rr_m = rr - rr_mean

    cov = np.nansum(rs_m * rr_m, axis=0)
    var_s = np.nansum(rs_m**2, axis=0)
    var_r = np.nansum(rr_m**2, axis=0)

    denom = np.sqrt(var_s * var_r)

    with np.errstate(divide='ignore', invalid='ignore'):
        ic_series = cov / denom

    valid_count = valid.sum(axis=0)
    ic_series[denom < 1e-12] = 0.0
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
    return compute_ic(signals, returns)


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

    Parameters
    ----------
    signals_a : np.ndarray, shape (M, T)
    signals_b : np.ndarray, shape (M, T)

    Returns
    -------
    float
        Average cross-sectional Spearman correlation.
    """
    valid = ~(np.isnan(signals_a) | np.isnan(signals_b))

    a_masked = np.where(valid, signals_a, np.nan)
    b_masked = np.where(valid, signals_b, np.nan)

    ra = rankdata(a_masked, axis=0, nan_policy='omit')
    rb = rankdata(b_masked, axis=0, nan_policy='omit')

    ra_mean = np.nanmean(ra, axis=0)
    rb_mean = np.nanmean(rb, axis=0)

    ra_m = ra - ra_mean
    rb_m = rb - rb_mean

    cov = np.nansum(ra_m * rb_m, axis=0)
    var_a = np.nansum(ra_m**2, axis=0)
    var_b = np.nansum(rb_m**2, axis=0)

    denom = np.sqrt(var_a * var_b)

    with np.errstate(divide='ignore', invalid='ignore'):
        corrs = cov / denom

    valid_count = valid.sum(axis=0)
    corrs[denom < 1e-12] = 0.0
    corrs[valid_count < 5] = np.nan

    mean_corr = np.nanmean(corrs)
    if np.isnan(mean_corr):
        return 0.0
    return float(mean_corr)


# ---------------------------------------------------------------------------
# Quintile analysis
# ---------------------------------------------------------------------------

def compute_quintile_returns(
    signals: np.ndarray,
    returns: np.ndarray,
    n_quantiles: int = 5,
) -> dict:
    """Sort assets into quintiles by factor signal, compute average returns.

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
    valid = ~(np.isnan(signals) | np.isnan(returns))

    s_masked = np.where(valid, signals, np.nan)
    r_masked = np.where(valid, returns, np.nan)

    ranks = rankdata(s_masked, axis=0, nan_policy='omit')
    valid_count = valid.sum(axis=0)

    process_mask = valid_count >= n_quantiles

    means = {q: 0.0 for q in range(1, n_quantiles + 1)}
    result = {}

    if process_mask.any():
        q_labels = np.clip(
            np.ceil(ranks / valid_count * n_quantiles),
            1,
            n_quantiles
        )

        for q in range(1, n_quantiles + 1):
            q_mask = (q_labels == q) & process_mask
            if q_mask.any():
                col_sums = np.nansum(np.where(q_mask, r_masked, np.nan), axis=0)
                col_counts = np.sum(q_mask, axis=0)

                valid_cols = col_counts > 0
                if valid_cols.any():
                    col_means = col_sums[valid_cols] / col_counts[valid_cols]
                    means[q] = float(np.mean(col_means))

    for q in range(1, n_quantiles + 1):
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
