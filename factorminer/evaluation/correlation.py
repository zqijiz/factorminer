"""Efficient correlation computation for factor evaluation.

Provides batch Spearman rank correlation, vectorized cross-sectional
correlation, and incremental correlation matrix updates for the
factor library.  Supports both numpy and optional torch backends.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Batch cross-sectional Spearman rank correlation
# ---------------------------------------------------------------------------

def _rank_columns(x: np.ndarray) -> np.ndarray:
    """Rank each column of x independently, leaving NaN as NaN.

    Parameters
    ----------
    x : np.ndarray, shape (M, T)

    Returns
    -------
    np.ndarray, shape (M, T)
        Ranks per column, NaN where input was NaN.
    """
    from scipy.stats import rankdata
    M, T = x.shape
    ranked = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(T):
        col = x[:, t]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            continue
        ranked[valid, t] = rankdata(col[valid])
    return ranked


def batch_spearman_correlation(
    candidate_signals: np.ndarray,
    library_signals: np.ndarray,
) -> np.ndarray:
    """Compute Spearman correlation between one candidate and multiple library factors.

    For each library factor g, computes:
        rho = (1/|T_valid|) * sum_t Corr_rank(candidate_t, g_t)

    Parameters
    ----------
    candidate_signals : np.ndarray, shape (M, T)
        Signal array for the candidate factor.
    library_signals : np.ndarray, shape (N, M, T)
        Signal arrays for N library factors.

    Returns
    -------
    np.ndarray, shape (N,)
        Average cross-sectional Spearman correlation with each library factor.
    """
    N = library_signals.shape[0]
    if N == 0:
        return np.array([], dtype=np.float64)

    M, T = candidate_signals.shape
    correlations = np.zeros(N, dtype=np.float64)

    # Rank candidate columns once
    cand_ranked = _rank_columns(candidate_signals)

    for i in range(N):
        lib_ranked = _rank_columns(library_signals[i])

        valid = ~(np.isnan(cand_ranked) | np.isnan(lib_ranked))
        c_vals = np.where(valid, cand_ranked, 0.0)
        l_vals = np.where(valid, lib_ranked, 0.0)

        counts = valid.sum(axis=0)

        c_mean = c_vals.sum(axis=0) / np.maximum(counts, 1)
        l_mean = l_vals.sum(axis=0) / np.maximum(counts, 1)

        c_centered = c_vals - c_mean
        l_centered = l_vals - l_mean

        c_centered[~valid] = 0.0
        l_centered[~valid] = 0.0

        cov = (c_centered * l_centered).sum(axis=0)
        var_c = (c_centered**2).sum(axis=0)
        var_l = (l_centered**2).sum(axis=0)

        denom = np.sqrt(var_c * var_l)

        mask = (counts >= 5) & (denom > 1e-12)
        if np.any(mask):
            # Using np.divide with where for safe division as per memory instructions
            corrs = np.divide(cov, denom, out=np.zeros_like(cov), where=mask)
            correlations[i] = np.mean(corrs[mask])

    return correlations


def batch_spearman_pairwise(
    signals_list: list[np.ndarray],
) -> np.ndarray:
    """Compute pairwise Spearman correlation matrix for a list of signal arrays.

    Parameters
    ----------
    signals_list : list of np.ndarray, each shape (M, T)
        Signal arrays for K candidate factors.

    Returns
    -------
    np.ndarray, shape (K, K)
        Symmetric correlation matrix. Diagonal is 1.0.
    """
    K = len(signals_list)
    if K == 0:
        return np.array([], dtype=np.float64).reshape(0, 0)

    M, T = signals_list[0].shape

    # Pre-compute ranks for all candidates
    ranked_list = [_rank_columns(s) for s in signals_list]

    corr_matrix = np.eye(K, dtype=np.float64)

    for i in range(K):
        for j in range(i + 1, K):
            ri = ranked_list[i]
            rj = ranked_list[j]

            valid = ~(np.isnan(ri) | np.isnan(rj))
            val_i = np.where(valid, ri, 0.0)
            val_j = np.where(valid, rj, 0.0)

            counts = valid.sum(axis=0)

            mi = val_i.sum(axis=0) / np.maximum(counts, 1)
            mj = val_j.sum(axis=0) / np.maximum(counts, 1)

            ci = val_i - mi
            cj = val_j - mj

            ci[~valid] = 0.0
            cj[~valid] = 0.0

            cov = (ci * cj).sum(axis=0)
            var_i = (ci**2).sum(axis=0)
            var_j = (cj**2).sum(axis=0)

            denom = np.sqrt(var_i * var_j)

            mask = (counts >= 5) & (denom > 1e-12)
            if np.any(mask):
                corrs = np.divide(cov, denom, out=np.zeros_like(cov), where=mask)
                corr = np.mean(corrs[mask])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

    return corr_matrix


# ---------------------------------------------------------------------------
# Incremental correlation matrix update
# ---------------------------------------------------------------------------

class IncrementalCorrelationMatrix:
    """Maintains a correlation matrix that can be incrementally updated.

    Supports adding new factors and removing existing ones without
    recomputing the entire matrix from scratch.
    """

    def __init__(self) -> None:
        self._signals: dict[str, np.ndarray] = {}
        self._ranked: dict[str, np.ndarray] = {}
        self._corr_cache: dict[tuple[str, str], float] = {}
        self._factor_ids: list[str] = []

    @property
    def size(self) -> int:
        return len(self._factor_ids)

    @property
    def factor_ids(self) -> list[str]:
        return list(self._factor_ids)

    def _compute_pair_corr(self, id_a: str, id_b: str) -> float:
        """Compute average cross-sectional Spearman between two factors."""
        ra = self._ranked[id_a]
        rb = self._ranked[id_b]
        M, T = ra.shape
        corr_sum = 0.0
        count = 0
        for t in range(T):
            a_col = ra[:, t]
            b_col = rb[:, t]
            valid = ~(np.isnan(a_col) | np.isnan(b_col))
            n = valid.sum()
            if n < 5:
                continue
            a_v = a_col[valid]
            b_v = b_col[valid]
            a_m = a_v - a_v.mean()
            b_m = b_v - b_v.mean()
            denom = np.sqrt((a_m ** 2).sum() * (b_m ** 2).sum())
            if denom > 1e-12:
                corr_sum += (a_m * b_m).sum() / denom
            count += 1
        return corr_sum / count if count > 0 else 0.0

    def add_factor(self, factor_id: str, signals: np.ndarray) -> dict[str, float]:
        """Add a factor and compute its correlation with all existing factors.

        Parameters
        ----------
        factor_id : str
        signals : np.ndarray, shape (M, T)

        Returns
        -------
        dict
            Mapping from existing factor_id to correlation with the new factor.
        """
        self._signals[factor_id] = signals
        self._ranked[factor_id] = _rank_columns(signals)

        correlations: dict[str, float] = {}
        for existing_id in self._factor_ids:
            corr = self._compute_pair_corr(factor_id, existing_id)
            key = (min(factor_id, existing_id), max(factor_id, existing_id))
            self._corr_cache[key] = corr
            correlations[existing_id] = corr

        self._factor_ids.append(factor_id)
        return correlations

    def remove_factor(self, factor_id: str) -> None:
        """Remove a factor from the matrix."""
        if factor_id not in self._signals:
            return
        self._signals.pop(factor_id, None)
        self._ranked.pop(factor_id, None)
        self._factor_ids = [fid for fid in self._factor_ids if fid != factor_id]
        # Remove cached correlations involving this factor
        keys_to_remove = [
            k for k in self._corr_cache if factor_id in k
        ]
        for k in keys_to_remove:
            del self._corr_cache[k]

    def get_correlation(self, id_a: str, id_b: str) -> float:
        """Get cached correlation between two factors."""
        key = (min(id_a, id_b), max(id_a, id_b))
        if key in self._corr_cache:
            return self._corr_cache[key]
        if id_a == id_b:
            return 1.0
        return 0.0

    def get_max_correlation(self, factor_id: str) -> tuple[float, str | None]:
        """Get the maximum absolute correlation of a factor with all others.

        Returns
        -------
        tuple of (max_abs_corr, most_correlated_factor_id)
        """
        max_corr = 0.0
        max_id: str | None = None
        for other_id in self._factor_ids:
            if other_id == factor_id:
                continue
            corr = abs(self.get_correlation(factor_id, other_id))
            if corr > max_corr:
                max_corr = corr
                max_id = other_id
        return max_corr, max_id

    def to_matrix(self) -> np.ndarray:
        """Return the full correlation matrix as a numpy array.

        Returns
        -------
        np.ndarray, shape (N, N)
        """
        N = len(self._factor_ids)
        mat = np.eye(N, dtype=np.float64)
        for i in range(N):
            for j in range(i + 1, N):
                corr = self.get_correlation(self._factor_ids[i], self._factor_ids[j])
                mat[i, j] = corr
                mat[j, i] = corr
        return mat


# ---------------------------------------------------------------------------
# Torch backend (optional)
# ---------------------------------------------------------------------------

def _try_torch_rank_correlation(
    candidate: np.ndarray,
    library: np.ndarray,
) -> np.ndarray | None:
    """Attempt to compute rank correlations using PyTorch for GPU acceleration.

    Falls back to None if torch is not available.

    Parameters
    ----------
    candidate : np.ndarray, shape (M, T)
    library : np.ndarray, shape (N, M, T)

    Returns
    -------
    np.ndarray, shape (N,) or None if torch unavailable.
    """
    try:
        import torch
    except ImportError:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N, M, T = library.shape
    cand_t = torch.from_numpy(candidate).to(device, dtype=torch.float64)
    lib_t = torch.from_numpy(library).to(device, dtype=torch.float64)

    correlations = torch.zeros(N, dtype=torch.float64, device=device)

    for t in range(T):
        c_col = cand_t[:, t]
        l_cols = lib_t[:, :, t]  # (N, M)

        # Skip if too many NaN
        c_valid = ~torch.isnan(c_col)
        if c_valid.sum() < 5:
            continue

        # Rank the candidate column
        c_col[c_valid].argsort().argsort().float() + 1.0

        for i in range(N):
            l_col = l_cols[i]
            valid = c_valid & ~torch.isnan(l_col)
            n = valid.sum()
            if n < 5:
                continue
            # Rank both
            c_v = c_col[valid]
            l_v = l_col[valid]
            c_rank = c_v.argsort().argsort().float() + 1.0
            l_rank = l_v.argsort().argsort().float() + 1.0
            c_m = c_rank - c_rank.mean()
            l_m = l_rank - l_rank.mean()
            denom = torch.sqrt((c_m ** 2).sum() * (l_m ** 2).sum())
            if denom > 1e-12:
                correlations[i] += (c_m * l_m).sum() / denom

    correlations /= max(T, 1)
    return correlations.cpu().numpy()


def compute_correlation_batch(
    candidate: np.ndarray,
    library: np.ndarray,
    backend: str = "numpy",
) -> np.ndarray:
    """Compute correlations between candidate and library, with backend selection.

    Parameters
    ----------
    candidate : np.ndarray, shape (M, T)
    library : np.ndarray, shape (N, M, T)
    backend : str
        "numpy" or "gpu"

    Returns
    -------
    np.ndarray, shape (N,)
    """
    if backend == "gpu":
        result = _try_torch_rank_correlation(candidate, library)
        if result is not None:
            return result

    return batch_spearman_correlation(candidate, library)
