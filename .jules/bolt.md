# Bolt's Performance Journal

## 2024-05-15 - Vectorizing Cross-sectional Ranking Operations
**Learning:** Cross-sectional financial time-series ranking and quantile binning in numpy can be massively sped up (from O(N*T) looping to vectorized operation) using `scipy.stats.rankdata(..., axis=0, nan_policy='omit', method='ordinal')`. The `method='ordinal'` parameter accurately replicates the original `argsort().argsort()` tie-breaking behavior, while properly preserving `np.nan` positions when combined with `nan_policy='omit'`.
**Action:** Always favor vectorization over manual time-step loops in array processing. Use explicit masking with `np.errstate(divide='ignore', invalid='ignore')` for zero-division cases during vectorization rather than defensive looping.

## 2024-05-15 - Vectorizing Cross-sectional Ranking Operations
**Learning:** Adding new dependencies for performance is risky when pure numpy can do the job almost as well. `x.argsort(axis=0).argsort(axis=0)` natively replicates the `scipy.stats.rankdata(..., method='ordinal')` tie-breaking behavior and sorts NaNs to the end without adding new dependencies.
**Action:** Always verify if standard library or already-included native numpy methods can achieve the same vectorization before reaching for external dependencies like `scipy`, especially when minimizing dependencies is a constraint.
