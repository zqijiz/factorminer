
## 2023-10-27 - Vectorizing Cumulative Max/Min Operations
**Learning:** For cumulative or rolling max/min operations on NumPy arrays containing NaNs, using Python `for t in range(T)` loops is slow.
**Action:** Use `np.fmax.accumulate(x, axis=1)` and `np.fmin.accumulate(x, axis=1)` correctly handles missing values and provides significant speedups (e.g., ~4.5x) over Python-level loops.
