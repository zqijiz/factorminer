
## 2025-05-18 - [Optimize Cumulative Max/Min using `np.fmax.accumulate`]
**Learning:** For cumulative or rolling max/min operations on NumPy arrays containing NaNs, using `np.fmax.accumulate(x, axis=1)` and `np.fmin.accumulate(x, axis=1)` correctly handles missing values exactly like Python-level `np.fmax` / `np.fmin` loops, and provides significant speedups (~5-6x) over the naive `for t in range(T)` Python loops.
**Action:** Always prefer `np.fmax.accumulate` / `np.fmin.accumulate` instead of iterating sequentially over axes for cumulative comparisons when working with 2D time series arrays that contain missing values (NaNs).
