## 2024-06-04 - Optimize `cummax` and `cummin` using `np.fmax.accumulate` / `np.fmin.accumulate`
**Learning:** Python loops over time dimension in time series operations `cummax_np` and `cummin_np` are significantly slower than native NumPy ufunc accumulation methods, particularly when arrays contain NaNs.
**Action:** Replaced loop-based cumulative maximum and minimum over time operations with `np.fmax.accumulate` and `np.fmin.accumulate`.
