## 2024-06-16 - Optimize cummax/cummin with accumulate
**Learning:** In NumPy, explicitly looping over columns for cumulative functions like `cummax` and `cummin` using `np.fmax` or `np.fmin` is significantly slower (~5x slower) than using the built-in vectorized `np.fmax.accumulate(x, axis=1)` and `np.fmin.accumulate(x, axis=1)`.
**Action:** Always prefer `np.ufunc.accumulate` for sequential aggregations over Python-level loops when writing NumPy operators, as it preserves identical NaN handling while leveraging C-level vectorization.
