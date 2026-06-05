## 2024-06-05 - NumPy Advanced Indexing & Ufuncs

**Learning:** Vectorizing cross-sectional rankings utilizing `np.argsort` and `np.put_along_axis` offers significant performance benefits over looping across the time dimension `T`. Similarly, using native accumulation operations like `np.fmax.accumulate` replaces iterative slicing with C-level loops, achieving magnitudes of speedup. Additionally, utilizing the `out` and `where` parameters in ufuncs (like `np.divide`) is optimal for preventing memory allocations on discarded values.

**Action:** Whenever identifying sequential `for` loops or repeated intermediate array allocations in NumPy sequences, always replace them with low-level ufuncs (accumulate, reduce) or pre-allocated output arrays using conditional `where` arguments to speed up the loop execution.
