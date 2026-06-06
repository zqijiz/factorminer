
## 2025-06-06 - [np.fmax/fmin.accumulate avoids sequential time loops]
**Learning:** For rolling cumulative maximum/minimum operations (`CumMax`, `CumMin`) on NumPy arrays containing NaNs, using `np.fmax.accumulate(x, axis=1)` and `np.fmin.accumulate(x, axis=1)` handles missing values correctly natively and avoids iterating through the sequential time dimension using a Python for loop. This typically yields a massive ~9-10x performance speedup on large 2D arrays since vectorizing the accumulation entirely avoids the Python execution overhead.
**Action:** When implementing rolling or sequential operators that aggregate elements from beginning of series, consider `np.accumulate` ufuncs over manual Python loops.
