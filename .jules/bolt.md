
## 2024-05-23 - Safe Division NumPy Pattern
**Learning:** In NumPy, when you need to perform safe division (returning 0 when the denominator is near zero), utilizing `np.divide(a, b, out=np.zeros_like(a), where=np.abs(b) > _EPS)` is significantly faster (>30%) than using nested `np.where` constructs (e.g., `np.where(np.abs(b) > _EPS, a / np.where(np.abs(b) > _EPS, b, 1.0), 0.0)`). This is because the `out/where` pattern avoids creating intermediate arrays for the masked division.
**Action:** Use `np.divide` with the `out` and `where` keyword arguments for conditional division over arrays whenever avoiding zero-division is required.
