## 2024-07-24 - Performance Pattern: Safe Division in NumPy
**Learning:** In NumPy, when replacing zeros or performing division that guards against division by zero, nested `np.where(mask, a / np.where(mask, b, 1.0), 0.0)` constructs allocate expensive intermediate arrays and slow down operations.
**Action:** Replace explicit boolean array masking in safe divisions with `np.divide(a, b, out=out, where=mask)` and preallocate `out` efficiently with `np.zeros` using `np.broadcast_shapes`. This avoids intermediate array allocations and evaluates invalid divisions safely.
