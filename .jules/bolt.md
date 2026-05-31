## 2024-05-31 - Safe division optimization

**Learning:** When performing safe division in NumPy with conditions (e.g. avoiding division by zero), allocating an array and then masking `out[mask] = x[mask] / y[mask]` is around 30% slower than using `np.divide(x, y, out=out, where=mask)` directly, because it avoids the creation of intermediate masked arrays and boolean indexing operations. It's especially useful in arithmetic primitives where broadcast shapes must be handled carefully. Similarly, `torch.where` performs much better than boolean masking on tensors.

**Action:** Replace `out[mask] = x[mask] / y[mask]` with `np.divide` (and `torch.where`) in fundamental operators like `div_np`, `inv_np`, `div_torch`, and `inv_torch` in `factorminer/operators/arithmetic.py`. Be sure to use `np.broadcast_shapes` and `np.result_type` to correctly allocate the `out` array when inputs `x` and `y` might not be strictly the same shape (e.g., scalars broadcasting against arrays).
