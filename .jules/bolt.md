
## 2025-02-20 - [Performance Optimization: Smoothing Operators]
**Learning:** PyTorch slice indexing (`x[:, t - window:t + 1]`) and reduction operations (`nansum`) inside a loop launch multiple small kernels which severely blocks performance due to host-device synchronization and kernel overhead.
**Action:** Replace in-loop aggregations with pre-calculated tensors using `unfold` or `cumsum` outside the loop, and use branchless assignments (`torch.where`) instead of boolean array assignments (`out[valid, t] = ...`) inside sequential loops.

## 2025-02-20 - [Performance Optimization: Numpy Masking vs Where]
**Learning:** In tight numpy loops, boolean masking assignments like `out[mask, t] = val[mask]` are slower than full-column branchless `np.where(mask, val, out[:, t])` because they avoid intermediate dynamic-sized array allocations.
**Action:** Use branchless `np.where` for updates inside time-series loops rather than subsetting arrays with boolean masks.
