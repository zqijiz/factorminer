
## 2024-06-30 - Move Time Series Aggregations Out of Loops
**Learning:** Computing sequential rolling aggregations like `np.nansum` or `diff().abs().nansum()` directly inside a python time loop forces expensive slice allocations, python loop overhead, and in PyTorch, numerous tiny kernel launches causing massive host-device sync bottlenecks.
**Action:** Always pre-calculate aggregations outside of sequential time loops using vectorized `np.cumsum`/`torch.cumsum` (padded appropriately) or PyTorch's `unfold(...).nansum()` to create O(1) loop-time access patterns. Replace boolean slice assignment inside loops with branchless `where` operations to prevent reallocation overhead.
