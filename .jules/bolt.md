
## 2024-05-18 - [Optimize SMA using direct cumsum slicing]
**Learning:** In rolling window operations using cumulative sums (`np.nancumsum`), using `np.concatenate` to zero-pad the shifted cumulative sum arrays creates unnecessary intermediate memory allocations and slows down execution significantly. Since `cumsum` already contains all information, we can derive window sums safely using direct array slicing (`cs[:, window:] - cs[:, :-window]`) while avoiding `concatenate` overhead entirely.
**Action:** Always prefer direct slicing of the `cumsum` matrix to compute rolling sums or moving averages where mathematically equivalent. Avoid concatenating zero-columns for shifting array operations.
