## 2024-05-18 - [Vectorized Ranking Performance]
**Learning:** Using `scipy.stats.rankdata(method='average', axis=0, nan_policy='omit')` on entire financial arrays eliminates loop overhead and provides >2x speedup compared to Python loops over time steps, while maintaining mathematical correctness for ties.
**Action:** Always prefer fully vectorized operations over explicit loops for large multi-dimensional ranking tasks.
