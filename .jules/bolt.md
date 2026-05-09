## 2024-05-10 - [Cross-sectional Ranking Optimization & Correctness]
**Learning:** Using `scipy.stats.rankdata(method='average', axis=0, nan_policy='omit')` for cross-sectional ranking is not only over 2x faster than a Python loop combining `.argsort().argsort()`, but also mathematically more correct since `argsort` incorrectly breaks ties by assigning unique ordinal ranks.
**Action:** Always prefer `scipy.stats.rankdata` over `.argsort().argsort()` when computing ranks, especially for cross-sectional variables, to gain both performance and correctness.
