## 2024-05-24 - Cross-sectional ranking optimization
**Learning:** For 1D array cross-sectional ranking with tie-breaking, using `scipy.stats.rankdata(x)` (which defaults to `method='average'`) is significantly faster (~5x) than manual Python `while` loops iterating through sorted arrays to resolve ties.
**Action:** Replace `_rank_array` implementation in `factorminer/evaluation/portfolio.py` with `scipy.stats.rankdata(x)`.
