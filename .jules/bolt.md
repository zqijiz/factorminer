## 2024-05-17 - Optimize portfolio percentile ranking
**Learning:** Manual Python `while` loops for computing cross-sectional ranks and handling ties (e.g. `_rank_array` in portfolio backtests) create a severe bottleneck because they run sequentially for every timestamp.
**Action:** Replace manual ranking loops with `scipy.stats.rankdata(x, method='average')`. It computes ranks fully in C and scales ties correctly, offering a >6x speedup during portfolio backtesting routines without changing correct output.
