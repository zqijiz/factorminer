## 2026-05-28 - Optimize KAMA sequential loop by vectorizing prep calculations
**Learning:** In Kaufman Adaptive Moving Average (KAMA), calculating volatility and direction inside the sequential time loop creates significant overhead (especially slicing 'x[:, t-window:t+1]').
**Action:** Pre-calculate the volatility and direction using cumulative sums ('np.nancumsum') before the sequential loop, moving O(T) repeated slice operations out of the tight loop. Applying this pattern (along with branchless 'np.where' assignments) can yield speedups around 1.7x to 1.85x on large arrays.
