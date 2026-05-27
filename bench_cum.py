import numpy as np
import time

def cummax_np_orig(x: np.ndarray) -> np.ndarray:
    out = np.copy(x)
    for t in range(1, x.shape[1]):
        out[:, t] = np.fmax(out[:, t - 1], x[:, t])
    return out

def cummax_np_vec(x: np.ndarray) -> np.ndarray:
    # np.fmax ignores nans
    return np.fmax.accumulate(x, axis=1)

def cummin_np_orig(x: np.ndarray) -> np.ndarray:
    out = np.copy(x)
    for t in range(1, x.shape[1]):
        out[:, t] = np.fmin(out[:, t - 1], x[:, t])
    return out

def cummin_np_vec(x: np.ndarray) -> np.ndarray:
    return np.fmin.accumulate(x, axis=1)

np.random.seed(42)
x_large = np.random.randn(3000, 1000)
x_large[np.random.rand(3000, 1000) < 0.1] = np.nan

print("Max equal?", np.allclose(np.nan_to_num(cummax_np_orig(x_large)), np.nan_to_num(cummax_np_vec(x_large))))
print("Min equal?", np.allclose(np.nan_to_num(cummin_np_orig(x_large)), np.nan_to_num(cummin_np_vec(x_large))))

t0 = time.time()
cummax_np_orig(x_large)
print(f"Orig max: {time.time()-t0:.4f}s")

t0 = time.time()
cummax_np_vec(x_large)
print(f"Vec max: {time.time()-t0:.4f}s")

t0 = time.time()
cummin_np_orig(x_large)
print(f"Orig min: {time.time()-t0:.4f}s")

t0 = time.time()
cummin_np_vec(x_large)
print(f"Vec min: {time.time()-t0:.4f}s")
