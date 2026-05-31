"""Element-wise arithmetic operators (unary and binary).

Every function accepts arrays of shape ``(M, T)`` and returns the same shape.
Both NumPy and PyTorch implementations are provided.
"""

from __future__ import annotations

from typing import Union

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

Array = Union[np.ndarray, "torch.Tensor"]

# ---- helpers ---------------------------------------------------------------

_EPS_NP = np.float32(1e-10)


def _eps(x: Array) -> float:
    return 1e-10


# ---- NumPy implementations ------------------------------------------------

def add_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.add(x, y)


def sub_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.subtract(x, y)


def mul_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.multiply(x, y)


def div_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Use np.divide with where mask to avoid expensive intermediate masked array allocations
    out = np.full(np.broadcast_shapes(np.shape(x), np.shape(y)), np.nan, dtype=np.result_type(x, y, float))
    np.divide(x, y, out=out, where=np.abs(y) > _EPS_NP)
    return out


def neg_np(x: np.ndarray) -> np.ndarray:
    return np.negative(x)


def abs_np(x: np.ndarray) -> np.ndarray:
    return np.abs(x)


def sign_np(x: np.ndarray) -> np.ndarray:
    return np.sign(x)


def log_np(x: np.ndarray) -> np.ndarray:
    """log(1 + |x|) * sign(x) -- safe log that handles negatives."""
    return np.log1p(np.abs(x)) * np.sign(x)


def sqrt_np(x: np.ndarray) -> np.ndarray:
    """sqrt(|x|) * sign(x)."""
    return np.sqrt(np.abs(x)) * np.sign(x)


def square_np(x: np.ndarray) -> np.ndarray:
    return np.square(x)


def inv_np(x: np.ndarray) -> np.ndarray:
    # Use np.divide with where mask to avoid expensive intermediate masked array allocations
    out = np.full_like(x, np.nan, dtype=np.float64)
    np.divide(1.0, x, out=out, where=np.abs(x) > _EPS_NP)
    return out


def pow_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """x^y with safe handling."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isnan(x) | np.isnan(y), np.nan, np.power(np.abs(x), y) * np.sign(x))


def max_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.fmax(x, y)


def min_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.fmin(x, y)


def clip_np(x: np.ndarray, lower: float = -3.0, upper: float = 3.0) -> np.ndarray:
    return np.clip(x, lower, upper)


def exp_np(x: np.ndarray) -> np.ndarray:
    """Clamped exp to avoid overflow."""
    return np.exp(np.clip(x, -50.0, 50.0))


def tanh_np(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def signed_power_np(x: np.ndarray, e: float = 2.0) -> np.ndarray:
    return np.sign(x) * np.power(np.abs(x), e)


def power_np(x: np.ndarray, e: float = 2.0) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.power(x, e)


# ---- PyTorch (GPU) implementations ----------------------------------------

def add_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


def sub_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x - y


def mul_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x * y


def div_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Use torch.where to avoid expensive boolean slice indexing and intermediate allocations
    return torch.where(y.abs() > 1e-10, x / y, torch.tensor(float("nan"), device=x.device))


def neg_torch(x: torch.Tensor) -> torch.Tensor:
    return -x


def abs_torch(x: torch.Tensor) -> torch.Tensor:
    return x.abs()


def sign_torch(x: torch.Tensor) -> torch.Tensor:
    return x.sign()


def log_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(x.abs()) * x.sign()


def sqrt_torch(x: torch.Tensor) -> torch.Tensor:
    return x.abs().sqrt() * x.sign()


def square_torch(x: torch.Tensor) -> torch.Tensor:
    return x * x


def inv_torch(x: torch.Tensor) -> torch.Tensor:
    # Use torch.where to avoid expensive boolean slice indexing and intermediate allocations
    return torch.where(x.abs() > 1e-10, 1.0 / x, torch.tensor(float("nan"), device=x.device))


def pow_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    safe = x.abs().pow(y) * x.sign()
    return torch.where(torch.isnan(x) | torch.isnan(y), torch.tensor(float("nan"), device=x.device), safe)


def max_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.fmax(x, y)


def min_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.fmin(x, y)


def clip_torch(x: torch.Tensor, lower: float = -3.0, upper: float = 3.0) -> torch.Tensor:
    return x.clamp(lower, upper)


def exp_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x.clamp(-50.0, 50.0))


def tanh_torch(x: torch.Tensor) -> torch.Tensor:
    return x.tanh()


def signed_power_torch(x: torch.Tensor, e: float = 2.0) -> torch.Tensor:
    return x.sign() * x.abs().pow(e)


def power_torch(x: torch.Tensor, e: float = 2.0) -> torch.Tensor:
    return x.pow(e)


# ---- Registration table ----------------------------------------------------
# Maps operator name -> (numpy_fn, torch_fn)

ARITHMETIC_OPS = {
    "Add": (add_np, add_torch),
    "Sub": (sub_np, sub_torch),
    "Mul": (mul_np, mul_torch),
    "Div": (div_np, div_torch),
    "Neg": (neg_np, neg_torch),
    "Abs": (abs_np, abs_torch),
    "Sign": (sign_np, sign_torch),
    "Log": (log_np, log_torch),
    "Sqrt": (sqrt_np, sqrt_torch),
    "Square": (square_np, square_torch),
    "Inv": (inv_np, inv_torch),
    "Pow": (pow_np, pow_torch),
    "Max": (max_np, max_torch),
    "Min": (min_np, min_torch),
    "Clip": (clip_np, clip_torch),
}
