from __future__ import annotations

import numpy as np


def generate_toy_linear_data(
    n_samples: int, n_features: int, noise: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, n_features))
    coefficients = np.arange(1.0, n_features + 1.0)
    intercept = 3.0
    y = x @ coefficients + intercept + rng.normal(scale=noise, size=n_samples)
    return x, y
