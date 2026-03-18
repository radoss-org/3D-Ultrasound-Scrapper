from __future__ import annotations

import numpy as np


def apply_enhancement(
    image: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
) -> np.ndarray:
    enhanced = image.astype(np.float64)
    enhanced = enhanced + brightness

    if contrast != 1.0:
        middle = np.mean(enhanced)
        enhanced = middle + (enhanced - middle) * contrast

    if gamma != 1.0:
        min_val = np.min(enhanced)
        max_val = np.max(enhanced)
        if max_val > min_val:
            normalized = (enhanced - min_val) / (max_val - min_val)
            gamma_corrected = np.power(normalized, gamma)
            enhanced = min_val + gamma_corrected * (max_val - min_val)

    max_val = (
        np.iinfo(image.dtype).max
        if np.issubdtype(image.dtype, np.integer)
        else 1.0
    )
    enhanced = np.clip(enhanced, 0, max_val)

    return enhanced.astype(image.dtype)
