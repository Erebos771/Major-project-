"""Stream 2 input construction: derived vegetation/color-index channel stack.

Per RGB image (float32, [0,1], HWC) computes and stacks:
  - ExG   (Excess Green Index)
  - ExG-ExR
  - VARI  (Visible Atmospherically Resistant Index)
  - HSV   (3 channels)
  - Lab   (3 channels)

Output: (H, W, 8) float32 array: [ExG, ExG-ExR, VARI, H, S, V, L, a, b] would be 9 channels;
we keep it explicit below (9 channels total) rather than hand-wave a count.
"""
from __future__ import annotations

import cv2
import numpy as np

VEG_CHANNELS = 9  # ExG, ExG-ExR, VARI, H, S, V, L, a, b


def compute_vegetation_stack(img_float01: np.ndarray) -> np.ndarray:
    """img_float01: (H, W, 3) float32 RGB in [0,1]. Returns (H, W, VEG_CHANNELS) float32."""
    r, g, b = img_float01[..., 0], img_float01[..., 1], img_float01[..., 2]
    eps = 1e-6

    exg = 2 * g - r - b
    exr = 1.4 * r - g
    exg_exr = exg - exr
    vari = (g - r) / (g + r - b + eps)

    img_u8 = (img_float01 * 255).clip(0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0

    stack = np.stack(
        [exg, exg_exr, vari, hsv[..., 0], hsv[..., 1], hsv[..., 2], lab[..., 0], lab[..., 1], lab[..., 2]],
        axis=-1,
    ).astype(np.float32)
    return stack
