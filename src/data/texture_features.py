"""Stream 3 input construction: texture features via sliding-window GLCM stats + a Gabor
filter bank. This is the branch most directly aimed at fixing grass recall — grass blade
texture, broadleaf's flat leaves, and soybean's compound leaflets differ mainly in texture,
not just color.

Output per image: (H, W, TEXTURE_CHANNELS) float32:
  - 4 GLCM maps (contrast, homogeneity, energy, correlation), computed on a downsampled
    grayscale image via sliding-window GLCM then upsampled back to HxW (GLCM is expensive at
    full resolution: we compute it on a coarse grid of windows for speed).
  - Gabor filter bank responses: 4 orientations x 2 frequencies = 8 channels.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

GLCM_PROPS = ["contrast", "homogeneity", "energy", "correlation"]
GABOR_ORIENTATIONS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GABOR_FREQUENCIES = [0.1, 0.3]

TEXTURE_CHANNELS = len(GLCM_PROPS) + len(GABOR_ORIENTATIONS) * len(GABOR_FREQUENCIES)  # 4 + 8 = 12

_GLCM_GRID = 16  # compute GLCM on a 16x16 grid of image patches (coarse, then upsample)
_GLCM_LEVELS = 32  # quantize grayscale to this many levels for GLCM speed


def _to_gray_u8(img_float01: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor((img_float01 * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return gray


def _sliding_glcm_maps(gray_u8: np.ndarray) -> np.ndarray:
    h, w = gray_u8.shape
    grid = _GLCM_GRID
    win_h, win_w = h // grid, w // grid
    quant = (gray_u8.astype(np.float32) / 255.0 * (_GLCM_LEVELS - 1)).astype(np.uint8)

    maps = np.zeros((grid, grid, len(GLCM_PROPS)), dtype=np.float32)
    for i in range(grid):
        for j in range(grid):
            patch = quant[i * win_h : (i + 1) * win_h, j * win_w : (j + 1) * win_w]
            if patch.size == 0:
                continue
            glcm = graycomatrix(
                patch, distances=[1], angles=[0], levels=_GLCM_LEVELS, symmetric=True, normed=True
            )
            for k, prop in enumerate(GLCM_PROPS):
                maps[i, j, k] = graycoprops(glcm, prop)[0, 0]

    full = cv2.resize(maps, (w, h), interpolation=cv2.INTER_LINEAR)
    if full.ndim == 2:  # cv2.resize can drop the channel dim when it's 1
        full = full[..., None]
    return full.astype(np.float32)


def _gabor_bank(gray_u8: np.ndarray) -> np.ndarray:
    responses = []
    for theta in GABOR_ORIENTATIONS:
        for freq in GABOR_FREQUENCIES:
            kernel = cv2.getGaborKernel((15, 15), sigma=4.0, theta=theta, lambd=1 / freq, gamma=0.5, psi=0)
            resp = cv2.filter2D(gray_u8, cv2.CV_32F, kernel)
            responses.append(resp)
    stack = np.stack(responses, axis=-1)
    # Normalize each channel to a stable range.
    stack = stack / (np.abs(stack).max() + 1e-6)
    return stack.astype(np.float32)


def compute_texture_stack(img_float01: np.ndarray) -> np.ndarray:
    """img_float01: (H, W, 3) float32 RGB in [0,1]. Returns (H, W, TEXTURE_CHANNELS) float32."""
    gray = _to_gray_u8(img_float01)
    glcm_maps = _sliding_glcm_maps(gray)
    gabor_maps = _gabor_bank(gray)
    return np.concatenate([glcm_maps, gabor_maps], axis=-1)
