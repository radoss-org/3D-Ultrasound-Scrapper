from __future__ import annotations

from typing import Optional
import numpy as np

from .types import CropParams, CurveParams, CornerWarpParams, OrientationParams


def apply_orientation(volume: np.ndarray, orientation: OrientationParams) -> np.ndarray:
    if volume is None or orientation.orientation_ops is None:
        return volume

    image_data = volume

    axis_map = {"z": 0, "y": 1, "x": 2}

    for op in orientation.orientation_ops:
        if not op:
            continue
        if op[0] == "flip":
            _, axis = op
            image_data = np.flip(image_data, axis=axis_map[axis])
        elif op[0] == "rotate":
            _, axis, direction = op
            k = 1 if direction > 0 else 3
            if axis == "z":
                image_data = np.rot90(image_data, k=k, axes=(1, 2))
            elif axis == "x":
                image_data = np.rot90(image_data, k=k, axes=(0, 1))
            elif axis == "y":
                image_data = np.rot90(image_data, k=k, axes=(0, 2))

    return image_data


def _identity_corners_for_volume(volume: np.ndarray) -> np.ndarray:
    d, h, w = volume.shape[:3]
    cp = np.zeros((8, 3), dtype=np.float64)
    for idx in range(8):
        ix, iy, iz = (idx >> 0) & 1, (idx >> 1) & 1, (idx >> 2) & 1
        cp[idx] = [ix * (w - 1), iy * (h - 1), iz * (d - 1)]
    return cp


def _prepare_corner_positions(volume: np.ndarray, warp: CornerWarpParams) -> np.ndarray:
    if warp.corner_positions is not None:
        return np.array(warp.corner_positions, dtype=np.float64)
    return _identity_corners_for_volume(volume)


def apply_curve_deformation(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    D: int,
    H: int,
    W: int,
    curve: CurveParams,
):
    if not (
        abs(curve.curve_x_pos) > 1e-6
        or abs(curve.curve_x_neg) > 1e-6
        or abs(curve.curve_y_pos) > 1e-6
        or abs(curve.curve_y_neg) > 1e-6
        or abs(curve.curve_z_pos) > 1e-6
        or abs(curve.curve_z_neg) > 1e-6
    ):
        return X, Y, Z

    x_norm = X / (W - 1) if W > 1 else np.zeros_like(X)
    y_norm = Y / (H - 1) if H > 1 else np.zeros_like(Y)
    z_norm = Z / (D - 1) if D > 1 else np.zeros_like(Z)

    if abs(curve.curve_x_pos) > 1e-6 or abs(curve.curve_x_neg) > 1e-6:
        curve_x = np.where(
            x_norm >= 0.5,
            curve.curve_x_pos * (x_norm - 0.5) * 2.0,
            curve.curve_x_neg * (0.5 - x_norm) * 2.0,
        )
        Y += curve_x * (H - 1) * 0.5 * np.sin(np.pi * y_norm)
        Z += curve_x * (D - 1) * 0.5 * np.sin(np.pi * z_norm)

    if abs(curve.curve_y_pos) > 1e-6 or abs(curve.curve_y_neg) > 1e-6:
        curve_y = np.where(
            y_norm >= 0.5,
            curve.curve_y_pos * (y_norm - 0.5) * 2.0,
            curve.curve_y_neg * (0.5 - y_norm) * 2.0,
        )
        X += curve_y * (W - 1) * 0.5 * np.sin(np.pi * x_norm)
        Z += curve_y * (D - 1) * 0.5 * np.sin(np.pi * z_norm)

    if abs(curve.curve_z_pos) > 1e-6 or abs(curve.curve_z_neg) > 1e-6:
        curve_z = np.where(
            z_norm >= 0.5,
            curve.curve_z_pos * (z_norm - 0.5) * 2.0,
            curve.curve_z_neg * (0.5 - z_norm) * 2.0,
        )
        X += curve_z * (W - 1) * 0.5 * np.sin(np.pi * x_norm)
        Y += curve_z * (H - 1) * 0.5 * np.sin(np.pi * y_norm)

    return X, Y, Z


def _warp_slice(vol: np.ndarray, z_idx: int, cp: np.ndarray, curve: CurveParams) -> np.ndarray:
    try:
        from scipy import ndimage
    except ImportError:
        ndimage = None

    D, H, W = vol.shape[:3]
    C = 1 if vol.ndim == 3 else vol.shape[3]

    xs = np.linspace(0.0, 1.0, W) if W > 1 else np.zeros(W)
    ys = np.linspace(0.0, 1.0, H) if H > 1 else np.zeros(H)
    wval = 0.0 if D <= 1 else z_idx / (D - 1)

    uu, vv = np.meshgrid(xs, ys)
    s0, s1 = 1.0 - uu, uu
    t0, t1 = 1.0 - vv, vv
    p0, p1 = 1.0 - wval, wval

    X = (
        cp[0][0] * s0 * t0 * p0
        + cp[1][0] * s1 * t0 * p0
        + cp[2][0] * s0 * t1 * p0
        + cp[3][0] * s1 * t1 * p0
        + cp[4][0] * s0 * t0 * p1
        + cp[5][0] * s1 * t0 * p1
        + cp[6][0] * s0 * t1 * p1
        + cp[7][0] * s1 * t1 * p1
    )
    Y = (
        cp[0][1] * s0 * t0 * p0
        + cp[1][1] * s1 * t0 * p0
        + cp[2][1] * s0 * t1 * p0
        + cp[3][1] * s1 * t1 * p0
        + cp[4][1] * s0 * t0 * p1
        + cp[5][1] * s1 * t0 * p1
        + cp[6][1] * s0 * t1 * p1
        + cp[7][1] * s1 * t1 * p1
    )
    Z = (
        cp[0][2] * s0 * t0 * p0
        + cp[1][2] * s1 * t0 * p0
        + cp[2][2] * s0 * t1 * p0
        + cp[3][2] * s1 * t1 * p0
        + cp[4][2] * s0 * t0 * p1
        + cp[5][2] * s1 * t0 * p1
        + cp[6][2] * s0 * t1 * p1
        + cp[7][2] * s1 * t1 * p1
    )

    X, Y, Z = apply_curve_deformation(X, Y, Z, D, H, W, curve)

    if ndimage is None:
        xi = np.clip(np.rint(X), 0, W - 1).astype(int)
        yi = np.clip(np.rint(Y), 0, H - 1).astype(int)
        zi = np.clip(np.rint(Z), 0, D - 1).astype(int)
        if C == 1:
            return vol[zi, yi, xi]

        out = np.zeros((H, W, C), dtype=vol.dtype)
        for c in range(C):
            out[..., c] = vol[zi, yi, xi, c]
        return out

    if C == 1:
        return ndimage.map_coordinates(
            vol, [Z, Y, X], order=1, mode="constant", cval=0.0
        )

    out = np.zeros((H, W, C), dtype=np.float64)
    for c in range(C):
        out[..., c] = ndimage.map_coordinates(
            vol[..., c], [Z, Y, X], order=1, mode="constant", cval=0.0
        )

    if np.issubdtype(vol.dtype, np.integer):
        info = np.iinfo(vol.dtype)
        out = np.clip(out, info.min, info.max).astype(vol.dtype)

    return out.astype(vol.dtype)


def warp_slice(volume: np.ndarray, z_idx: int, warp: CornerWarpParams, curve: CurveParams) -> np.ndarray:
    """Warp a single z-slice using the shared 3D corner mapping + curve deformation."""
    cp = _prepare_corner_positions(volume, warp)
    return _warp_slice(volume, z_idx, cp, curve)


def warp_volume_slices(
    volume: np.ndarray,
    warp: CornerWarpParams,
    curve: CurveParams,
) -> np.ndarray:
    if volume is None:
        return None

    cp = _prepare_corner_positions(volume, warp)
    D = volume.shape[0]

    slices = []
    for z in range(D):
        slices.append(_warp_slice(volume, z, cp, curve))
    return np.stack(slices, axis=0)


def apply_crop_to_slice(slice_data: np.ndarray, crop: CropParams) -> np.ndarray:
    if not (crop.crop_top or crop.crop_bottom or crop.crop_left or crop.crop_right):
        return slice_data

    H, W = slice_data.shape[:2]
    if crop.crop_top + crop.crop_bottom >= H:
        return slice_data
    if crop.crop_left + crop.crop_right >= W:
        return slice_data

    if slice_data.ndim == 2:
        return slice_data[
            crop.crop_top : H - crop.crop_bottom,
            crop.crop_left : W - crop.crop_right,
        ]

    return slice_data[
        crop.crop_top : H - crop.crop_bottom,
        crop.crop_left : W - crop.crop_right,
        :,
    ]


def apply_crop(volume: np.ndarray, crop: CropParams) -> np.ndarray:
    if volume is None:
        return None
    if not (crop.crop_top or crop.crop_bottom or crop.crop_left or crop.crop_right):
        return volume

    slices = [apply_crop_to_slice(volume[z], crop) for z in range(volume.shape[0])]
    return np.stack(slices, axis=0)


def are_corners_identity(
    volume: np.ndarray,
    corner_positions: np.ndarray,
) -> bool:
    """Return True when *corner_positions* matches the identity corners for *volume*."""
    default_corners = _identity_corners_for_volume(volume)
    return np.array_equal(corner_positions, default_corners)


def build_export_volume(
    volume: np.ndarray,
    warp: CornerWarpParams,
    curve: CurveParams,
    crop: CropParams,
) -> np.ndarray:
    """Apply corner warp + curve deformation and then crop (if non-identity / non-zero)."""
    cp = _prepare_corner_positions(volume, warp)
    needs_warp = not np.array_equal(cp, _identity_corners_for_volume(volume))
    needs_crop = crop.crop_top or crop.crop_bottom or crop.crop_left or crop.crop_right

    if not needs_warp and not needs_crop:
        return volume

    result = _warp_volume_from_cp(volume, cp, curve) if needs_warp else volume
    result = apply_crop(result, crop) if needs_crop else result
    return result


def _warp_volume_from_cp(
    volume: np.ndarray,
    cp: np.ndarray,
    curve: CurveParams,
) -> np.ndarray:
    """Warp every slice using pre-computed corner positions."""
    slices = [_warp_slice(volume, z, cp, curve) for z in range(volume.shape[0])]
    return np.stack(slices, axis=0)
