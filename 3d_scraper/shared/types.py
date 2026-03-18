from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class VolumeParams:
    width: int
    height: int
    depth: int
    pixel_type: str
    endianness: str
    spacing_x: float = 1.0
    spacing_y: float = 2.6
    spacing_z: float = 1.0


@dataclass(frozen=True)
class RawLayoutParams:
    header_size: int = 0
    footer_size: int = 0
    row_stride: int = 0
    row_padding: int = 0
    slice_stride: int = 0
    skip_slices: int = 0
    header_end_marker: str = "[SCALPEL]\ncount=0"
    use_header_offset: bool = False
    header_offset: int = 0


@dataclass(frozen=True)
class OrientationParams:
    # ops shape matches current config: list of tuples like
    #   ("flip", "x"|"y"|"z")
    #   ("rotate", "x"|"y"|"z", direction)
    # JSON round-tripping turns tuples into lists.
    orientation_ops: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True)
class CornerWarpParams:
    # If corner_positions is None -> use identity corners for the current volume dims.
    # Otherwise it is expected to be shape (8, 3) (or list convertible).
    corner_positions: Optional[np.ndarray] = None


@dataclass(frozen=True)
class CurveParams:
    # Values should be in the same numeric units expected by the current math.
    # (In this repo, batch currently divides config curve_* by 100; keep that responsibility in entrypoints.)
    curve_x_pos: float = 0.0
    curve_x_neg: float = 0.0
    curve_y_pos: float = 0.0
    curve_y_neg: float = 0.0
    curve_z_pos: float = 0.0
    curve_z_neg: float = 0.0


@dataclass(frozen=True)
class CropParams:
    crop_top: int = 0
    crop_bottom: int = 0
    crop_left: int = 0
    crop_right: int = 0


@dataclass(frozen=True)
class NrrdParams:
    # Header writer uses volume_params.spacing_{x,y,z}
    # and volume_params.pixel_type/endianness.
    # Additional header fields/behavior can be passed separately as overrides.
    endian: Optional[str] = None
    # Allow callers to override header behavior differences without forking core logic.
    header_extras: tuple[str, ...] = ()
