from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .types import (
    CornerWarpParams,
    CropParams,
    CurveParams,
    OrientationParams,
    RawLayoutParams,
    VolumeParams,
)


@dataclass
class ParsedParams:
    volume: VolumeParams
    raw_layout: RawLayoutParams
    orientation: OrientationParams
    warp: CornerWarpParams
    curve: CurveParams
    crop: CropParams

    # Display / enhancement fields (not part of the processing pipeline).
    brightness: int = 0
    contrast: int = 100
    gamma: int = 100
    window_min: int = 0
    window_max: int = 100
    current_file: str = ""


def config_to_params(config: dict[str, Any]) -> ParsedParams:
    """Build typed parameter objects from a plain config dict (JSON round-trip safe)."""
    return ParsedParams(
        volume=VolumeParams(
            width=config.get("width", 424),
            height=config.get("height", 127),
            depth=config.get("depth", 317),
            pixel_type=config.get("pixel_type", "8 bit unsigned"),
            endianness=config.get("endianness", "Little endian"),
            spacing_x=config.get("spacing_x", 1.0),
            spacing_y=config.get("spacing_y", 2.6),
            spacing_z=config.get("spacing_z", 1.0),
        ),
        raw_layout=RawLayoutParams(
            header_size=config.get("header_size", 0),
            footer_size=config.get("footer_size", 0),
            row_stride=config.get("row_stride", 0),
            row_padding=config.get("row_padding", 0),
            slice_stride=config.get("slice_stride", 0),
            skip_slices=config.get("skip_slices", 0),
            header_end_marker=config.get("header_end_marker", "[SCALPEL]\ncount=0"),
            use_header_offset=bool(config.get("use_header_offset", False)),
            header_offset=int(config.get("header_offset", 0)),
        ),
        orientation=OrientationParams(
            orientation_ops=tuple(
                tuple(op) for op in config.get("orientation_ops", [])
            ),
        ),
        warp=CornerWarpParams(
            corner_positions=(
                np.array(config["corner_positions"])
                if config.get("corner_positions")
                else None
            ),
        ),
        curve=CurveParams(
            curve_x_pos=config.get("curve_x_pos", 0) / 100.0,
            curve_x_neg=config.get("curve_x_neg", 0) / 100.0,
            curve_y_pos=config.get("curve_y_pos", 0) / 100.0,
            curve_y_neg=config.get("curve_y_neg", 0) / 100.0,
            curve_z_pos=config.get("curve_z_pos", 0) / 100.0,
            curve_z_neg=config.get("curve_z_neg", 0) / 100.0,
        ),
        crop=CropParams(
            crop_top=config.get("crop_top", 0),
            crop_bottom=config.get("crop_bottom", 0),
            crop_left=config.get("crop_left", 0),
            crop_right=config.get("crop_right", 0),
        ),
        brightness=config.get("brightness", 0),
        contrast=config.get("contrast", 100),
        gamma=config.get("gamma", 100),
        window_min=config.get("window_min", 0),
        window_max=config.get("window_max", 100),
        current_file=config.get("current_file", ""),
    )


def params_to_config(p: ParsedParams) -> dict[str, Any]:
    """Serialize a ParsedParams to a plain dict suitable for JSON."""
    return {
        "pixel_type": p.volume.pixel_type,
        "endianness": p.volume.endianness,
        "header_size": p.raw_layout.header_size,
        "footer_size": p.raw_layout.footer_size,
        "width": p.volume.width,
        "height": p.volume.height,
        "row_stride": p.raw_layout.row_stride,
        "row_padding": p.raw_layout.row_padding,
        "depth": p.volume.depth,
        "skip_slices": p.raw_layout.skip_slices,
        "slice_stride": p.raw_layout.slice_stride,
        "spacing_x": p.volume.spacing_x,
        "spacing_y": p.volume.spacing_y,
        "spacing_z": p.volume.spacing_z,
        "orientation_ops": [list(op) for op in p.orientation.orientation_ops],
        "curve_x_pos": int(p.curve.curve_x_pos * 100),
        "curve_x_neg": int(p.curve.curve_x_neg * 100),
        "curve_y_pos": int(p.curve.curve_y_pos * 100),
        "curve_y_neg": int(p.curve.curve_y_neg * 100),
        "curve_z_pos": int(p.curve.curve_z_pos * 100),
        "curve_z_neg": int(p.curve.curve_z_neg * 100),
        "crop_top": p.crop.crop_top,
        "crop_bottom": p.crop.crop_bottom,
        "crop_left": p.crop.crop_left,
        "crop_right": p.crop.crop_right,
        "brightness": p.brightness,
        "contrast": p.contrast,
        "gamma": p.gamma,
        "window_min": p.window_min,
        "window_max": p.window_max,
        "header_end_marker": p.raw_layout.header_end_marker,
        "use_header_offset": p.raw_layout.use_header_offset,
        "header_offset": p.raw_layout.header_offset,
        "current_file": p.current_file,
    }
