from __future__ import annotations

from typing import Iterable, Optional
import base64
import os

import numpy as np

from .types import NrrdParams, VolumeParams


_TYPE_MAP = {
    "8 bit unsigned": "uchar",
    "8 bit signed": "signed char",
    "16 bit unsigned": "ushort",
    "16 bit signed": "short",
    "float": "float",
    "double": "double",
    "24 bit RGB": "uchar",
}

_BASE_DTYPE_MAP = {
    "8 bit unsigned": np.uint8,
    "8 bit signed": np.int8,
    "16 bit unsigned": np.uint16,
    "16 bit signed": np.int16,
    "float": np.float32,
    "double": np.float64,
    "24 bit RGB": np.uint8,
}


def build_nrrd_header_text(
    volume: np.ndarray,
    volume_params: VolumeParams,
    nrrd_params: NrrdParams,
    *,
    header_extras: Optional[Iterable[str]] = None,
) -> str:
    if volume.ndim == 4:
        depth, height, width, components = volume.shape
    else:
        depth, height, width = volume.shape
        components = 1

    spacing_x = volume_params.spacing_x
    spacing_y = volume_params.spacing_y
    spacing_z = volume_params.spacing_z

    endian_str = volume_params.endianness
    endian_word = "little" if endian_str == "Little endian" else "big"

    lines = []
    lines.append("NRRD0004")
    lines.append("# Complete NRRD file format specification at:")
    lines.append("# http://teem.sourceforge.net/nrrd/format.html")
    lines.append(f"type: {_TYPE_MAP[volume_params.pixel_type]}")
    lines.append("space: left-posterior-superior")

    if components > 1:
        lines.append("dimension: 4")
        lines.append(f"sizes: {components} {width} {height} {depth}")
        lines.append(
            "space directions: none "
            f"({spacing_x},0,0) "
            f"(0,{spacing_y},0) (0,0,{spacing_z})"
        )
        lines.append("kinds: vector domain domain domain")
    else:
        lines.append("dimension: 3")
        lines.append(f"sizes: {width} {height} {depth}")
        lines.append(
            "space directions: "
            f"({spacing_x},0,0) "
            f"(0,{spacing_y},0) (0,0,{spacing_z})"
        )
        lines.append("kinds: domain domain domain")

    lines.append(f"endian: {endian_word}")
    lines.append("encoding: raw")
    lines.append("space origin: (0,0,0)")

    extras = []
    if nrrd_params.header_extras:
        extras.extend(list(nrrd_params.header_extras))
    if header_extras:
        extras.extend(list(header_extras))

    if extras:
        lines.extend(list(extras))

    lines.append("")
    return "\n".join(lines)


def write_nrrd_raw_data(output_path: str, volume: np.ndarray, volume_params: VolumeParams, nrrd_params: NrrdParams) -> None:
    if volume.ndim == 4:
        depth, height, width, components = volume.shape
    else:
        depth, height, width = volume.shape
        components = 1

    base_dtype = _BASE_DTYPE_MAP[volume_params.pixel_type]

    data_to_save = volume.astype(base_dtype, copy=False)
    if volume_params.endianness == "Big endian" and data_to_save.dtype.itemsize > 1:
        data_to_save = data_to_save.byteswap().newbyteorder()

    if components > 1:
        data_to_save = np.moveaxis(data_to_save, -1, 0)

    with open(output_path, "ab") as f:
        f.write(data_to_save.tobytes())


def save_nrrd(
    output_path: str,
    volume: np.ndarray,
    volume_params: VolumeParams,
    nrrd_params: NrrdParams,
    *,
    header_extras: Optional[Iterable[str]] = None,
) -> None:
    header = build_nrrd_header_text(
        volume,
        volume_params,
        nrrd_params,
        header_extras=header_extras,
    )

    with open(output_path, "w", newline="\n") as f:
        f.write(header)

    write_nrrd_raw_data(output_path, volume, volume_params, nrrd_params)
