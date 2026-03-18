from __future__ import annotations

from typing import Optional, Tuple
import os

import numpy as np

from .types import RawLayoutParams, VolumeParams


def find_header_end(data: bytes, header_end_marker: str) -> int:
    if not header_end_marker or not data:
        return 0

    import re

    marker_text = header_end_marker
    marker_bytes = marker_text.encode("utf-8")

    pos_exact = data.find(marker_bytes)
    if pos_exact != -1:
        end_pos = pos_exact + len(marker_bytes)
        while (
            end_pos < len(data) and data[end_pos : end_pos + 1] in (b"\n", b"\r")
        ):
            end_pos += 1
        return end_pos

    marker_lines = [line for line in marker_text.splitlines() if line.strip()]
    if not marker_lines:
        return 0

    alt_candidates = []
    if len(marker_lines) >= 2:
        alt_candidates = [
            "\n".join(marker_lines).encode("utf-8"),
            "\n\n".join(marker_lines).encode("utf-8"),
            "\r\n".join(marker_lines).encode("utf-8"),
            "\r\n\r\n".join(marker_lines).encode("utf-8"),
        ]

    for alt in alt_candidates:
        alt_pos = data.find(alt)
        if alt_pos != -1:
            end_pos = alt_pos + len(alt)
            while (
                end_pos < len(data)
                and data[end_pos : end_pos + 1] in (b"\n", b"\r")
            ):
                end_pos += 1
            return end_pos

    regex_pattern = rb""
    for i, line in enumerate(marker_lines):
        if i > 0:
            regex_pattern += rb"\r?\n(?:\r?\n)*"
        regex_pattern += re.escape(line.encode("utf-8"))

    regex_match = re.search(regex_pattern, data, flags=re.DOTALL)
    if regex_match is not None:
        end_pos = regex_match.end()
        while end_pos < len(data) and data[end_pos : end_pos + 1] in (b"\n", b"\r"):
            end_pos += 1
        return end_pos

    return 0


def get_pixel_info(pixel_type: str, endianness: str):
    type_map = {
        "8 bit unsigned": (np.uint8, 1, 1),
        "8 bit signed": (np.int8, 1, 1),
        "16 bit unsigned": (np.uint16, 2, 1),
        "16 bit signed": (np.int16, 2, 1),
        "float": (np.float32, 4, 1),
        "double": (np.float64, 8, 1),
        "24 bit RGB": (np.uint8, 1, 3),
    }

    dtype, byte_size, components = type_map[pixel_type]

    if endianness == "Big endian" and byte_size > 1:
        dtype_map = {
            np.uint16: ">u2",
            np.int16: ">i2",
            np.float32: ">f4",
            np.float64: ">f8",
        }
        dtype = dtype_map.get(dtype, dtype)

    return dtype, byte_size, components


def read_raw_volume(
    input_path: str,
    volume_params: VolumeParams,
    raw_layout: RawLayoutParams,
) -> Optional[np.ndarray]:
    if not os.path.exists(input_path):
        return None

    header_size = raw_layout.header_size

    if (raw_layout.header_end_marker or "").strip():
        raw_bytes_for_search = b""
        try:
            with open(input_path, "rb") as f:
                raw_bytes_for_search = f.read(512 * 1024)
        except OSError:
            raw_bytes_for_search = b""

        header_size_from_marker = find_header_end(raw_bytes_for_search, raw_layout.header_end_marker)
        if header_size_from_marker > 0:
            header_size = header_size_from_marker

    if raw_layout.use_header_offset:
        header_size += int(raw_layout.header_offset)

    dtype, byte_size, components = get_pixel_info(volume_params.pixel_type, volume_params.endianness)

    row_data_size = volume_params.width * byte_size * components
    if raw_layout.row_stride > 0:
        effective_row_stride = raw_layout.row_stride
    else:
        effective_row_stride = row_data_size + raw_layout.row_padding

    slice_data_size = volume_params.height * effective_row_stride
    effective_slice_stride = slice_data_size + raw_layout.slice_stride

    total_header_size = header_size + raw_layout.skip_slices * effective_slice_stride

    file_size = os.path.getsize(input_path)
    available_data_size = file_size - total_header_size - raw_layout.footer_size
    if available_data_size <= 0:
        return None

    max_slices = int((available_data_size + raw_layout.slice_stride) / effective_slice_stride)
    final_depth = min(volume_params.depth, max_slices)
    if final_depth <= 0:
        return None

    image_slices = []
    with open(input_path, "rb") as f:
        for slice_idx in range(final_depth):
            slice_position = total_header_size + slice_idx * effective_slice_stride

            if effective_row_stride == row_data_size:
                f.seek(slice_position)
                slice_bytes = f.read(volume_params.height * row_data_size)
                if len(slice_bytes) < volume_params.height * row_data_size:
                    break
                slice_data = np.frombuffer(slice_bytes, dtype=dtype)
            else:
                row_data_list = []
                for row_idx in range(volume_params.height):
                    row_position = slice_position + row_idx * effective_row_stride
                    f.seek(row_position)
                    row_bytes = f.read(row_data_size)
                    if len(row_bytes) < row_data_size:
                        break
                    row_data_list.append(np.frombuffer(row_bytes, dtype=dtype))

                if len(row_data_list) != volume_params.height:
                    break
                slice_data = np.concatenate(row_data_list)

            if components == 1:
                slice_data = slice_data.reshape((volume_params.height, volume_params.width))
            else:
                slice_data = slice_data.reshape((volume_params.height, volume_params.width, components))

            image_slices.append(slice_data)

    if not image_slices:
        return None

    return np.array(image_slices)
