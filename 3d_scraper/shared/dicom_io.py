from __future__ import annotations

import os
from typing import Optional, Tuple

import pydicom


MIN_OB_TAG_SIZE = 100


def parse_dicom_tags(
    filepath: str,
    min_tag_size: int = MIN_OB_TAG_SIZE,
) -> Tuple[list, object]:
    """Parse DICOM file and return (tags, dataset).

    *tags* is a list of ``(name, length, tag)`` tuples for every element
    whose byte-length exceeds *min_tag_size*.
    """
    ds = pydicom.dcmread(filepath, force=True)
    tags = []
    for elem in ds.iterall():
        if elem.value is None:
            continue
        try:
            length = len(elem.value)
        except TypeError:
            continue
        if length <= min_tag_size:
            continue
        tags.append((elem.name, length, elem.tag))
    return tags, ds


def get_original_header_bytes(
    filepath: str,
    header_size: int,
    dicom_ds=None,
    dicom_tag=None,
) -> bytes:
    """Extract *header_size* bytes from a DICOM tag, falling back to the raw file."""
    if header_size <= 0:
        return b""

    if dicom_tag is not None and dicom_ds is not None and dicom_tag in dicom_ds:
        elem = dicom_ds[dicom_tag]
        if isinstance(elem.value, (bytes, bytearray)):
            return bytes(elem.value)[:header_size]
        return b""

    if not os.path.exists(filepath):
        return b""

    with open(filepath, "rb") as f:
        return f.read(header_size)
