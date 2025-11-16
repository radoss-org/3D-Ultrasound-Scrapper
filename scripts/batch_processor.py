"""
3D Ultrasound Scrapper — Batch Processor
Processes multiple raw image files using a configuration template to generate NRRD files.
"""

import glob
import json
import os
import sys

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class BatchProcessor(QThread):
    """Worker thread for batch processing files"""

    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        input_folder,
        file_pattern,
        config_file,
        output_folder,
        preserve_names=True,
        min_file_size=0,
        max_file_size=None,
        output_format="nrrd",
        video_fps=10,
        save_next_to_original=False,
    ):
        super().__init__()
        self.input_folder = input_folder
        self.file_pattern = file_pattern
        self.config_file = config_file
        self.output_folder = output_folder
        self.preserve_names = preserve_names
        self.min_file_size = min_file_size
        self.max_file_size = max_file_size
        self.output_format = output_format
        self.video_fps = (
            int(video_fps) if str(video_fps).strip().isdigit() else 10
        )
        self.save_next_to_original = save_next_to_original
        self.should_stop = False

    def stop(self):
        self.should_stop = True

    def format_file_size(self, size_bytes):
        """Format file size in human-readable format"""
        if size_bytes is None:
            return "unlimited"

        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

    def run(self):
        try:
            # Load configuration
            with open(self.config_file, "r") as f:
                config = json.load(f)

            # Find all matching files
            search_pattern = os.path.join(
                self.input_folder, "**", self.file_pattern
            )
            files = glob.glob(search_pattern, recursive=True)

            if not files:
                self.error_occurred.emit(
                    f"No files found matching pattern: {self.file_pattern}"
                )
                return

            # Apply file size filtering
            filtered_files = []
            for file_path in files:
                try:
                    file_size = os.path.getsize(file_path)

                    if file_size < self.min_file_size:
                        continue

                    if (
                        self.max_file_size is not None
                        and file_size > self.max_file_size
                    ):
                        continue

                    filtered_files.append(file_path)
                except OSError:
                    continue

            files = filtered_files

            if not files:
                size_info = f"min: {self.format_file_size(self.min_file_size)}"
                if self.max_file_size is not None:
                    size_info += (
                        f", max: {self.format_file_size(self.max_file_size)}"
                    )
                else:
                    size_info += f", max: {self.format_file_size(None)}"
                self.error_occurred.emit(
                    f"No files found matching pattern '{self.file_pattern}' "
                    f"with size constraints ({size_info})"
                )
                return

            self.status_updated.emit(f"Found {len(files)} files to process")

            # Create output folder if not saving next to original
            if not self.save_next_to_original:
                os.makedirs(self.output_folder, exist_ok=True)

            # Process each file
            for i, input_file in enumerate(files):
                if self.should_stop:
                    break

                self.status_updated.emit(
                    f"Processing: {os.path.relpath(input_file, self.input_folder)}"
                )

                try:
                    processor = BatchImageProcessor(config)
                    processor.set_input_file(input_file)
                    processed_data = processor.process_file()

                    if processed_data is not None:
                        # Determine output path
                        if self.save_next_to_original:
                            # Save in the same directory as the input file
                            output_dir = os.path.dirname(input_file)
                            base_name = os.path.splitext(
                                os.path.basename(input_file)
                            )[0]
                        else:
                            # Save in output folder with relative path
                            rel_path = os.path.relpath(
                                input_file, self.input_folder
                            )
                            rel_dir = os.path.dirname(rel_path)

                            output_dir = (
                                os.path.join(self.output_folder, rel_dir)
                                if rel_dir
                                else self.output_folder
                            )
                            os.makedirs(output_dir, exist_ok=True)

                            if self.preserve_names:
                                base_name = os.path.splitext(
                                    os.path.basename(input_file)
                                )[0]
                            else:
                                base_name = f"processed_{i+1:04d}"

                        # Save file
                        if self.output_format.lower() == "mp4":
                            output_file = os.path.join(
                                output_dir, f"{base_name}.mp4"
                            )
                            processor.save_video(
                                processed_data, output_file, fps=self.video_fps
                            )
                        else:
                            output_file = os.path.join(
                                output_dir, f"{base_name}.nrrd"
                            )
                            processor.save_nrrd(processed_data, output_file)

                        if self.save_next_to_original:
                            self.status_updated.emit(f"Saved: {output_file}")
                        else:
                            self.status_updated.emit(
                                f"Saved: {os.path.relpath(output_file, self.output_folder)}"
                            )
                    else:
                        self.status_updated.emit(
                            f"Failed to process: "
                            f"{os.path.relpath(input_file, self.input_folder)}"
                        )

                except Exception as e:
                    self.status_updated.emit(
                        f"Error processing "
                        f"{os.path.relpath(input_file, self.input_folder)}: "
                        f"{str(e)}"
                    )

                progress = int((i + 1) / len(files) * 100)
                self.progress_updated.emit(progress)

            if not self.should_stop:
                self.status_updated.emit("Batch processing completed!")
            else:
                self.status_updated.emit("Batch processing stopped by user.")

        except Exception as e:
            self.error_occurred.emit(f"Batch processing error: {str(e)}")
        finally:
            self.finished.emit()


class BatchImageProcessor:
    """Non-UI processor for batch operations"""

    def __init__(self, config):
        self.config = config
        self.image_data = None
        self.current_file = None

        # File parameters
        self.pixel_type = config.get("pixel_type", "8 bit unsigned")
        self.endianness = config.get("endianness", "Little endian")
        self.header_size = config.get("header_size", 0)
        self.footer_size = config.get("footer_size", 0)
        self.width = config.get("width", 424)
        self.height = config.get("height", 127)
        self.depth = config.get("depth", 317)
        self.row_stride = config.get("row_stride", 0)
        self.row_padding = config.get("row_padding", 0)
        self.slice_stride = config.get("slice_stride", 0)
        self.skip_slices = config.get("skip_slices", 0)

        # Spacing
        self.spacing_x = config.get("spacing_x", 1.0)
        self.spacing_y = config.get("spacing_y", 2.6)
        self.spacing_z = config.get("spacing_z", 1.0)

        # Orientation operations
        self.orientation_ops = list(config.get("orientation_ops", []))

        # Crop parameters
        self.crop_top = config.get("crop_top", 0)
        self.crop_bottom = config.get("crop_bottom", 0)
        self.crop_left = config.get("crop_left", 0)
        self.crop_right = config.get("crop_right", 0)

        # Corner positions (to be applied after orientation)
        self._pending_corner_positions = None
        if config.get("corner_positions") is not None:
            self._pending_corner_positions = np.array(
                config["corner_positions"]
            )

        self.corner_positions = None
        self.use_corner_symmetry = bool(
            config.get("use_corner_symmetry", True)
        )

        # Curve parameters
        self.curve_x_pos = config.get("curve_x_pos", 0) / 100.0
        self.curve_x_neg = config.get("curve_x_neg", 0) / 100.0
        self.curve_y_pos = config.get("curve_y_pos", 0) / 100.0
        self.curve_y_neg = config.get("curve_y_neg", 0) / 100.0
        self.curve_z_pos = config.get("curve_z_pos", 0) / 100.0
        self.curve_z_neg = config.get("curve_z_neg", 0) / 100.0

    def set_input_file(self, file_path):
        self.current_file = file_path

    def get_pixel_info(self):
        """Get pixel type information"""
        type_map = {
            "8 bit unsigned": (np.uint8, 1, 1),
            "8 bit signed": (np.int8, 1, 1),
            "16 bit unsigned": (np.uint16, 2, 1),
            "16 bit signed": (np.int16, 2, 1),
            "float": (np.float32, 4, 1),
            "double": (np.float64, 8, 1),
            "24 bit RGB": (np.uint8, 1, 3),
        }

        dtype, byte_size, components = type_map[self.pixel_type]

        if self.endianness == "Big endian" and byte_size > 1:
            dtype_map = {
                np.uint16: ">u2",
                np.int16: ">i2",
                np.float32: ">f4",
                np.float64: ">f8",
            }
            dtype = dtype_map.get(dtype, dtype)

        return dtype, byte_size, components

    def apply_orientation_ops(self):
        """Apply all recorded orientation operations - matches test.py logic"""
        if not self.orientation_ops or self.image_data is None:
            return

        for op in self.orientation_ops:
            if op[0] == "flip":
                _, axis = op
                axis_map = {"z": 0, "y": 1, "x": 2}
                self.image_data = np.flip(self.image_data, axis=axis_map[axis])

            elif op[0] == "rotate":
                _, axis, direction = op
                k = 1 if direction > 0 else 3
                if axis == "z":
                    self.image_data = np.rot90(
                        self.image_data, k=k, axes=(1, 2)
                    )
                elif axis == "x":
                    self.image_data = np.rot90(
                        self.image_data, k=k, axes=(0, 1)
                    )
                elif axis == "y":
                    self.image_data = np.rot90(
                        self.image_data, k=k, axes=(0, 2)
                    )

    def reset_corners(self):
        """Reset corners to identity positions for current volume dimensions"""
        if self.image_data is not None:
            d, h, w = self.image_data.shape[:3]
        else:
            w = self.width
            h = self.height
            d = self.depth

        self.corner_positions = np.zeros((8, 3), dtype=np.float64)
        for idx in range(8):
            ix, iy, iz = (idx >> 0) & 1, (idx >> 1) & 1, (idx >> 2) & 1
            self.corner_positions[idx] = [
                ix * (w - 1),
                iy * (h - 1),
                iz * (d - 1),
            ]

    def apply_curve_deformation(self, X, Y, Z, D, H, W):
        """Apply curve bending to coordinates"""
        if not any(
            [
                abs(self.curve_x_pos) > 1e-6,
                abs(self.curve_x_neg) > 1e-6,
                abs(self.curve_y_pos) > 1e-6,
                abs(self.curve_y_neg) > 1e-6,
                abs(self.curve_z_pos) > 1e-6,
                abs(self.curve_z_neg) > 1e-6,
            ]
        ):
            return X, Y, Z

        x_norm = X / (W - 1) if W > 1 else np.zeros_like(X)
        y_norm = Y / (H - 1) if H > 1 else np.zeros_like(Y)
        z_norm = Z / (D - 1) if D > 1 else np.zeros_like(Z)

        # X curves
        if abs(self.curve_x_pos) > 1e-6 or abs(self.curve_x_neg) > 1e-6:
            curve_x = np.where(
                x_norm >= 0.5,
                self.curve_x_pos * (x_norm - 0.5) * 2.0,
                self.curve_x_neg * (0.5 - x_norm) * 2.0,
            )
            Y += curve_x * (H - 1) * 0.5 * np.sin(np.pi * y_norm)
            Z += curve_x * (D - 1) * 0.5 * np.sin(np.pi * z_norm)

        # Y curves
        if abs(self.curve_y_pos) > 1e-6 or abs(self.curve_y_neg) > 1e-6:
            curve_y = np.where(
                y_norm >= 0.5,
                self.curve_y_pos * (y_norm - 0.5) * 2.0,
                self.curve_y_neg * (0.5 - y_norm) * 2.0,
            )
            X += curve_y * (W - 1) * 0.5 * np.sin(np.pi * x_norm)
            Z += curve_y * (D - 1) * 0.5 * np.sin(np.pi * z_norm)

        # Z curves
        if abs(self.curve_z_pos) > 1e-6 or abs(self.curve_z_neg) > 1e-6:
            curve_z = np.where(
                z_norm >= 0.5,
                self.curve_z_pos * (z_norm - 0.5) * 2.0,
                self.curve_z_neg * (0.5 - z_norm) * 2.0,
            )
            X += curve_z * (W - 1) * 0.5 * np.sin(np.pi * x_norm)
            Y += curve_z * (H - 1) * 0.5 * np.sin(np.pi * y_norm)

        return X, Y, Z

    def warp_slice(self, z_idx):
        """Warp slice using 3D trilinear corner mapping + curves"""
        try:
            from scipy import ndimage
        except ImportError:
            ndimage = None

        vol = self.image_data
        D, H, W = vol.shape[:3]
        C = 1 if vol.ndim == 3 else vol.shape[3]

        xs = np.linspace(0.0, 1.0, W) if W > 1 else np.zeros(W)
        ys = np.linspace(0.0, 1.0, H) if H > 1 else np.zeros(H)
        wval = 0.0 if D <= 1 else z_idx / (D - 1)

        uu, vv = np.meshgrid(xs, ys)
        s0, s1 = 1.0 - uu, uu
        t0, t1 = 1.0 - vv, vv
        p0, p1 = 1.0 - wval, wval

        cp = self.corner_positions
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

        # Apply curves
        X, Y, Z = self.apply_curve_deformation(X, Y, Z, D, H, W)

        # Resample
        if ndimage is None:
            xi = np.clip(np.rint(X), 0, W - 1).astype(int)
            yi = np.clip(np.rint(Y), 0, H - 1).astype(int)
            zi = np.clip(np.rint(Z), 0, D - 1).astype(int)
            if C == 1:
                return vol[zi, yi, xi]
            else:
                out = np.zeros((H, W, C), dtype=vol.dtype)
                for c in range(C):
                    out[..., c] = vol[zi, yi, xi, c]
                return out
        else:
            if C == 1:
                return ndimage.map_coordinates(
                    vol, [Z, Y, X], order=1, mode="constant", cval=0.0
                )
            else:
                out = np.zeros((H, W, C), dtype=np.float64)
                for c in range(C):
                    out[..., c] = ndimage.map_coordinates(
                        vol[..., c],
                        [Z, Y, X],
                        order=1,
                        mode="constant",
                        cval=0.0,
                    )
                if np.issubdtype(vol.dtype, np.integer):
                    info = np.iinfo(vol.dtype)
                    out = np.clip(out, info.min, info.max).astype(vol.dtype)
                return out.astype(vol.dtype)

    def apply_crop_to_slice(self, slice_data):
        """Apply crop to a single slice"""
        if not (
            self.crop_top
            or self.crop_bottom
            or self.crop_left
            or self.crop_right
        ):
            return slice_data

        H, W = slice_data.shape[:2]

        if self.crop_top + self.crop_bottom >= H:
            return slice_data
        if self.crop_left + self.crop_right >= W:
            return slice_data

        if slice_data.ndim == 2:
            return slice_data[
                self.crop_top : H - self.crop_bottom,
                self.crop_left : W - self.crop_right,
            ]
        else:
            return slice_data[
                self.crop_top : H - self.crop_bottom,
                self.crop_left : W - self.crop_right,
                :,
            ]

    def are_corners_identity(self):
        """Check if corners are in default positions"""
        if self.image_data is not None:
            d, h, w = self.image_data.shape[:3]
        else:
            w = self.width
            h = self.height
            d = self.depth

        default_corners = np.zeros((8, 3), dtype=np.float64)
        for idx in range(8):
            ix, iy, iz = (idx >> 0) & 1, (idx >> 1) & 1, (idx >> 2) & 1
            default_corners[idx] = [ix * (w - 1), iy * (h - 1), iz * (d - 1)]

        return np.array_equal(self.corner_positions, default_corners)

    def process_file(self):
        """Process a single file using the configuration"""
        if not os.path.exists(self.current_file):
            return None

        try:
            dtype, byte_size, components = self.get_pixel_info()

            row_data_size = self.width * byte_size * components
            if self.row_stride > 0:
                effective_row_stride = self.row_stride
            else:
                effective_row_stride = row_data_size + self.row_padding

            slice_data_size = self.height * effective_row_stride
            effective_slice_stride = slice_data_size + self.slice_stride

            total_header_size = (
                self.header_size + self.skip_slices * effective_slice_stride
            )

            file_size = os.path.getsize(self.current_file)
            available_data_size = (
                file_size - total_header_size - self.footer_size
            )

            if available_data_size <= 0:
                return None

            max_slices = int(
                (available_data_size + self.slice_stride)
                / effective_slice_stride
            )
            final_depth = min(self.depth, max_slices)

            if final_depth <= 0:
                return None

            # Load image data
            image_slices = []
            with open(self.current_file, "rb") as f:
                for slice_idx in range(final_depth):
                    slice_position = (
                        total_header_size + slice_idx * effective_slice_stride
                    )

                    if effective_row_stride == row_data_size:
                        f.seek(slice_position)
                        slice_bytes = f.read(self.height * row_data_size)
                        if len(slice_bytes) < self.height * row_data_size:
                            break
                        slice_data = np.frombuffer(slice_bytes, dtype=dtype)
                    else:
                        row_data_list = []
                        for row_idx in range(self.height):
                            row_position = (
                                slice_position + row_idx * effective_row_stride
                            )
                            f.seek(row_position)
                            row_bytes = f.read(row_data_size)
                            if len(row_bytes) < row_data_size:
                                break
                            row_data = np.frombuffer(row_bytes, dtype=dtype)
                            row_data_list.append(row_data)
                        if len(row_data_list) != self.height:
                            break
                        slice_data = np.concatenate(row_data_list)

                    if components == 1:
                        slice_data = slice_data.reshape(
                            (self.height, self.width)
                        )
                    else:
                        slice_data = slice_data.reshape(
                            (self.height, self.width, components)
                        )

                    image_slices.append(slice_data)

            if not image_slices:
                return None

            self.image_data = np.array(image_slices)

            # Apply orientation operations
            self.apply_orientation_ops()

            # Handle pending corner positions from config
            # Apply AFTER orientation, just like test.py
            if self._pending_corner_positions is not None:
                self.corner_positions = self._pending_corner_positions
            else:
                self.reset_corners()

            # Build export volume with warping and cropping
            needs_warp = not self.are_corners_identity()
            needs_crop = (
                self.crop_top
                or self.crop_bottom
                or self.crop_left
                or self.crop_right
            )

            if needs_warp or needs_crop:
                D = self.image_data.shape[0]
                slices = []
                for z in range(D):
                    slice_data = self.warp_slice(z)
                    slice_data = self.apply_crop_to_slice(slice_data)
                    slices.append(slice_data)
                self.image_data = np.stack(slices, axis=0)

            return self.image_data

        except Exception as e:
            print(f"Error processing {self.current_file}: {str(e)}")
            return None

    def save_nrrd(self, data, output_path):
        """Save data as NRRD file"""
        try:
            if data.ndim == 4:
                depth, height, width, components = data.shape
            else:
                depth, height, width = data.shape
                components = 1

            type_map = {
                "8 bit unsigned": "uchar",
                "8 bit signed": "signed char",
                "16 bit unsigned": "ushort",
                "16 bit signed": "short",
                "float": "float",
                "double": "double",
                "24 bit RGB": "uchar",
            }

            with open(output_path, "w") as f:
                f.write("NRRD0004\n")
                f.write("# Complete NRRD file format specification at:\n")
                f.write("# http://teem.sourceforge.net/nrrd/format.html\n")
                f.write(f"type: {type_map[self.pixel_type]}\n")
                f.write("space: left-posterior-superior\n")

                if components > 1:
                    f.write("dimension: 4\n")
                    f.write(f"sizes: {components} {width} {height} {depth}\n")
                    f.write(
                        f"space directions: none ({self.spacing_x},0,0) "
                        f"(0,{self.spacing_y},0) (0,0,{self.spacing_z})\n"
                    )
                    f.write("kinds: vector domain domain domain\n")
                else:
                    f.write("dimension: 3\n")
                    f.write(f"sizes: {width} {height} {depth}\n")
                    f.write(
                        f"space directions: ({self.spacing_x},0,0) "
                        f"(0,{self.spacing_y},0) (0,0,{self.spacing_z})\n"
                    )
                    f.write("kinds: domain domain domain\n")

                f.write(
                    f"endian: {'little' if self.endianness == 'Little endian' else 'big'}\n"
                )
                f.write("encoding: raw\n")
                f.write("space origin: (0,0,0)\n")
                f.write("\n")

            # Append binary data
            with open(output_path, "ab") as f:
                base_dtype_map = {
                    "8 bit unsigned": np.uint8,
                    "8 bit signed": np.int8,
                    "16 bit unsigned": np.uint16,
                    "16 bit signed": np.int16,
                    "float": np.float32,
                    "double": np.float64,
                    "24 bit RGB": np.uint8,
                }
                base_dtype = base_dtype_map[self.pixel_type]

                data_to_save = data.astype(base_dtype, copy=False)
                if (
                    self.endianness == "Big endian"
                    and data_to_save.dtype.itemsize > 1
                ):
                    data_to_save = data_to_save.byteswap().newbyteorder()

                if components > 1:
                    data_to_save = np.moveaxis(data_to_save, -1, 0)

                f.write(data_to_save.tobytes())

        except Exception as e:
            raise Exception(f"Failed to save NRRD file: {str(e)}")

    def _normalize_to_uint8(self, arr: np.ndarray) -> np.ndarray:
        """Normalize any numeric array to uint8 [0, 255]"""
        arr = np.asarray(arr)
        if arr.size == 0:
            return np.zeros(arr.shape, dtype=np.uint8)
        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            amin, amax = float(info.min), float(info.max)
        else:
            amin = float(np.nanmin(arr))
            amax = float(np.nanmax(arr))
        if not np.isfinite(amin) or not np.isfinite(amax) or amax <= amin:
            return np.zeros(arr.shape, dtype=np.uint8)
        scaled = (arr.astype(np.float32) - amin) / (amax - amin)
        scaled = np.clip(scaled, 0.0, 1.0)
        return (scaled * 255.0 + 0.5).astype(np.uint8)

    def save_video(self, data, output_path, fps=10):
        """Save depth stack as an MP4 video"""
        try:
            import imageio.v2 as imageio
            from scipy import ndimage
        except ImportError as e:
            if "imageio" in str(e):
                raise Exception(
                    "Saving MP4 requires imageio with ffmpeg. "
                    'Install with: pip install "imageio[ffmpeg]"'
                )
            else:
                ndimage = None

        if data.ndim == 4:
            depth, height, width, components = data.shape
        else:
            depth, height, width = data.shape
            components = 1

        # Calculate scaling for correct aspect ratio
        scale_x = self.spacing_x if self.spacing_x > 0 else 1.0
        scale_y = self.spacing_y if self.spacing_y > 0 else 1.0

        min_scale = min(scale_x, scale_y)
        if min_scale > 0:
            scale_x /= min_scale
            scale_y /= min_scale

        target_width = max(1, int(round(width * scale_x)))
        target_height = max(1, int(round(height * scale_y)))

        # Ensure even dimensions for H.264
        target_width += target_width % 2
        target_height += target_height % 2

        # Initialize writer
        try:
            writer = imageio.get_writer(
                output_path, fps=max(1, int(fps)), codec="libx264", quality=8
            )
        except TypeError:
            writer = imageio.get_writer(output_path, fps=max(1, int(fps)))

        with writer:
            for z in range(depth):
                frame = data[z]
                if components == 1:
                    frame_u8 = self._normalize_to_uint8(frame)

                    if ndimage is not None and (
                        target_width != width or target_height != height
                    ):
                        frame_u8 = ndimage.zoom(
                            frame_u8,
                            (target_height / height, target_width / width),
                            order=1,
                            mode="nearest",
                        )
                    elif target_width != width or target_height != height:
                        y_indices = np.linspace(
                            0, height - 1, target_height
                        ).astype(int)
                        x_indices = np.linspace(
                            0, width - 1, target_width
                        ).astype(int)
                        frame_u8 = frame_u8[np.ix_(y_indices, x_indices)]
                else:
                    if frame.dtype != np.uint8:
                        frame_u8 = self._normalize_to_uint8(frame)
                    else:
                        frame_u8 = frame
                    if frame_u8.ndim == 2:
                        frame_u8 = np.stack([frame_u8] * 3, axis=-1)

                    if ndimage is not None and (
                        target_width != width or target_height != height
                    ):
                        frame_u8 = ndimage.zoom(
                            frame_u8,
                            (target_height / height, target_width / width, 1),
                            order=1,
                            mode="nearest",
                        )
                    elif target_width != width or target_height != height:
                        y_indices = np.linspace(
                            0, height - 1, target_height
                        ).astype(int)
                        x_indices = np.linspace(
                            0, width - 1, target_width
                        ).astype(int)
                        frame_u8 = frame_u8[np.ix_(y_indices, x_indices)]

                writer.append_data(frame_u8)


class BatchProcessorGUI(QMainWindow):
    """GUI for batch processing raw image files"""

    def __init__(self):
        super().__init__()
        self.processor_thread = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("3D Ultrasound Scrapper - Batch Processor")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Input settings group
        input_group = QGroupBox("Input Settings")
        input_layout = QGridLayout(input_group)

        input_layout.addWidget(QLabel("Input Folder:"), 0, 0)
        self.input_folder_edit = QLineEdit()
        input_layout.addWidget(self.input_folder_edit, 0, 1)
        browse_input_btn = QPushButton("Browse...")
        browse_input_btn.clicked.connect(self.browse_input_folder)
        input_layout.addWidget(browse_input_btn, 0, 2)

        input_layout.addWidget(QLabel("File Pattern:"), 1, 0)
        self.file_pattern_edit = QLineEdit("*.raw")
        input_layout.addWidget(self.file_pattern_edit, 1, 1, 1, 2)

        # File size filtering
        input_layout.addWidget(QLabel("Min File Size:"), 2, 0)
        min_size_layout = QHBoxLayout()
        self.min_size_edit = QLineEdit("0")
        self.min_size_edit.setMaximumWidth(100)
        min_size_layout.addWidget(self.min_size_edit)
        self.min_size_unit = QComboBox()
        self.min_size_unit.addItems(["B", "KB", "MB", "GB"])
        self.min_size_unit.setMaximumWidth(60)
        min_size_layout.addWidget(self.min_size_unit)
        min_size_layout.addStretch()
        min_size_widget = QWidget()
        min_size_widget.setLayout(min_size_layout)
        input_layout.addWidget(min_size_widget, 2, 1, 1, 2)

        input_layout.addWidget(QLabel("Max File Size:"), 3, 0)
        max_size_layout = QHBoxLayout()
        self.max_size_edit = QLineEdit("")
        self.max_size_edit.setPlaceholderText("No limit")
        self.max_size_edit.setMaximumWidth(100)
        max_size_layout.addWidget(self.max_size_edit)
        self.max_size_unit = QComboBox()
        self.max_size_unit.addItems(["B", "KB", "MB", "GB"])
        self.max_size_unit.setCurrentText("MB")
        self.max_size_unit.setMaximumWidth(60)
        max_size_layout.addWidget(self.max_size_unit)
        max_size_layout.addStretch()
        max_size_widget = QWidget()
        max_size_widget.setLayout(max_size_layout)
        input_layout.addWidget(max_size_widget, 3, 1, 1, 2)

        input_layout.addWidget(QLabel("Config File:"), 4, 0)
        self.config_file_edit = QLineEdit()
        input_layout.addWidget(self.config_file_edit, 4, 1)
        browse_config_btn = QPushButton("Browse...")
        browse_config_btn.clicked.connect(self.browse_config_file)
        input_layout.addWidget(browse_config_btn, 4, 2)

        layout.addWidget(input_group)

        # Output settings group
        output_group = QGroupBox("Output Settings")
        output_layout = QGridLayout(output_group)

        # Save next to original checkbox
        self.save_next_to_original_checkbox = QCheckBox(
            "Save next to original file"
        )
        self.save_next_to_original_checkbox.stateChanged.connect(
            self.on_save_next_to_original_changed
        )
        output_layout.addWidget(
            self.save_next_to_original_checkbox, 0, 0, 1, 3
        )

        output_layout.addWidget(QLabel("Output Folder:"), 1, 0)
        self.output_folder_edit = QLineEdit()
        output_layout.addWidget(self.output_folder_edit, 1, 1)
        self.browse_output_btn = QPushButton("Browse...")
        self.browse_output_btn.clicked.connect(self.browse_output_folder)
        output_layout.addWidget(self.browse_output_btn, 1, 2)

        self.preserve_names_checkbox = QCheckBox("Preserve original filenames")
        self.preserve_names_checkbox.setChecked(True)
        output_layout.addWidget(self.preserve_names_checkbox, 2, 0, 1, 3)

        output_layout.addWidget(QLabel("Output Format:"), 3, 0)
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["NRRD (.nrrd)", "MP4 Video (.mp4)"])
        output_layout.addWidget(self.output_format_combo, 3, 1, 1, 2)

        output_layout.addWidget(QLabel("Video FPS:"), 4, 0)
        self.fps_edit = QLineEdit("10")
        self.fps_edit.setMaximumWidth(100)
        output_layout.addWidget(self.fps_edit, 4, 1)

        layout.addWidget(output_group)

        # Control buttons
        control_layout = QHBoxLayout()

        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)

        control_layout.addStretch()
        layout.addLayout(control_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Status/log area
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        layout.addWidget(QLabel("Processing Log:"))
        layout.addWidget(self.log_text)

    def on_save_next_to_original_changed(self, state):
        """Enable/disable output folder controls based on checkbox state"""
        is_checked = state == Qt.Checked
        self.output_folder_edit.setEnabled(not is_checked)
        self.browse_output_btn.setEnabled(not is_checked)
        self.preserve_names_checkbox.setEnabled(not is_checked)

    def browse_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.input_folder_edit.setText(folder)

    def browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder_edit.setText(folder)

    def browse_config_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Configuration File",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if file_path:
            self.config_file_edit.setText(file_path)

    def log_message(self, message):
        self.log_text.append(message)

    def convert_size_to_bytes(self, size_str, unit):
        """Convert size string and unit to bytes"""
        if not size_str.strip():
            return None

        try:
            size = float(size_str)
            multipliers = {
                "B": 1,
                "KB": 1024,
                "MB": 1024 * 1024,
                "GB": 1024 * 1024 * 1024,
            }
            return int(size * multipliers.get(unit, 1))
        except ValueError:
            return None

    def get_file_size_constraints(self):
        """Get min and max file size in bytes from GUI inputs"""
        min_size = self.convert_size_to_bytes(
            self.min_size_edit.text(), self.min_size_unit.currentText()
        )
        max_size = self.convert_size_to_bytes(
            self.max_size_edit.text(), self.max_size_unit.currentText()
        )

        if min_size is None:
            min_size = 0

        return min_size, max_size

    def start_processing(self):
        # Validate inputs
        input_folder = self.input_folder_edit.text().strip()
        file_pattern = self.file_pattern_edit.text().strip()
        config_file = self.config_file_edit.text().strip()
        save_next_to_original = self.save_next_to_original_checkbox.isChecked()

        if not input_folder or not os.path.exists(input_folder):
            QMessageBox.warning(
                self, "Warning", "Please select a valid input folder"
            )
            return

        if not file_pattern:
            QMessageBox.warning(
                self, "Warning", "Please specify a file pattern"
            )
            return

        if not config_file or not os.path.exists(config_file):
            QMessageBox.warning(
                self, "Warning", "Please select a valid configuration file"
            )
            return

        # Validate output folder only if not saving next to original
        if not save_next_to_original:
            output_folder = self.output_folder_edit.text().strip()
            if not output_folder:
                QMessageBox.warning(
                    self, "Warning", "Please specify an output folder"
                )
                return
        else:
            output_folder = ""  # Not used when saving next to original

        # Get file size constraints
        min_file_size, max_file_size = self.get_file_size_constraints()

        # Output format and fps
        output_format = (
            "mp4" if self.output_format_combo.currentIndex() == 1 else "nrrd"
        )
        try:
            video_fps = int(self.fps_edit.text().strip())
        except ValueError:
            video_fps = 10
        if video_fps <= 0:
            video_fps = 10

        # Start processing thread
        self.processor_thread = BatchProcessor(
            input_folder=input_folder,
            file_pattern=file_pattern,
            config_file=config_file,
            output_folder=output_folder,
            preserve_names=self.preserve_names_checkbox.isChecked(),
            min_file_size=min_file_size,
            max_file_size=max_file_size,
            output_format=output_format,
            video_fps=video_fps,
            save_next_to_original=save_next_to_original,
        )

        # Connect signals
        self.processor_thread.progress_updated.connect(
            self.progress_bar.setValue
        )
        self.processor_thread.status_updated.connect(self.log_message)
        self.processor_thread.finished.connect(self.processing_finished)
        self.processor_thread.error_occurred.connect(self.processing_error)

        # Update UI state
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()

        # Start processing
        self.processor_thread.start()
        self.log_message("Starting batch processing...")

    def stop_processing(self):
        if self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.stop()
            self.log_message("Stopping processing...")

    def processing_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log_message("Processing finished.")

    def processing_error(self, error_message):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log_message(f"Error: {error_message}")
        QMessageBox.critical(self, "Processing Error", error_message)


def main():
    app = QApplication(sys.argv)
    window = BatchProcessorGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
