#!/usr/bin/env python3
"""
3D Ultrasound Scrapper — Batch Processor
Processes multiple raw image files using a configuration template to generate NRRD files.
"""

import os
import sys
import json
import glob
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

# Import the processing functions from the main script
# We'll import the core processing logic without the UI
try:
	from test import RawImageGuessQt
except ImportError:
	print("Error: Cannot import test.py. Make sure test.py is in the same directory.")
	sys.exit(1)


class BatchProcessor(QThread):
	"""Worker thread for batch processing files"""
	progress_updated = pyqtSignal(int)
	status_updated = pyqtSignal(str)
	finished = pyqtSignal()
	error_occurred = pyqtSignal(str)

	def __init__(self, input_folder, file_pattern, config_file, output_folder, preserve_names=True, min_file_size=0, max_file_size=None, output_format="nrrd", video_fps=10):
		super().__init__()
		self.input_folder = input_folder
		self.file_pattern = file_pattern
		self.config_file = config_file
		self.output_folder = output_folder
		self.preserve_names = preserve_names
		self.min_file_size = min_file_size  # in bytes
		self.max_file_size = max_file_size  # in bytes, None means no limit
		self.output_format = output_format  # 'nrrd' or 'mp4'
		self.video_fps = int(video_fps) if str(video_fps).strip().isdigit() else 10
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
			with open(self.config_file, 'r') as f:
				config = json.load(f)

			# Find all matching files
			search_pattern = os.path.join(self.input_folder, "**", self.file_pattern)
			files = glob.glob(search_pattern, recursive=True)

			if not files:
				self.error_occurred.emit(f"No files found matching pattern: {self.file_pattern}")
				return

			# Apply file size filtering
			filtered_files = []
			for file_path in files:
				try:
					file_size = os.path.getsize(file_path)

					# Check minimum file size
					if file_size < self.min_file_size:
						continue

					# Check maximum file size (if specified)
					if self.max_file_size is not None and file_size > self.max_file_size:
						continue

					filtered_files.append(file_path)
				except OSError:
					# Skip files that can't be accessed
					continue

			files = filtered_files

			if not files:
				size_info = f"min: {self.format_file_size(self.min_file_size)}"
				if self.max_file_size is not None:
					size_info += f", max: {self.format_file_size(self.max_file_size)}"
				else:
					size_info += f", max: {self.format_file_size(None)}"
				self.error_occurred.emit(f"No files found matching pattern '{self.file_pattern}' with size constraints ({size_info})")
				return

			self.status_updated.emit(f"Found {len(files)} files to process")

			# Create output folder if it doesn't exist
			os.makedirs(self.output_folder, exist_ok=True)

			# Process each file
			for i, input_file in enumerate(files):
				if self.should_stop:
					break

				self.status_updated.emit(f"Processing: {os.path.relpath(input_file, self.input_folder)}")

				try:
					# Create a temporary processor instance for this file
					processor = BatchImageProcessor(config)

					# Set the input file
					processor.set_input_file(input_file)

					# Process the file
					processed_data = processor.process_file()

					if processed_data is not None:
						# Calculate relative path from input folder
						rel_path = os.path.relpath(input_file, self.input_folder)
						rel_dir = os.path.dirname(rel_path)

						# Create corresponding directory structure in output folder
						output_dir = os.path.join(self.output_folder, rel_dir) if rel_dir else self.output_folder
						os.makedirs(output_dir, exist_ok=True)

						# Generate output filename
						if self.preserve_names:
							base_name = os.path.splitext(os.path.basename(input_file))[0]
						else:
							base_name = f"processed_{i+1:04d}"

						# Save in selected format
						if self.output_format.lower() == "mp4":
							output_file = os.path.join(output_dir, f"{base_name}.mp4")
							processor.save_video(processed_data, output_file, fps=self.video_fps)
						else:
							output_file = os.path.join(output_dir, f"{base_name}.nrrd")
							processor.save_nrrd(processed_data, output_file)

						self.status_updated.emit(f"Saved: {os.path.relpath(output_file, self.output_folder)}")
					else:
						self.status_updated.emit(f"Failed to process: {os.path.relpath(input_file, self.input_folder)}")

				except Exception as e:
					self.status_updated.emit(f"Error processing {os.path.relpath(input_file, self.input_folder)}: {str(e)}")

				# Update progress
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

		# Apply configuration parameters
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

		# Orientation operations (persisted flips/rotations from UI)
		self.orientation_ops = list(config.get("orientation_ops", []))

		# Scale parameters
		self.scale_x = float(config.get("scale_x", 1.0))
		self.scale_y = float(config.get("scale_y", 1.0))
		self.scale_z = float(config.get("scale_z", 1.0))
		self.preserve_size = bool(config.get("preserve_size", True))

		# Corner positions for warping
		self.corner_positions = None
		if config.get("corner_positions") is not None:
			self.corner_positions = np.array(config["corner_positions"])
		self.use_corner_symmetry = bool(config.get("use_corner_symmetry", True))

		# Curve parameters
		self.curve_x_pos = config.get("curve_x_pos", 0) / 100.0
		self.curve_x_neg = config.get("curve_x_neg", 0) / 100.0
		self.curve_y_pos = config.get("curve_y_pos", 0) / 100.0
		self.curve_y_neg = config.get("curve_y_neg", 0) / 100.0
		self.curve_z_pos = config.get("curve_z_pos", 0) / 100.0
		self.curve_z_neg = config.get("curve_z_neg", 0) / 100.0

	def set_input_file(self, file_path):
		self.current_file = file_path

	def _swap_spacing(self, a, b):
		if a == "x" and b == "y":
			self.spacing_x, self.spacing_y = self.spacing_y, self.spacing_x
		elif a == "y" and b == "z":
			self.spacing_y, self.spacing_z = self.spacing_z, self.spacing_y
		elif a == "x" and b == "z":
			self.spacing_x, self.spacing_z = self.spacing_z, self.spacing_x
		elif a == "y" and b == "x":
			self.spacing_y, self.spacing_x = self.spacing_x, self.spacing_y
		elif a == "z" and b == "y":
			self.spacing_z, self.spacing_y = self.spacing_y, self.spacing_z
		elif a == "z" and b == "x":
			self.spacing_z, self.spacing_x = self.spacing_x, self.spacing_z

	def apply_saved_orientation(self):
		"""Apply recorded flips/rotations to current volume and update spacings."""
		if self.image_data is None or not self.orientation_ops:
			return

		for op in self.orientation_ops:
			if not op:
				continue
			if op[0] == "flip":
				_, axis = op
				axis_map = {"z": 0, "y": 1, "x": 2}
				if axis not in axis_map:
					continue
				self.image_data = np.flip(self.image_data, axis=axis_map[axis])
			elif op[0] == "rotate":
				_, axis, direction = op
				k = 1 if int(direction) > 0 else 3
				if axis == "z":
					self.image_data = np.rot90(self.image_data, k=k, axes=(1, 2))
				elif axis == "x":
					self.image_data = np.rot90(self.image_data, k=k, axes=(0, 1))
				elif axis == "y":
					self.image_data = np.rot90(self.image_data, k=k, axes=(0, 2))

	def get_pixel_info(self):
		"""Get pixel type information - copied from main script"""
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
			if dtype == np.uint16:
				dtype = ">u2"
			elif dtype == np.int16:
				dtype = ">i2"
			elif dtype == np.float32:
				dtype = ">f4"
			elif dtype == np.float64:
				dtype = ">f8"

		return dtype, byte_size, components

	def _apply_stretch_if_needed(self):
		sx, sy, sz = self.scale_x, self.scale_y, self.scale_z
		if (abs(sx - 1.0) < 1e-9 and abs(sy - 1.0) < 1e-9 and abs(sz - 1.0) < 1e-9):
			return
		try:
			from scipy import ndimage
		except ImportError:
			ndimage = None

		arr = self.image_data
		if arr is None:
			return

		if arr.ndim == 3:
			zoom_factors = (sz, sy, sx)
		else:
			zoom_factors = (sz, sy, sx, 1.0)

		if ndimage is None:
			def nn_scale_axis(a, axis, factor):
				if abs(factor - 1.0) < 1e-9:
					return a
				n_old = a.shape[axis]
				n_new = max(1, int(round(n_old * factor)))
				idx = np.linspace(0, n_old - 1, n_new)
				idx = np.clip(np.rint(idx), 0, n_old - 1).astype(int)
				return np.take(a, idx, axis=axis)

			out = arr
			out = nn_scale_axis(out, 2, sx)
			out = nn_scale_axis(out, 1, sy)
			out = nn_scale_axis(out, 0, sz)
		else:
			out = ndimage.zoom(arr, zoom=zoom_factors, order=1, mode="nearest")

		self.image_data = out

		if self.preserve_size:
			# Adjust spacings inversely to maintain physical size
			if sx != 0:
				self.spacing_x = self.spacing_x / sx
			if sy != 0:
				self.spacing_y = self.spacing_y / sy
			if sz != 0:
				self.spacing_z = self.spacing_z / sz

	def are_curves_active(self):
		"""Check if any curve deformations are active"""
		return (abs(self.curve_x_pos) > 1e-6 or abs(self.curve_x_neg) > 1e-6 or
				abs(self.curve_y_pos) > 1e-6 or abs(self.curve_y_neg) > 1e-6 or
				abs(self.curve_z_pos) > 1e-6 or abs(self.curve_z_neg) > 1e-6)

	def ensure_corner_positions_for_current_dims(self):
		"""Ensure corner_positions is defined for current image dims (identity if missing)."""
		if self.image_data is None:
			return
		if self.corner_positions is not None and self.corner_positions.shape == (8, 3):
			return
		# Build identity corners for current dims
		if self.image_data.ndim == 4:
			D, H, W, _ = self.image_data.shape
		else:
			D, H, W = self.image_data.shape
		cps = np.zeros((8, 3), dtype=np.float64)
		for idx in range(8):
			ix = (idx >> 0) & 1
			iy = (idx >> 1) & 1
			iz = (idx >> 2) & 1
			cps[idx, 0] = ix * (W - 1)
			cps[idx, 1] = iy * (H - 1)
			cps[idx, 2] = iz * (D - 1)
		self.corner_positions = cps

	def apply_curve_transformation(self, X, Y, Z, D, H, W):
		"""Apply curve deformations to coordinate arrays - copied from main script"""
		if not self.are_curves_active():
			return X, Y, Z

		# Normalize coordinates to [0,1] range for curve calculations
		x_norm = X / (W - 1) if W > 1 else np.zeros_like(X)
		y_norm = Y / (H - 1) if H > 1 else np.zeros_like(Y)
		z_norm = Z / (D - 1) if D > 1 else np.zeros_like(Z)

		# Apply X-axis curves (bend along YZ plane)
		if abs(self.curve_x_pos) > 1e-6 or abs(self.curve_x_neg) > 1e-6:
			curve_factor_x = np.zeros_like(x_norm)

			if abs(self.curve_x_pos) > 1e-6:
				mask_pos = x_norm >= 0.5
				intensity = (x_norm - 0.5) * 2.0
				curve_factor_x[mask_pos] += self.curve_x_pos * intensity[mask_pos]

			if abs(self.curve_x_neg) > 1e-6:
				mask_neg = x_norm <= 0.5
				intensity = (0.5 - x_norm) * 2.0
				curve_factor_x[mask_neg] += self.curve_x_neg * intensity[mask_neg]

			Y += curve_factor_x * (H - 1) * 0.5 * np.sin(np.pi * y_norm)
			Z += curve_factor_x * (D - 1) * 0.5 * np.sin(np.pi * z_norm)

		# Apply Y-axis curves (bend along XZ plane)
		if abs(self.curve_y_pos) > 1e-6 or abs(self.curve_y_neg) > 1e-6:
			curve_factor_y = np.zeros_like(y_norm)

			if abs(self.curve_y_pos) > 1e-6:
				mask_pos = y_norm >= 0.5
				intensity = (y_norm - 0.5) * 2.0
				curve_factor_y[mask_pos] += self.curve_y_pos * intensity[mask_pos]

			if abs(self.curve_y_neg) > 1e-6:
				mask_neg = y_norm <= 0.5
				intensity = (0.5 - y_norm) * 2.0
				curve_factor_y[mask_neg] += self.curve_y_neg * intensity[mask_neg]

			X += curve_factor_y * (W - 1) * 0.5 * np.sin(np.pi * x_norm)
			Z += curve_factor_y * (D - 1) * 0.5 * np.sin(np.pi * z_norm)

		# Apply Z-axis curves (bend along XY plane)
		if abs(self.curve_z_pos) > 1e-6 or abs(self.curve_z_neg) > 1e-6:
			curve_factor_z = np.zeros_like(z_norm)

			if abs(self.curve_z_pos) > 1e-6:
				mask_pos = z_norm >= 0.5
				intensity = (z_norm - 0.5) * 2.0
				curve_factor_z[mask_pos] += self.curve_z_pos * intensity[mask_pos]

			if abs(self.curve_z_neg) > 1e-6:
				mask_neg = z_norm <= 0.5
				intensity = (0.5 - z_norm) * 2.0
				curve_factor_z[mask_neg] += self.curve_z_neg * intensity[mask_neg]

			X += curve_factor_z * (W - 1) * 0.5 * np.sin(np.pi * x_norm)
			Y += curve_factor_z * (H - 1) * 0.5 * np.sin(np.pi * y_norm)

		return X, Y, Z

	def warp_slice_with_corners(self, z_idx):
		"""Warp slice using corner mapping - adapted from main script"""
		try:
			from scipy import ndimage
		except ImportError:
			ndimage = None

		vol = self.image_data
		if vol.ndim == 4:
			D, H, W, C = vol.shape
		else:
			D, H, W = vol.shape
			C = 1

		# Ensure we have corners (identity if not provided)
		self.ensure_corner_positions_for_current_dims()

		xs = np.linspace(0.0, 1.0, W) if W > 1 else np.zeros(W)
		ys = np.linspace(0.0, 1.0, H) if H > 1 else np.zeros(H)
		wval = 0.0 if D <= 1 else float(z_idx) / float(D - 1)

		uu, vv = np.meshgrid(xs, ys)
		s0 = 1.0 - uu
		s1 = uu
		t0 = 1.0 - vv
		t1 = vv
		p0 = 1.0 - wval
		p1 = wval

		cp = self.corner_positions
		c000, c100, c010, c110 = cp[0], cp[1], cp[2], cp[3]
		c001, c101, c011, c111 = cp[4], cp[5], cp[6], cp[7]

		X = (c000[0] * s0 * t0 * p0 + c100[0] * s1 * t0 * p0 +
			 c010[0] * s0 * t1 * p0 + c110[0] * s1 * t1 * p0 +
			 c001[0] * s0 * t0 * p1 + c101[0] * s1 * t0 * p1 +
			 c011[0] * s0 * t1 * p1 + c111[0] * s1 * t1 * p1)

		Y = (c000[1] * s0 * t0 * p0 + c100[1] * s1 * t0 * p0 +
			 c010[1] * s0 * t1 * p0 + c110[1] * s1 * t1 * p0 +
			 c001[1] * s0 * t0 * p1 + c101[1] * s1 * t0 * p1 +
			 c011[1] * s0 * t1 * p1 + c111[1] * s1 * t1 * p1)

		Z = (c000[2] * s0 * t0 * p0 + c100[2] * s1 * t0 * p0 +
			 c010[2] * s0 * t1 * p0 + c110[2] * s1 * t1 * p0 +
			 c001[2] * s0 * t0 * p1 + c101[2] * s1 * t0 * p1 +
			 c011[2] * s0 * t1 * p1 + c111[2] * s1 * t1 * p1)

		# Apply curve deformations
		X, Y, Z = self.apply_curve_transformation(X, Y, Z, D, H, W)

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
						vol[..., c], [Z, Y, X], order=1, mode="constant", cval=0.0
					)
				return out.astype(vol.dtype)

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

			total_header_size = self.header_size + self.skip_slices * effective_slice_stride

			file_size = os.path.getsize(self.current_file)
			available_data_size = file_size - total_header_size - self.footer_size

			if available_data_size <= 0:
				return None

			max_slices = int((available_data_size + self.slice_stride) / effective_slice_stride)
			final_depth = min(self.depth, max_slices)

			if final_depth <= 0:
				return None

			# Load image data
			image_slices = []
			with open(self.current_file, "rb") as f:
				for slice_idx in range(final_depth):
					slice_position = total_header_size + slice_idx * effective_slice_stride

					if effective_row_stride == row_data_size:
						f.seek(slice_position)
						slice_bytes = f.read(self.height * row_data_size)
						if len(slice_bytes) < self.height * row_data_size:
							break
						slice_data = np.frombuffer(slice_bytes, dtype=dtype)
					else:
						row_data_list = []
						for row_idx in range(self.height):
							row_position = slice_position + row_idx * effective_row_stride
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
						slice_data = slice_data.reshape((self.height, self.width))
					else:
						slice_data = slice_data.reshape((self.height, self.width, components))

					image_slices.append(slice_data)

			if not image_slices:
				return None

			self.image_data = np.array(image_slices)

			# Apply orientation from config (flips/rotations) and update spacings
			self.apply_saved_orientation()

			# Apply warping if corners provided, or curves are active
			if self.corner_positions is not None or self.are_curves_active():
				warped_slices = []
				for z in range(self.image_data.shape[0]):
					warped_slices.append(self.warp_slice_with_corners(z))
				self.image_data = np.stack(warped_slices, axis=0)

			# Apply stretch (resample) if requested
			self._apply_stretch_if_needed()

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
					f.write(f"space directions: none ({self.spacing_x},0,0) (0,{self.spacing_y},0) (0,0,{self.spacing_z})\n")
					f.write("kinds: vector domain domain domain\n")
				else:
					f.write("dimension: 3\n")
					f.write(f"sizes: {width} {height} {depth}\n")
					f.write(f"space directions: ({self.spacing_x},0,0) (0,{self.spacing_y},0) (0,0,{self.spacing_z})\n")
					f.write("kinds: domain domain domain\n")

				f.write(f"endian: {'little' if self.endianness == 'Little endian' else 'big'}\n")
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
				if self.endianness == "Big endian" and data_to_save.dtype.itemsize > 1:
					data_to_save = data_to_save.byteswap().newbyteorder()

				if components > 1:
					data_to_save = np.moveaxis(data_to_save, -1, 0)

				f.write(data_to_save.tobytes())

		except Exception as e:
			raise Exception(f"Failed to save NRRD file: {str(e)}")

	def _normalize_any_to_uint8(self, arr: np.ndarray) -> np.ndarray:
		"""Normalize any numeric array to uint8 [0, 255]."""
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
		"""Save depth stack as an MP4 video over slices."""
		try:
			import imageio.v2 as imageio
			from scipy import ndimage
		except ImportError as e:
			if "imageio" in str(e):
				raise Exception("Saving MP4 requires imageio with ffmpeg. Install with: pip install \"imageio[ffmpeg]\"")
			else:
				# scipy not available, will use basic nearest neighbor
				ndimage = None

		if data.ndim == 4:
			depth, height, width, components = data.shape
		else:
			depth, height, width = data.shape
			components = 1

				# Calculate scaling based on spacing to get correct aspect ratio
		# If spacing_y > spacing_x, pixels are taller than wide, so we need to compress height
		scale_factor_x = self.spacing_x if self.spacing_x > 0 else 1.0
		scale_factor_y = self.spacing_y if self.spacing_y > 0 else 1.0

		# Normalize scaling factors to maintain reasonable video size
		min_scale = min(scale_factor_x, scale_factor_y)
		if min_scale > 0:
			scale_factor_x /= min_scale
			scale_factor_y /= min_scale

		# Calculate target dimensions
		target_width = max(1, int(round(width * scale_factor_x)))
		target_height = max(1, int(round(height * scale_factor_y)))

		# Ensure even dimensions for H.264 compatibility
		target_width += target_width % 2
		target_height += target_height % 2

		# Initialize writer
		try:
			writer = imageio.get_writer(output_path, fps=max(1, int(fps)), codec="libx264", quality=8)
		except TypeError:
			writer = imageio.get_writer(output_path, fps=max(1, int(fps)))

		with writer:
			for z in range(depth):
				frame = data[z]
				if components == 1:
					frame_u8 = self._normalize_any_to_uint8(frame)

					# Apply scaling to correct aspect ratio
					if ndimage is not None and (target_width != width or target_height != height):
						frame_u8 = ndimage.zoom(frame_u8, (target_height/height, target_width/width), order=1, mode='nearest')
					elif target_width != width or target_height != height:
						# Fallback: simple nearest neighbor resize
						y_indices = np.linspace(0, height-1, target_height).astype(int)
						x_indices = np.linspace(0, width-1, target_width).astype(int)
						frame_u8 = frame_u8[np.ix_(y_indices, x_indices)]
				else:
					# Expecting 3 channels; normalize if not uint8
					if frame.dtype != np.uint8:
						frame_u8 = self._normalize_any_to_uint8(frame)
					else:
						frame_u8 = frame
					# Ensure shape (H, W, 3)
					if frame_u8.ndim == 2:
						frame_u8 = np.stack([frame_u8] * 3, axis=-1)

					# Apply scaling to correct aspect ratio
					if ndimage is not None and (target_width != width or target_height != height):
						frame_u8 = ndimage.zoom(frame_u8, (target_height/height, target_width/width, 1), order=1, mode='nearest')
					elif target_width != width or target_height != height:
						# Fallback: simple nearest neighbor resize
						y_indices = np.linspace(0, height-1, target_height).astype(int)
						x_indices = np.linspace(0, width-1, target_width).astype(int)
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

		output_layout.addWidget(QLabel("Output Folder:"), 0, 0)
		self.output_folder_edit = QLineEdit()
		output_layout.addWidget(self.output_folder_edit, 0, 1)
		browse_output_btn = QPushButton("Browse...")
		browse_output_btn.clicked.connect(self.browse_output_folder)
		output_layout.addWidget(browse_output_btn, 0, 2)

		self.preserve_names_checkbox = QCheckBox("Preserve original filenames")
		self.preserve_names_checkbox.setChecked(True)
		output_layout.addWidget(self.preserve_names_checkbox, 1, 0, 1, 3)

		# New: Output format and FPS for video
		output_layout.addWidget(QLabel("Output Format:"), 2, 0)
		self.output_format_combo = QComboBox()
		self.output_format_combo.addItems(["NRRD (.nrrd)", "MP4 Video (.mp4)"])
		output_layout.addWidget(self.output_format_combo, 2, 1, 1, 2)

		output_layout.addWidget(QLabel("FPS:"), 3, 0)
		self.fps_edit = QLineEdit("10")
		self.fps_edit.setMaximumWidth(100)
		output_layout.addWidget(self.fps_edit, 3, 1)

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
			self, "Select Configuration File", "", "JSON Files (*.json);;All Files (*)"
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
				"GB": 1024 * 1024 * 1024
			}
			return int(size * multipliers.get(unit, 1))
		except ValueError:
			return None

	def get_file_size_constraints(self):
		"""Get min and max file size in bytes from GUI inputs"""
		min_size = self.convert_size_to_bytes(
			self.min_size_edit.text(),
			self.min_size_unit.currentText()
		)
		max_size = self.convert_size_to_bytes(
			self.max_size_edit.text(),
			self.max_size_unit.currentText()
		)

		# Default to 0 for min if invalid
		if min_size is None:
			min_size = 0

		return min_size, max_size

	def start_processing(self):
		# Validate inputs
		input_folder = self.input_folder_edit.text().strip()
		file_pattern = self.file_pattern_edit.text().strip()
		config_file = self.config_file_edit.text().strip()
		output_folder = self.output_folder_edit.text().strip()

		if not input_folder or not os.path.exists(input_folder):
			QMessageBox.warning(self, "Warning", "Please select a valid input folder")
			return

		if not file_pattern:
			QMessageBox.warning(self, "Warning", "Please specify a file pattern")
			return

		if not config_file or not os.path.exists(config_file):
			QMessageBox.warning(self, "Warning", "Please select a valid configuration file")
			return

		if not output_folder:
			QMessageBox.warning(self, "Warning", "Please specify an output folder")
			return

		# Get file size constraints
		min_file_size, max_file_size = self.get_file_size_constraints()

		# Output format and fps
		output_format = "mp4" if self.output_format_combo.currentIndex() == 1 else "nrrd"
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
			video_fps=video_fps
		)

		# Connect signals
		self.processor_thread.progress_updated.connect(self.progress_bar.setValue)
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