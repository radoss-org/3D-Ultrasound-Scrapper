import base64
import json
import os
import tempfile

import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox
from shared.config_parsing import params_to_config
from shared.dicom_io import get_original_header_bytes, parse_dicom_tags
from shared.geometry import _identity_corners_for_volume
from shared.geometry import apply_crop as apply_crop_to_volume
from shared.geometry import apply_orientation as shared_apply_orientation
from shared.geometry import are_corners_identity as shared_are_corners_identity
from shared.geometry import build_export_volume as shared_build_export_volume
from shared.geometry import warp_slice as shared_warp_slice
from shared.image_enhancement import (
    apply_enhancement as shared_apply_enhancement,
)
from shared.nrrd_io import build_nrrd_header_text
from shared.nrrd_io import save_nrrd as save_nrrd_shared
from shared.raw_io import find_header_end as shared_find_header_end
from shared.raw_io import get_pixel_info as shared_get_pixel_info
from shared.raw_io import read_raw_volume
from shared.types import (
    CornerWarpParams,
    CropParams,
    CurveParams,
    NrrdParams,
    OrientationParams,
    RawLayoutParams,
    VolumeParams,
)
from state_helpers import RawImageGuessStateMixin
from ui_builder import RawImageGuessUiMixin


class RawImageGuessQt(
    QMainWindow, RawImageGuessUiMixin, RawImageGuessStateMixin
):
    def __init__(self, config_file=None):
        super().__init__()
        self.image_data = None
        self.current_slice = 0

        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.mouse_pressed = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        self.brightness = 0.0
        self.contrast = 1.0
        self.gamma = 1.0
        self.vmin = None
        self.vmax = None

        self.corner_positions = None
        self.selected_corner_index = 0
        self.use_corner_symmetry = True
        self.show_corner_notes = True

        self.curve_x_pos = 0.0
        self.curve_x_neg = 0.0
        self.curve_y_pos = 0.0
        self.curve_y_neg = 0.0
        self.curve_z_pos = 0.0
        self.curve_z_neg = 0.0

        self.crop_top = 0
        self.crop_bottom = 0
        self.crop_left = 0
        self.crop_right = 0
        self.dicom_ds = None
        self.ob_tags = []
        self.dicom_selected_tag = None

        self.orientation_ops = []
        self.header_end_marker = "[SCALPEL]\ncount=0"
        self.use_header_offset = False
        self.header_offset = 0

        self.init_ui()

        if config_file and os.path.exists(config_file):
            self.load_config(config_file)

    def on_file_param_changed(self):
        self.update_all_labels()
        self.load_image()

    def on_visual_param_changed(self):
        self.update_all_labels()
        self.update_enhancement_params()
        self.update_curve_params()
        self.update_crop_params()
        self.show_corner_notes = self.corner_notes_checkbox.isChecked()
        if self.image_data is not None:
            self.update_slice_display()

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image File", "", "All Files (*)"
        )
        if file_path:
            self.current_file = file_path
            self.file_path_label.setText(
                f"File: {os.path.basename(file_path)}"
            )
            self.parse_dicom()

    def load_image(self):
        if not os.path.exists(self.current_file):
            self.status_text.append(
                f"Error: File {self.current_file} not found"
            )
            return

        try:
            header_size_from_marker = 0
            if self.header_end_marker.strip():
                raw_bytes_for_search = b""
                if (
                    self.dicom_selected_tag is not None
                    and self.dicom_ds is not None
                    and self.dicom_selected_tag in self.dicom_ds
                ):
                    elem = self.dicom_ds[self.dicom_selected_tag]
                    if isinstance(elem.value, (bytes, bytearray)):
                        raw_bytes_for_search = bytes(elem.value)
                else:
                    try:
                        with open(self.current_file, "rb") as f:
                            raw_bytes_for_search = f.read(512 * 1024)
                    except Exception:
                        pass

                if raw_bytes_for_search:
                    header_size_from_marker = shared_find_header_end(
                        raw_bytes_for_search, self.header_end_marker
                    )

            if header_size_from_marker > 0:
                self.header_size_slider.blockSignals(True)
                self.header_size_slider.setValue(header_size_from_marker)
                self.header_size_slider.blockSignals(False)
                self.update_all_labels()
                self.status_text.append(
                    f"Auto-detected header size: "
                    f"{header_size_from_marker} bytes"
                )

            width = self.width_slider.value()
            height = self.height_slider.value()
            pixel_type = self.pixel_type_combo.currentText()
            endianness = self.endianness_combo.currentText()

            volume_params = VolumeParams(
                width=width,
                height=height,
                depth=self.depth_slider.value(),
                pixel_type=pixel_type,
                endianness=endianness,
                spacing_x=self.spacing_x_spin.value(),
                spacing_y=self.spacing_y_spin.value(),
                spacing_z=self.spacing_z_spin.value(),
            )
            raw_layout = RawLayoutParams(
                header_size=self.header_size_slider.value(),
                footer_size=self.footer_size_slider.value(),
                row_stride=self.row_stride_slider.value(),
                row_padding=self.row_padding_slider.value(),
                slice_stride=self.slice_stride_slider.value(),
                skip_slices=self.skip_slices_slider.value(),
                header_end_marker=self.header_end_marker,
                use_header_offset=self.use_header_offset,
                header_offset=self.header_offset,
            )

            raw_source_name = self.current_file
            effective_file_path = self.current_file

            if (
                self.dicom_selected_tag is not None
                and self.dicom_ds is not None
                and self.dicom_selected_tag in self.dicom_ds
            ):
                elem = self.dicom_ds[self.dicom_selected_tag]
                if not isinstance(elem.value, (bytes, bytearray)):
                    self.status_text.append(
                        "Error: Selected DICOM tag does not contain raw bytes"
                    )
                    return
                raw_bytes = bytes(elem.value)
                tmp = tempfile.NamedTemporaryFile(suffix=".raw", delete=False)
                tmp.write(raw_bytes)
                tmp.close()
                effective_file_path = tmp.name
                raw_source_name = (
                    f"DICOM tag (0x{self.dicom_selected_tag.group:04X}"
                    f"{self.dicom_selected_tag.element:04X})"
                )

            volume = read_raw_volume(
                effective_file_path, volume_params, raw_layout
            )

            if (
                self.dicom_selected_tag is not None
                and self.dicom_ds is not None
                and self.dicom_selected_tag in self.dicom_ds
            ):
                try:
                    os.unlink(effective_file_path)
                except OSError:
                    pass

            if volume is None:
                self.status_text.append("Error: No valid slices could be read")
                return

            self.image_data = volume
            final_depth = volume.shape[0]

            self.slice_slider.setMaximum(final_depth - 1)
            self.current_slice = min(self.current_slice, final_depth - 1)
            self.slice_slider.setValue(self.current_slice)

            self.reset_zoom()
            self.apply_orientation_ops()
            self.reset_corners()

            self.crop_top_slider.setMaximum(height - 1)
            self.crop_bottom_slider.setMaximum(height - 1)
            self.crop_left_slider.setMaximum(width - 1)
            self.crop_right_slider.setMaximum(width - 1)

            self.update_slice_display()

            self.status_text.append(
                f"Loaded image from {raw_source_name}: "
                f"{width}x{height}x{final_depth} "
                f"(header={self.header_size_slider.value()})"
            )

        except Exception as e:
            self.status_text.append(f"Error loading image: {str(e)}")

    def apply_orientation_ops(self):
        if not self.orientation_ops or self.image_data is None:
            return

        ops = tuple(tuple(op) for op in self.orientation_ops)
        self.image_data = shared_apply_orientation(
            self.image_data, OrientationParams(orientation_ops=ops)
        )

        for op in self.orientation_ops:
            if op[0] == "flip" and op[1] == "z":
                self.current_slice = (
                    self.image_data.shape[0] - 1 - self.current_slice
                )

        self.slice_slider.setMaximum(self.image_data.shape[0] - 1)
        self.current_slice = min(
            self.current_slice, self.image_data.shape[0] - 1
        )

    def update_slice_display(self):
        if self.image_data is None:
            return

        self.current_slice = self.slice_slider.value()
        max_slice = self.image_data.shape[0] - 1
        self.slice_label.setText(f"{self.current_slice}/{max_slice}")

        slice_data = self.warp_slice(self.current_slice)
        crop_params = CropParams(
            crop_top=self.crop_top,
            crop_bottom=self.crop_bottom,
            crop_left=self.crop_left,
            crop_right=self.crop_right,
        )
        from shared.geometry import apply_crop_to_slice as shared_crop_slice

        slice_data = shared_crop_slice(slice_data, crop_params)

        if len(slice_data.shape) == 2:
            slice_data = self.apply_enhancement(slice_data)

        self.ax.clear()
        if len(slice_data.shape) == 3:
            self.ax.imshow(slice_data)
        else:
            self.ax.imshow(
                slice_data, cmap="gray", vmin=self.vmin, vmax=self.vmax
            )

        aspect_ratio = (
            self.spacing_y_spin.value() / self.spacing_x_spin.value()
        )
        self.ax.set_aspect(aspect_ratio)

        z_pos = self.current_slice * self.spacing_z_spin.value()
        crop_parts = []
        if self.crop_top or self.crop_bottom:
            crop_parts.append(f"T{self.crop_top}/B{self.crop_bottom}")
        if self.crop_left or self.crop_right:
            crop_parts.append(f"L{self.crop_left}/R{self.crop_right}")
        crop_text = f" | Crop: {', '.join(crop_parts)}" if crop_parts else ""
        title = f"Slice {self.current_slice} (z={z_pos:.3f}){crop_text}"
        self.ax.set_title(title)
        self.ax.axis("off")

        if self.show_corner_notes:
            self.draw_corner_notes()

        self.apply_zoom_and_pan()
        self.canvas.draw()

    def draw_corner_notes(self):
        if self.corner_positions is None:
            return

        d, h, w = self.image_data.shape[:3]
        t = 0.0 if d <= 1 else self.current_slice / (d - 1)

        cp = self.corner_positions
        corners = ["TL", "TR", "BR", "BL"]

        h_display = h - self.crop_top - self.crop_bottom
        w_display = w - self.crop_left - self.crop_right

        positions = [
            (4, 12, "left", "top"),
            (w_display - 4, 12, "right", "top"),
            (w_display - 4, h_display - 4, "right", "bottom"),
            (4, h_display - 4, "left", "bottom"),
        ]
        pairs = [(0, 4), (1, 5), (3, 7), (2, 6)]

        for tag, (sx, sy, ha, va), (i0, i1) in zip(corners, positions, pairs):
            p = (1.0 - t) * cp[i0] + t * cp[i1]
            txt = f"{tag}: X={p[0]:.1f}, Y={p[1]:.1f}, Z={p[2]:.1f}"
            self.ax.text(
                sx,
                sy,
                txt,
                color=(0.2, 1.0, 1.0),
                ha=ha,
                va=va,
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    fc=(0, 0, 0, 0.35),
                    ec=(0, 0, 0, 0.6),
                ),
            )

    def apply_enhancement(self, image):
        return shared_apply_enhancement(
            image, self.brightness, self.contrast, self.gamma
        )

    def apply_zoom_and_pan(self):
        if self.image_data is None:
            return

        from shared.geometry import apply_crop_to_slice as shared_crop_slice

        slice_data = self.warp_slice(self.current_slice)
        crop_params = CropParams(
            crop_top=self.crop_top,
            crop_bottom=self.crop_bottom,
            crop_left=self.crop_left,
            crop_right=self.crop_right,
        )
        slice_data = shared_crop_slice(slice_data, crop_params)
        height, width = slice_data.shape[:2]

        center_x = width / 2 + self.pan_x
        center_y = height / 2 + self.pan_y
        half_width = width / (2 * self.zoom_factor)
        half_height = height / (2 * self.zoom_factor)

        self.ax.set_xlim([center_x - half_width, center_x + half_width])
        self.ax.set_ylim([center_y + half_height, center_y - half_height])

    def zoom_in(self):
        self.zoom_factor *= 1.5
        self.update_zoom_display()

    def zoom_out(self):
        self.zoom_factor = max(0.1, self.zoom_factor / 1.5)
        self.update_zoom_display()

    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update_zoom_display()

    def update_zoom_display(self):
        self.zoom_label.setText(f"{int(self.zoom_factor * 100)}%")
        if self.image_data is not None:
            self.apply_zoom_and_pan()
            self.canvas.draw()

    def on_scroll(self, event):
        if self.image_data is None or event.inaxes != self.ax:
            return
        if event.button == "up":
            self.zoom_factor *= 1.2
        elif event.button == "down":
            self.zoom_factor = max(0.1, self.zoom_factor / 1.2)
        self.update_zoom_display()

    def on_mouse_press(self, event):
        if event.inaxes == self.ax and event.button == 1:
            self.mouse_pressed = True
            self.last_mouse_x = event.xdata
            self.last_mouse_y = event.ydata

    def on_mouse_move(self, event):
        if (
            not self.mouse_pressed
            or event.xdata is None
            or event.ydata is None
        ):
            return
        dx = self.last_mouse_x - event.xdata
        dy = self.last_mouse_y - event.ydata
        self.pan_x += dx
        self.pan_y += dy
        self.last_mouse_x = event.xdata
        self.last_mouse_y = event.ydata
        self.apply_zoom_and_pan()
        self.canvas.draw()

    def on_mouse_release(self, event):
        self.mouse_pressed = False

    def flip_axis(self, axis):
        if self.image_data is None:
            return
        axis_map = {"z": 0, "y": 1, "x": 2}
        self.image_data = np.flip(self.image_data, axis=axis_map[axis])
        if axis == "z":
            self.current_slice = (
                self.image_data.shape[0] - 1 - self.current_slice
            )
            self.slice_slider.setValue(self.current_slice)
        self.orientation_ops.append(("flip", axis))
        self.reset_corners()
        self.update_slice_display()

    def rotate_axis(self, axis, direction):
        if self.image_data is None:
            return
        k = 1 if direction > 0 else 3
        if axis == "z":
            self.image_data = np.rot90(self.image_data, k=k, axes=(1, 2))
        elif axis == "x":
            self.image_data = np.rot90(self.image_data, k=k, axes=(0, 1))
        elif axis == "y":
            self.image_data = np.rot90(self.image_data, k=k, axes=(0, 2))

        self.orientation_ops.append(("rotate", axis, direction))
        self.slice_slider.setMaximum(self.image_data.shape[0] - 1)
        self.current_slice = min(
            self.current_slice, self.image_data.shape[0] - 1
        )
        self.slice_slider.setValue(self.current_slice)
        self.reset_corners()
        self.update_slice_display()

    def reset_corners(self):
        d, h, w = self._volume_dims()
        dummy = np.empty((d, h, w), dtype=np.uint8)
        self.corner_positions = _identity_corners_for_volume(dummy)
        self.setup_corner_slider_ranges()
        self.update_slice_display()

    def on_corner_selection_changed(self, idx):
        if self.use_corner_symmetry:
            self.selected_corner_index = [0, 2][idx]
        else:
            self.selected_corner_index = idx
        self.sync_corner_sliders()
        self.update_slice_display()

    def on_corner_slider_changed(self):
        x = self.corner_x_slider.value()
        y = self.corner_y_slider.value()
        z = self.corner_z_slider.value()

        if self.corner_positions is not None:
            self.corner_positions[self.selected_corner_index] = [x, y, z]

            if self.use_corner_symmetry and self.selected_corner_index in [
                0,
                2,
            ]:
                self.apply_corner_symmetry()

        self.update_all_labels()
        if self.image_data is not None:
            self.update_slice_display()

    def apply_corner_symmetry(self):
        if self.corner_positions is None or self.image_data is None:
            return

        d, h, w = self.image_data.shape[:3]
        center = [(w - 1) / 2, (h - 1) / 2, (d - 1) / 2]

        c000 = self.corner_positions[0]
        c010 = self.corner_positions[2]

        self.corner_positions[1] = [
            2 * center[0] - c000[0],
            c000[1],
            c000[2],
        ]
        self.corner_positions[3] = [
            2 * center[0] - c010[0],
            c010[1],
            c010[2],
        ]
        self.corner_positions[4] = [
            c000[0],
            c000[1],
            2 * center[2] - c000[2],
        ]
        self.corner_positions[5] = [
            self.corner_positions[1][0],
            self.corner_positions[1][1],
            2 * center[2] - self.corner_positions[1][2],
        ]
        self.corner_positions[6] = [
            c010[0],
            c010[1],
            2 * center[2] - c010[2],
        ]
        self.corner_positions[7] = [
            self.corner_positions[3][0],
            self.corner_positions[3][1],
            2 * center[2] - self.corner_positions[3][2],
        ]

    def on_corner_symmetry_toggled(self, checked):
        self.use_corner_symmetry = checked
        self.update_corner_combo_items()
        if checked:
            self.apply_corner_symmetry()
        self.sync_corner_sliders()
        self.update_slice_display()

    def warp_slice(self, z_idx):
        warp_params = CornerWarpParams(corner_positions=self.corner_positions)
        curve_params = CurveParams(
            curve_x_pos=self.curve_x_pos,
            curve_x_neg=self.curve_x_neg,
            curve_y_pos=self.curve_y_pos,
            curve_y_neg=self.curve_y_neg,
            curve_z_pos=self.curve_z_pos,
            curve_z_neg=self.curve_z_neg,
        )
        return shared_warp_slice(
            self.image_data, z_idx, warp_params, curve_params
        )

    def reset_enhancement(self):
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)
        self.gamma_slider.setValue(100)
        self.window_min_slider.setValue(0)
        self.window_max_slider.setValue(100)

    def reset_curves(self):
        for slider in [
            self.curve_x_pos_slider,
            self.curve_x_neg_slider,
            self.curve_y_pos_slider,
            self.curve_y_neg_slider,
            self.curve_z_pos_slider,
            self.curve_z_neg_slider,
        ]:
            slider.setValue(0)

    def reset_crop(self):
        self.crop_top_slider.setValue(0)
        self.crop_bottom_slider.setValue(0)
        self.crop_left_slider.setValue(0)
        self.crop_right_slider.setValue(0)

    def apply_crop(self):
        if self.image_data is None or not (
            self.crop_top
            or self.crop_bottom
            or self.crop_left
            or self.crop_right
        ):
            return

        h = self.image_data.shape[1]
        w = self.image_data.shape[2]

        if self.crop_top + self.crop_bottom >= h:
            QMessageBox.warning(
                self,
                "Invalid Crop",
                "Vertical crop values exceed image height",
            )
            return
        if self.crop_left + self.crop_right >= w:
            QMessageBox.warning(
                self,
                "Invalid Crop",
                "Horizontal crop values exceed image width",
            )
            return

        crop_params = CropParams(
            crop_top=self.crop_top,
            crop_bottom=self.crop_bottom,
            crop_left=self.crop_left,
            crop_right=self.crop_right,
        )
        self.image_data = apply_crop_to_volume(self.image_data, crop_params)

        self.status_text.append(
            f"Applied crop: T{self.crop_top}/B{self.crop_bottom}/"
            f"L{self.crop_left}/R{self.crop_right}"
        )

        self.crop_top_slider.setValue(0)
        self.crop_bottom_slider.setValue(0)
        self.crop_left_slider.setValue(0)
        self.crop_right_slider.setValue(0)

        new_height = self.image_data.shape[1]
        new_width = self.image_data.shape[2]
        self.crop_top_slider.setMaximum(new_height - 1)
        self.crop_bottom_slider.setMaximum(new_height - 1)
        self.crop_left_slider.setMaximum(new_width - 1)
        self.crop_right_slider.setMaximum(new_width - 1)

        self.reset_corners()
        self.reset_zoom()
        self.update_slice_display()

    def generate_nrrd_header(self):
        if not os.path.exists(self.current_file):
            QMessageBox.warning(
                self, "Warning", "No valid input file selected"
            )
            return

        try:
            width = self.width_slider.value()
            height = self.height_slider.value()
            depth = self.depth_slider.value()
            header_size = self.header_size_slider.value()
            pixel_type = self.pixel_type_combo.currentText()
            endianness = self.endianness_combo.currentText()

            base_name = os.path.splitext(self.current_file)[0]
            header_file = base_name + ".nhdr"

            _, _, _ = shared_get_pixel_info(pixel_type, endianness)

            volume_params = VolumeParams(
                width=width,
                height=height,
                depth=depth,
                pixel_type=pixel_type,
                endianness=endianness,
                spacing_x=self.spacing_x_spin.value(),
                spacing_y=self.spacing_y_spin.value(),
                spacing_z=self.spacing_z_spin.value(),
            )
            nrrd_params = NrrdParams()

            extras = []
            if header_size > 0:
                extras.append(f"byte skip: {header_size}")
            extras.append(f"data file: {os.path.basename(self.current_file)}")

            header_text = build_nrrd_header_text(
                self.image_data,
                volume_params,
                nrrd_params,
                header_extras=extras,
            )

            with open(header_file, "w") as f:
                f.write(header_text)

            self.status_text.append(f"Generated NRRD header: {header_file}")
            QMessageBox.information(
                self, "Success", f"NRRD header generated: {header_file}"
            )

        except Exception as e:
            self.status_text.append(f"Error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed: {str(e)}")

    def save_as_nrrd(self):
        if self.image_data is None:
            QMessageBox.warning(self, "Warning", "No image data to save")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save NRRD File", "", "NRRD Files (*.nrrd);;All Files (*)"
        )
        if not file_path:
            return

        try:
            export_data = self.build_export_volume()
            pixel_type = self.pixel_type_combo.currentText()
            endianness = self.endianness_combo.currentText()

            if export_data.ndim == 4:
                depth, height, width, components = export_data.shape
            else:
                depth, height, width = export_data.shape
                components = 1

            _ = components

            original_header_bytes = self.get_original_header_bytes()
            header_extras = []
            if original_header_bytes:
                original_header_b64 = base64.b64encode(
                    original_header_bytes
                ).decode("ascii")
                header_extras.append(
                    f"raw_header_size:={len(original_header_bytes)}"
                )
                header_extras.append(
                    f"raw_header_base64:={original_header_b64}"
                )
                header_extras.append(
                    f"raw_header_source:="
                    f"{os.path.basename(self.current_file)}"
                )

            volume_params = VolumeParams(
                width=width,
                height=height,
                depth=depth,
                pixel_type=pixel_type,
                endianness=endianness,
                spacing_x=self.spacing_x_spin.value(),
                spacing_y=self.spacing_y_spin.value(),
                spacing_z=self.spacing_z_spin.value(),
            )
            nrrd_params = NrrdParams()

            save_nrrd_shared(
                file_path,
                export_data,
                volume_params,
                nrrd_params,
                header_extras=header_extras,
            )

            self.status_text.append(f"Saved NRRD file: {file_path}")
            QMessageBox.information(
                self, "Success", f"NRRD file saved: {file_path}"
            )

        except Exception as e:
            self.status_text.append(f"Error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed: {str(e)}")

    def build_export_volume(self):
        warp_params = CornerWarpParams(corner_positions=self.corner_positions)
        curve_params = CurveParams(
            curve_x_pos=self.curve_x_pos,
            curve_x_neg=self.curve_x_neg,
            curve_y_pos=self.curve_y_pos,
            curve_y_neg=self.curve_y_neg,
            curve_z_pos=self.curve_z_pos,
            curve_z_neg=self.curve_z_neg,
        )
        crop_params = CropParams(
            crop_top=self.crop_top,
            crop_bottom=self.crop_bottom,
            crop_left=self.crop_left,
            crop_right=self.crop_right,
        )
        return shared_build_export_volume(
            self.image_data, warp_params, curve_params, crop_params
        )

    def are_corners_identity(self):
        return shared_are_corners_identity(
            self.image_data, self.corner_positions
        )

    def get_current_config(self):
        from shared.config_parsing import ParsedParams

        p = ParsedParams(
            volume=VolumeParams(
                width=self.width_slider.value(),
                height=self.height_slider.value(),
                depth=self.depth_slider.value(),
                pixel_type=self.pixel_type_combo.currentText(),
                endianness=self.endianness_combo.currentText(),
                spacing_x=self.spacing_x_spin.value(),
                spacing_y=self.spacing_y_spin.value(),
                spacing_z=self.spacing_z_spin.value(),
            ),
            raw_layout=RawLayoutParams(
                header_size=self.header_size_slider.value(),
                footer_size=self.footer_size_slider.value(),
                row_stride=self.row_stride_slider.value(),
                row_padding=self.row_padding_slider.value(),
                slice_stride=self.slice_stride_slider.value(),
                skip_slices=self.skip_slices_slider.value(),
                header_end_marker=self.header_end_marker,
                use_header_offset=self.use_header_offset,
                header_offset=self.header_offset,
            ),
            orientation=OrientationParams(
                orientation_ops=tuple(tuple(op) for op in self.orientation_ops)
            ),
            warp=CornerWarpParams(
                corner_positions=(
                    self.corner_positions.tolist()
                    if self.corner_positions is not None
                    else None
                )
            ),
            curve=CurveParams(
                curve_x_pos=self.curve_x_pos,
                curve_x_neg=self.curve_x_neg,
                curve_y_pos=self.curve_y_pos,
                curve_y_neg=self.curve_y_neg,
                curve_z_pos=self.curve_z_pos,
                curve_z_neg=self.curve_z_neg,
            ),
            crop=CropParams(
                crop_top=self.crop_top,
                crop_bottom=self.crop_bottom,
                crop_left=self.crop_left,
                crop_right=self.crop_right,
            ),
            brightness=self.brightness_slider.value(),
            contrast=self.contrast_slider.value(),
            gamma=self.gamma_slider.value(),
            window_min=self.window_min_slider.value(),
            window_max=self.window_max_slider.value(),
            current_file=self.current_file,
        )
        config = params_to_config(p)
        config["use_corner_symmetry"] = self.use_corner_symmetry
        config["show_corner_notes"] = self.show_corner_notes
        config["selected_corner_index"] = self.selected_corner_index
        return config

    def apply_config(self, config):
        try:
            all_widgets = [
                self.pixel_type_combo,
                self.endianness_combo,
                self.header_size_slider,
                self.footer_size_slider,
                self.width_slider,
                self.height_slider,
                self.row_stride_slider,
                self.row_padding_slider,
                self.depth_slider,
                self.skip_slices_slider,
                self.slice_stride_slider,
                self.spacing_x_spin,
                self.spacing_y_spin,
                self.spacing_z_spin,
                self.brightness_slider,
                self.contrast_slider,
                self.gamma_slider,
                self.window_min_slider,
                self.window_max_slider,
                self.curve_x_pos_slider,
                self.curve_x_neg_slider,
                self.curve_y_pos_slider,
                self.curve_y_neg_slider,
                self.curve_z_pos_slider,
                self.curve_z_neg_slider,
                self.crop_top_slider,
                self.crop_bottom_slider,
                self.crop_left_slider,
                self.crop_right_slider,
                self.corner_notes_checkbox,
                self.corner_symmetry_checkbox,
                self.header_offset_checkbox,
            ]

            for widget in all_widgets:
                widget.blockSignals(True)

            if "current_file" in config:
                self.current_file = config["current_file"]
                self.file_path_label.setText(
                    f"File: {os.path.basename(self.current_file)}"
                )

            if "pixel_type" in config:
                idx = self.pixel_type_combo.findText(config["pixel_type"])
                if idx >= 0:
                    self.pixel_type_combo.setCurrentIndex(idx)

            if "endianness" in config:
                idx = self.endianness_combo.findText(config["endianness"])
                if idx >= 0:
                    self.endianness_combo.setCurrentIndex(idx)

            slider_map = {
                "header_size": self.header_size_slider,
                "footer_size": self.footer_size_slider,
                "width": self.width_slider,
                "height": self.height_slider,
                "row_stride": self.row_stride_slider,
                "row_padding": self.row_padding_slider,
                "depth": self.depth_slider,
                "skip_slices": self.skip_slices_slider,
                "slice_stride": self.slice_stride_slider,
                "brightness": self.brightness_slider,
                "contrast": self.contrast_slider,
                "gamma": self.gamma_slider,
                "window_min": self.window_min_slider,
                "window_max": self.window_max_slider,
                "curve_x_pos": self.curve_x_pos_slider,
                "curve_x_neg": self.curve_x_neg_slider,
                "curve_y_pos": self.curve_y_pos_slider,
                "curve_y_neg": self.curve_y_neg_slider,
                "curve_z_pos": self.curve_z_pos_slider,
                "curve_z_neg": self.curve_z_neg_slider,
                "crop_top": self.crop_top_slider,
                "crop_bottom": self.crop_bottom_slider,
                "crop_left": self.crop_left_slider,
                "crop_right": self.crop_right_slider,
            }

            for key, slider in slider_map.items():
                if key in config:
                    slider.setValue(int(config[key]))

            spinbox_map = {
                "spacing_x": self.spacing_x_spin,
                "spacing_y": self.spacing_y_spin,
                "spacing_z": self.spacing_z_spin,
            }

            for key, spinbox in spinbox_map.items():
                if key in config:
                    spinbox.setValue(float(config[key]))

            if "header_offset" in config:
                self.header_offset_spin.setValue(
                    int(float(config["header_offset"]))
                )

            if "show_corner_notes" in config:
                self.corner_notes_checkbox.setChecked(
                    bool(config["show_corner_notes"])
                )
            if "use_corner_symmetry" in config:
                self.corner_symmetry_checkbox.setChecked(
                    bool(config["use_corner_symmetry"])
                )
            if "use_header_offset" in config:
                self.header_offset_checkbox.setChecked(
                    bool(config["use_header_offset"])
                )

            self.use_corner_symmetry = config.get("use_corner_symmetry", True)
            self.show_corner_notes = config.get("show_corner_notes", True)
            self.use_header_offset = config.get("use_header_offset", False)

            if "selected_corner_index" in config:
                self.selected_corner_index = int(
                    config["selected_corner_index"]
                )

            if "orientation_ops" in config:
                self.orientation_ops = list(config["orientation_ops"])

            for widget in all_widgets:
                widget.blockSignals(False)

            self.update_all_labels()
            self.update_enhancement_params()
            self.update_curve_params()
            self.update_crop_params()
            self.update_corner_combo_items()

            if "header_end_marker" in config:
                self.header_end_marker = config["header_end_marker"]
                if hasattr(self, "header_marker_edit"):
                    self.header_marker_edit.blockSignals(True)
                    self.header_marker_edit.setPlainText(
                        self.header_end_marker
                    )
                    self.header_marker_edit.blockSignals(False)

            if "corner_positions" in config and config["corner_positions"]:
                self._pending_corner_positions = np.array(
                    config["corner_positions"]
                )
            else:
                self._pending_corner_positions = None

            self.load_image()
            self.status_text.append("Configuration loaded successfully")

        except Exception as e:
            self.status_text.append(f"Error applying config: {str(e)}")
            QMessageBox.critical(
                self, "Error", f"Failed to apply config: {str(e)}"
            )

    def save_config(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not file_path:
            return

        try:
            config = self.get_current_config()
            with open(file_path, "w") as f:
                json.dump(config, f, indent=4)
            self.status_text.append(f"Config saved: {file_path}")
            QMessageBox.information(
                self, "Success", f"Config saved: {file_path}"
            )
        except Exception as e:
            self.status_text.append(f"Error saving config: {str(e)}")
            QMessageBox.critical(
                self, "Error", f"Failed to save config: {str(e)}"
            )

    def load_config_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if file_path:
            self.load_config(file_path)

    def load_config(self, file_path):
        try:
            with open(file_path, "r") as f:
                config = json.load(f)
            self.apply_config(config)
            self.status_text.append(f"Config loaded: {file_path}")
        except Exception as e:
            self.status_text.append(f"Error loading config: {str(e)}")
            QMessageBox.critical(
                self, "Error", f"Failed to load config: {str(e)}"
            )

    def parse_dicom(self):
        try:
            self.ob_tags, self.dicom_ds = parse_dicom_tags(self.current_file)

            self.dicom_tag_combo.clear()
            self.dicom_tag_combo.addItem("-- No tag selected --")
            for name, length, tag in self.ob_tags:
                self.dicom_tag_combo.addItem(
                    f"{name} (0x{tag.group:04X}{tag.element:04X}) - "
                    f"{length} bytes"
                )

            self.dicom_tags_label.setText(
                f"Found {len(self.ob_tags)} tags > 100 bytes"
            )
            self.dicom_selected_tag = None
        except Exception as e:
            self.status_text.append(f"DICOM parse error: {str(e)}")
            self.dicom_ds = None
            self.ob_tags = []
            self.dicom_tag_combo.clear()
            self.dicom_tags_label.setText("DICOM parse error")

    def on_dicom_tag_changed(self, index):
        if index <= 0:
            self.dicom_selected_tag = None
        else:
            self.dicom_selected_tag = self.ob_tags[index - 1][2]
        self.on_file_param_changed()

    def get_original_header_bytes(self):
        return get_original_header_bytes(
            self.current_file,
            self.header_size_slider.value(),
            self.dicom_ds,
            self.dicom_selected_tag,
        )

    def on_header_marker_changed(self):
        new_marker = self.header_marker_edit.toPlainText().strip()
        if new_marker != self.header_end_marker:
            self.header_end_marker = new_marker
            self.on_file_param_changed()

    def on_header_offset_toggled(self, checked):
        self.use_header_offset = checked
        self.header_offset_spin.setEnabled(checked)
        if checked:
            self.on_file_param_changed()

    def on_header_offset_changed(self):
        self.header_offset = self.header_offset_spin.value()
        self.update_all_labels()
        self.load_image()
