import os

import numpy as np

DEFAULT_IMAGE_FILE = "sample_image.raw"


class RawImageGuessStateMixin:
    def adjust_slider(self, slider, direction):
        step = max(1, (slider.maximum() - slider.minimum()) // 100)
        new_value = slider.value() + (direction * step)
        slider.setValue(np.clip(new_value, slider.minimum(), slider.maximum()))

    def load_default_parameters(self):
        self.current_file = DEFAULT_IMAGE_FILE
        self.header_end_marker = "[SCALPEL]\ncount=0"
        self.use_header_offset = False
        self.header_offset = 0
        self.reset_corners()
        self.update_corner_combo_items()
        self.update_all_labels()

    def update_all_labels(self):
        for slider in [
            self.header_size_slider,
            self.footer_size_slider,
            self.width_slider,
            self.height_slider,
            self.row_stride_slider,
            self.row_padding_slider,
            self.depth_slider,
            self.skip_slices_slider,
            self.slice_stride_slider,
        ]:
            if hasattr(slider, "label_widget"):
                val = slider.value()
                if slider == self.row_stride_slider and val == 0:
                    slider.label_widget.setText("0 (auto)")
                else:
                    slider.label_widget.setText(str(val))

        if hasattr(self.brightness_slider, "label_widget"):
            self.brightness_slider.label_widget.setText(
                str(self.brightness_slider.value())
            )
        if hasattr(self.contrast_slider, "label_widget"):
            self.contrast_slider.label_widget.setText(
                f"{self.contrast_slider.value()}%"
            )
        if hasattr(self.gamma_slider, "label_widget"):
            self.gamma_slider.label_widget.setText(
                f"{self.gamma_slider.value() / 100.0:.1f}"
            )
        if hasattr(self.window_min_slider, "label_widget"):
            val = self.window_min_slider.value()
            self.window_min_slider.label_widget.setText(
                "Auto" if val == 0 else f"{val}%"
            )
        if hasattr(self.window_max_slider, "label_widget"):
            val = self.window_max_slider.value()
            self.window_max_slider.label_widget.setText(
                "Auto" if val == 100 else f"{val}%"
            )

        for slider in [
            self.curve_x_pos_slider,
            self.curve_x_neg_slider,
            self.curve_y_pos_slider,
            self.curve_y_neg_slider,
            self.curve_z_pos_slider,
            self.curve_z_neg_slider,
        ]:
            if hasattr(slider, "label_widget"):
                slider.label_widget.setText(str(slider.value()))

        for slider in [
            self.crop_top_slider,
            self.crop_bottom_slider,
            self.crop_left_slider,
            self.crop_right_slider,
        ]:
            if hasattr(slider, "label_widget"):
                slider.label_widget.setText(str(slider.value()))

        for slider in [
            self.corner_x_slider,
            self.corner_y_slider,
            self.corner_z_slider,
        ]:
            if hasattr(slider, "label_widget"):
                slider.label_widget.setText(str(slider.value()))

        self.header_offset_label.setText(str(self.header_offset_spin.value()))

    def update_enhancement_params(self):
        self.brightness = self.brightness_slider.value()
        self.contrast = self.contrast_slider.value() / 100.0
        self.gamma = self.gamma_slider.value() / 100.0

        if self.image_data is not None:
            current_slice = self.image_data[self.current_slice]
            if len(current_slice.shape) == 2:
                data_min = float(np.min(current_slice))
                data_max = float(np.max(current_slice))
                data_range = data_max - data_min

                self.vmin = (
                    None
                    if self.window_min_slider.value() == 0
                    else data_min
                    + (self.window_min_slider.value() / 100.0) * data_range
                )
                self.vmax = (
                    None
                    if self.window_max_slider.value() == 100
                    else data_min
                    + (self.window_max_slider.value() / 100.0) * data_range
                )

    def update_curve_params(self):
        self.curve_x_pos = self.curve_x_pos_slider.value() / 100.0
        self.curve_x_neg = self.curve_x_neg_slider.value() / 100.0
        self.curve_y_pos = self.curve_y_pos_slider.value() / 100.0
        self.curve_y_neg = self.curve_y_neg_slider.value() / 100.0
        self.curve_z_pos = self.curve_z_pos_slider.value() / 100.0
        self.curve_z_neg = self.curve_z_neg_slider.value() / 100.0

    def update_crop_params(self):
        self.crop_top = self.crop_top_slider.value()
        self.crop_bottom = self.crop_bottom_slider.value()
        self.crop_left = self.crop_left_slider.value()
        self.crop_right = self.crop_right_slider.value()

    def _volume_dims(self):
        if self.image_data is not None:
            return (
                self.image_data.shape[0],
                self.image_data.shape[1],
                self.image_data.shape[2],
            )
        return (
            self.depth_slider.value(),
            self.height_slider.value(),
            self.width_slider.value(),
        )

    def setup_corner_slider_ranges(self):
        d, h, w = self._volume_dims()

        ext = max(w, h, d)
        self.corner_x_slider.setRange(-ext, (w - 1) + ext)
        self.corner_y_slider.setRange(-ext, (h - 1) + ext)
        self.corner_z_slider.setRange(-ext, (d - 1) + ext)
        self.sync_corner_sliders()

    def sync_corner_sliders(self):
        if self.corner_positions is None:
            return
        idx = self.selected_corner_index
        x, y, z = self.corner_positions[idx].astype(int)

        for slider, val in [
            (self.corner_x_slider, x),
            (self.corner_y_slider, y),
            (self.corner_z_slider, z),
        ]:
            slider.blockSignals(True)
            slider.setValue(int(val))
            slider.blockSignals(False)
            if hasattr(slider, "label_widget"):
                slider.label_widget.setText(str(int(val)))

    def update_corner_combo_items(self):
        self.corner_combo.blockSignals(True)
        self.corner_combo.clear()
        if self.use_corner_symmetry:
            self.corner_combo.addItems(
                ["C000 (X-,Y-,Z-) [Master]", "C010 (X-,Y+,Z-) [Master]"]
            )
            idx = 0 if self.selected_corner_index == 0 else 1
            self.corner_combo.setCurrentIndex(idx)
        else:
            self.corner_combo.addItems(
                [
                    "C000 (X-,Y-,Z-)",
                    "C100 (X+,Y-,Z-)",
                    "C010 (X-,Y+,Z-)",
                    "C110 (X+,Y+,Z-)",
                    "C001 (X-,Y-,Z+)",
                    "C101 (X+,Y-,Z+)",
                    "C011 (X-,Y+,Z+)",
                    "C111 (X+,Y+,Z+)",
                ]
            )
            self.corner_combo.setCurrentIndex(self.selected_corner_index)
        self.corner_combo.blockSignals(False)
