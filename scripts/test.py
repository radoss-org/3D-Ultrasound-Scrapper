import argparse
import base64
import json
import os
import sys
from io import BytesIO

import numpy as np
import pydicom
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

DEFAULT_IMAGE_FILE = "sample_image.raw"


class RawImageGuessQt(QMainWindow):
    MIN_OB_TAG_SIZE = 100

    def __init__(self, config_file=None):
        super().__init__()
        self.image_data = None
        self.current_slice = 0

        # View
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.mouse_pressed = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        # Enhancement
        self.brightness = 0.0
        self.contrast = 1.0
        self.gamma = 1.0
        self.vmin = None
        self.vmax = None

        # Corner warp (8 corners: C000-C111)
        self.corner_positions = None
        self.selected_corner_index = 0
        self.use_corner_symmetry = True
        self.show_corner_notes = True

        # Curve deformation
        self.curve_x_pos = 0.0
        self.curve_x_neg = 0.0
        self.curve_y_pos = 0.0
        self.curve_y_neg = 0.0
        self.curve_z_pos = 0.0
        self.curve_z_neg = 0.0

        # Crop
        self.crop_top = 0
        self.crop_bottom = 0
        self.crop_left = 0
        self.crop_right = 0
        self.dicom_ds = None
        self.ob_tags = []
        self.dicom_selected_tag = None

        # Orientation history
        self.orientation_ops = []

        # Header marker
        self.header_end_marker = "[SCALPEL]\ncount=0"

        # Header offset
        self.use_header_offset = False
        self.header_offset = 0

        self.init_ui()

        if config_file and os.path.exists(config_file):
            self.load_config(config_file)

    def init_ui(self):
        self.setWindowTitle("3D Ultrasound Scrapper")
        self.setGeometry(100, 100, 1400, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.addWidget(self.create_controls_panel(), 1)
        main_layout.addWidget(self.create_image_panel(), 2)

        self.load_default_parameters()

    def create_controls_panel(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # File Selection
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)
        self.file_path_label = QLabel(f"File: {DEFAULT_IMAGE_FILE}")
        file_layout.addWidget(self.file_path_label)

        file_buttons = QHBoxLayout()
        browse_btn = QPushButton("Browse File")
        browse_btn.clicked.connect(self.browse_file)
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        file_buttons.addWidget(browse_btn)
        file_buttons.addWidget(load_btn)
        file_layout.addLayout(file_buttons)
        layout.addWidget(file_group)

        # DICOM OB Tags
        dicom_group = QGroupBox("DICOM OB Tags")
        dicom_layout = QVBoxLayout(dicom_group)
        self.dicom_tags_label = QLabel("No DICOM tags")
        dicom_layout.addWidget(self.dicom_tags_label)
        self.dicom_tag_combo = QComboBox()
        self.dicom_tag_combo.currentIndexChanged.connect(
            self.on_dicom_tag_changed
        )
        dicom_layout.addWidget(self.dicom_tag_combo)
        layout.addWidget(dicom_group)

        # Image Parameters
        params_group = QGroupBox("Image Parameters")
        params_layout = QGridLayout(params_group)

        params_layout.addWidget(QLabel("Pixel Type:"), 0, 0)
        self.pixel_type_combo = QComboBox()
        self.pixel_type_combo.addItems(
            [
                "8 bit unsigned",
                "8 bit signed",
                "16 bit unsigned",
                "16 bit signed",
                "float",
                "double",
                "24 bit RGB",
            ]
        )
        self.pixel_type_combo.currentTextChanged.connect(
            self.on_file_param_changed
        )
        params_layout.addWidget(self.pixel_type_combo, 0, 1, 1, 3)

        params_layout.addWidget(QLabel("Endianness:"), 1, 0)
        self.endianness_combo = QComboBox()
        self.endianness_combo.addItems(["Little endian", "Big endian"])
        self.endianness_combo.currentTextChanged.connect(
            self.on_file_param_changed
        )
        params_layout.addWidget(self.endianness_combo, 1, 1, 1, 3)

        self.header_size_slider = self.add_param_row(
            params_layout, 2, "Header Size:", 0, 1000000, 224999
        )

        # Header offset controls
        params_layout.addWidget(QLabel("Header Offset:"), 3, 0)
        self.header_offset_checkbox = QCheckBox("Enable offset")
        self.header_offset_checkbox.setChecked(False)
        self.header_offset_checkbox.toggled.connect(
            self.on_header_offset_toggled
        )
        params_layout.addWidget(self.header_offset_checkbox, 3, 1)

        self.header_offset_spin = QSpinBox()
        self.header_offset_spin.setRange(-1000000, 1000000)
        self.header_offset_spin.setValue(0)
        self.header_offset_spin.setSingleStep(1)
        self.header_offset_spin.setEnabled(False)
        self.header_offset_spin.valueChanged.connect(
            self.on_header_offset_changed
        )
        params_layout.addWidget(self.header_offset_spin, 3, 2)

        self.header_offset_label = QLabel("0")
        params_layout.addWidget(self.header_offset_label, 3, 3)

        self.footer_size_slider = self.add_param_row(
            params_layout, 4, "Footer Size:", 0, 1000000, 0
        )
        self.width_slider = self.add_param_row(
            params_layout, 5, "Width:", 1, 2000, 424
        )
        self.height_slider = self.add_param_row(
            params_layout, 6, "Height:", 1, 2000, 127
        )
        self.row_stride_slider = self.add_param_row(
            params_layout, 7, "Row Stride:", 0, 1000, 0
        )
        self.row_padding_slider = self.add_param_row(
            params_layout, 8, "Row Padding:", 0, 1000, 0
        )
        self.depth_slider = self.add_param_row(
            params_layout, 9, "Depth:", 1, 600, 317
        )
        self.skip_slices_slider = self.add_param_row(
            params_layout, 10, "Skip Slices:", 0, 100, 0
        )
        self.slice_stride_slider = self.add_param_row(
            params_layout, 11, "Slice Padding:", 0, 1000, 432
        )

        params_layout.addWidget(QLabel("Header End Marker:"), 12, 0)
        self.header_marker_edit = QTextEdit()
        self.header_marker_edit.setMaximumHeight(60)
        self.header_marker_edit.setPlainText(self.header_end_marker)
        self.header_marker_edit.textChanged.connect(
            self.on_header_marker_changed
        )
        params_layout.addWidget(self.header_marker_edit, 12, 1, 1, 3)

        layout.addWidget(params_group)

        # Spacing
        spacing_group = QGroupBox("Image Spacing")
        spacing_layout = QGridLayout(spacing_group)
        self.spacing_x_spin = self.add_spacing_row(
            spacing_layout, 0, "X Spacing:", 1.0
        )
        self.spacing_y_spin = self.add_spacing_row(
            spacing_layout, 1, "Y Spacing:", 2.6
        )
        self.spacing_z_spin = self.add_spacing_row(
            spacing_layout, 2, "Z Spacing:", 1.0
        )
        layout.addWidget(spacing_group)

        # Orientation
        orientation_group = QGroupBox("Orientation")
        orientation_layout = QGridLayout(orientation_group)
        self.add_orientation_buttons(orientation_layout)
        layout.addWidget(orientation_group)

        # Corner Deform
        corner_group = QGroupBox("3D Corner Deform")
        corner_layout = QGridLayout(corner_group)

        self.corner_notes_checkbox = QCheckBox("Show corner notes")
        self.corner_notes_checkbox.setChecked(True)
        self.corner_notes_checkbox.toggled.connect(
            self.on_visual_param_changed
        )
        corner_layout.addWidget(self.corner_notes_checkbox, 0, 0, 1, 2)

        self.corner_symmetry_checkbox = QCheckBox(
            "Use symmetry (C000+C010 control all)"
        )
        self.corner_symmetry_checkbox.setChecked(True)
        self.corner_symmetry_checkbox.toggled.connect(
            self.on_corner_symmetry_toggled
        )
        corner_layout.addWidget(self.corner_symmetry_checkbox, 0, 2, 1, 2)

        corner_layout.addWidget(QLabel("Corner:"), 1, 0)
        self.corner_combo = QComboBox()
        self.corner_combo.currentIndexChanged.connect(
            self.on_corner_selection_changed
        )
        corner_layout.addWidget(self.corner_combo, 1, 1, 1, 3)

        self.corner_x_slider = self.add_param_row(
            corner_layout, 2, "Corner X:", -1000, 4000, 0
        )
        self.corner_y_slider = self.add_param_row(
            corner_layout, 3, "Corner Y:", -1000, 4000, 0
        )
        self.corner_z_slider = self.add_param_row(
            corner_layout, 4, "Corner Z:", -1000, 4000, 0
        )

        reset_corners_btn = QPushButton("Reset corners")
        reset_corners_btn.clicked.connect(self.reset_corners)
        corner_layout.addWidget(reset_corners_btn, 5, 0, 1, 2)

        layout.addWidget(corner_group)

        # Curve Deformation
        curve_group = QGroupBox("Curve Deformation")
        curve_layout = QGridLayout(curve_group)
        self.curve_x_pos_slider = self.add_param_row(
            curve_layout, 0, "X+ Curve:", -100, 100, 0
        )
        self.curve_x_neg_slider = self.add_param_row(
            curve_layout, 1, "X- Curve:", -100, 100, 0
        )
        self.curve_y_pos_slider = self.add_param_row(
            curve_layout, 2, "Y+ Curve:", -100, 100, 0
        )
        self.curve_y_neg_slider = self.add_param_row(
            curve_layout, 3, "Y- Curve:", -100, 100, 0
        )
        self.curve_z_pos_slider = self.add_param_row(
            curve_layout, 4, "Z+ Curve:", -100, 100, 0
        )
        self.curve_z_neg_slider = self.add_param_row(
            curve_layout, 5, "Z- Curve:", -100, 100, 0
        )
        reset_curves_btn = QPushButton("Reset Curves")
        reset_curves_btn.clicked.connect(self.reset_curves)
        curve_layout.addWidget(reset_curves_btn, 6, 0, 1, 2)
        layout.addWidget(curve_group)

        # Crop
        crop_group = QGroupBox("Crop")
        crop_layout = QGridLayout(crop_group)
        self.crop_top_slider = self.add_param_row(
            crop_layout, 0, "Crop Top:", 0, 500, 0
        )
        self.crop_bottom_slider = self.add_param_row(
            crop_layout, 1, "Crop Bottom:", 0, 500, 0
        )
        self.crop_left_slider = self.add_param_row(
            crop_layout, 2, "Crop Left:", 0, 500, 0
        )
        self.crop_right_slider = self.add_param_row(
            crop_layout, 3, "Crop Right:", 0, 500, 0
        )
        reset_crop_btn = QPushButton("Reset Crop")
        reset_crop_btn.clicked.connect(self.reset_crop)
        crop_layout.addWidget(reset_crop_btn, 4, 0, 1, 2)
        apply_crop_btn = QPushButton("Apply Crop (Permanent)")
        apply_crop_btn.clicked.connect(self.apply_crop)
        crop_layout.addWidget(apply_crop_btn, 4, 2, 1, 2)
        layout.addWidget(crop_group)

        # Enhancement
        enhancement_group = QGroupBox("Image Enhancement")
        enhancement_layout = QGridLayout(enhancement_group)
        self.brightness_slider = self.add_param_row(
            enhancement_layout, 0, "Brightness:", -100, 100, 0
        )
        self.contrast_slider = self.add_param_row(
            enhancement_layout, 1, "Contrast:", 10, 300, 100
        )
        self.gamma_slider = self.add_param_row(
            enhancement_layout, 2, "Gamma:", 10, 300, 100
        )
        self.window_min_slider = self.add_param_row(
            enhancement_layout, 3, "Window Min:", 0, 100, 0
        )
        self.window_max_slider = self.add_param_row(
            enhancement_layout, 4, "Window Max:", 0, 100, 100
        )
        reset_enhancement_btn = QPushButton("Reset Enhancement")
        reset_enhancement_btn.clicked.connect(self.reset_enhancement)
        enhancement_layout.addWidget(reset_enhancement_btn, 5, 0, 1, 4)
        layout.addWidget(enhancement_group)

        # Actions
        self.generate_header_button = QPushButton("Generate NRRD Header")
        self.generate_header_button.clicked.connect(self.generate_nrrd_header)
        layout.addWidget(self.generate_header_button)

        self.save_nrrd_button = QPushButton("Save as NRRD")
        self.save_nrrd_button.clicked.connect(self.save_as_nrrd)
        layout.addWidget(self.save_nrrd_button)

        # Config
        config_group = QGroupBox("Configuration")
        config_layout = QHBoxLayout(config_group)
        save_config_btn = QPushButton("Save Config")
        save_config_btn.clicked.connect(self.save_config)
        config_layout.addWidget(save_config_btn)
        load_config_btn = QPushButton("Load Config")
        load_config_btn.clicked.connect(self.load_config_dialog)
        config_layout.addWidget(load_config_btn)
        layout.addWidget(config_group)

        # Status
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        layout.addWidget(QLabel("Status:"))
        layout.addWidget(self.status_text)

        scroll_area.setWidget(panel)
        return scroll_area

    def add_param_row(self, layout, row, label, min_val, max_val, default):
        layout.addWidget(QLabel(label), row, 0)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default)

        if label in [
            "Header Size:",
            "Footer Size:",
            "Width:",
            "Height:",
            "Row Stride:",
            "Row Padding:",
            "Depth:",
            "Skip Slices:",
            "Slice Padding:",
        ]:
            slider.valueChanged.connect(self.on_file_param_changed)
        elif label in ["Corner X:", "Corner Y:", "Corner Z:"]:
            slider.valueChanged.connect(self.on_corner_slider_changed)
        else:
            slider.valueChanged.connect(self.on_visual_param_changed)

        layout.addWidget(slider, row, 1)

        label_widget = QLabel(str(default))
        slider.label_widget = label_widget
        layout.addWidget(label_widget, row, 2)

        controls = QVBoxLayout()
        plus_btn = QPushButton("+")
        plus_btn.setMaximumSize(30, 20)
        plus_btn.clicked.connect(lambda: self.adjust_slider(slider, 1))
        minus_btn = QPushButton("-")
        minus_btn.setMaximumSize(30, 20)
        minus_btn.clicked.connect(lambda: self.adjust_slider(slider, -1))
        controls.addWidget(plus_btn)
        controls.addWidget(minus_btn)
        layout.addLayout(controls, row, 3)

        return slider

    def add_spacing_row(self, layout, row, label, default):
        layout.addWidget(QLabel(label), row, 0)
        spin = QDoubleSpinBox()
        spin.setRange(0.01, 10.0)
        spin.setValue(default)
        spin.setSingleStep(0.1)
        spin.valueChanged.connect(self.on_visual_param_changed)
        layout.addWidget(spin, row, 1)
        return spin

    def add_orientation_buttons(self, layout):
        buttons = [
            (0, 0, "Flip X", lambda: self.flip_axis("x")),
            (0, 1, "Flip Y", lambda: self.flip_axis("y")),
            (0, 2, "Flip Z", lambda: self.flip_axis("z")),
            (1, 0, "Rot X +90", lambda: self.rotate_axis("x", 1)),
            (1, 1, "Rot X -90", lambda: self.rotate_axis("x", -1)),
            (2, 0, "Rot Y +90", lambda: self.rotate_axis("y", 1)),
            (2, 1, "Rot Y -90", lambda: self.rotate_axis("y", -1)),
            (3, 0, "Rot Z +90", lambda: self.rotate_axis("z", 1)),
            (3, 1, "Rot Z -90", lambda: self.rotate_axis("z", -1)),
        ]
        for row, col, text, callback in buttons:
            btn = QPushButton(text)
            btn.clicked.connect(callback)
            layout.addWidget(btn, row, col)

    def create_image_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        slice_layout = QHBoxLayout()
        slice_layout.addWidget(QLabel("Slice:"))
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.valueChanged.connect(self.update_slice_display)
        slice_layout.addWidget(self.slice_slider)
        self.slice_label = QLabel("0/0")
        slice_layout.addWidget(self.slice_label)
        layout.addLayout(slice_layout)

        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom:"))
        zoom_out_btn = QPushButton("Zoom Out")
        zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(zoom_out_btn)
        zoom_reset_btn = QPushButton("Reset View")
        zoom_reset_btn.clicked.connect(self.reset_zoom)
        zoom_layout.addWidget(zoom_reset_btn)
        zoom_in_btn = QPushButton("Zoom In")
        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(zoom_in_btn)
        self.zoom_label = QLabel("100%")
        zoom_layout.addWidget(self.zoom_label)
        zoom_layout.addStretch()
        layout.addLayout(zoom_layout)

        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("3D Ultrasound Scrapper Preview")
        self.ax.axis("off")

        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)

        return panel

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

    def get_pixel_info(self):
        pixel_type = self.pixel_type_combo.currentText()
        endianness = self.endianness_combo.currentText()

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

    def load_image(self):
        if not os.path.exists(self.current_file):
            self.status_text.append(
                f"Error: File {self.current_file} not found"
            )
            return

        try:
            # Auto-detect header using marker
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
                    header_size_from_marker = self.find_header_end(
                        raw_bytes_for_search
                    )

            current_header = self.header_size_slider.value()
            if header_size_from_marker > 0:
                self.header_size_slider.blockSignals(True)
                self.header_size_slider.setValue(header_size_from_marker)
                self.header_size_slider.blockSignals(False)
                self.update_all_labels()
                self.status_text.append(
                    f"Auto-detected header size: {header_size_from_marker} bytes"
                )

            width = self.width_slider.value()
            height = self.height_slider.value()
            depth = self.depth_slider.value()
            header_size = self.header_size_slider.value()
            footer_size = self.footer_size_slider.value()
            skip_slices = self.skip_slices_slider.value()
            row_stride = self.row_stride_slider.value()
            row_padding = self.row_padding_slider.value()
            slice_stride = self.slice_stride_slider.value()

            # Apply header offset if enabled
            if self.use_header_offset:
                header_size += self.header_offset
                self.status_text.append(
                    f"Applied header offset: {self.header_offset}, "
                    f"final header size: {header_size}"
                )

            dtype, byte_size, components = self.get_pixel_info()

            row_data_size = width * byte_size * components
            effective_row_stride = (
                row_stride if row_stride > 0 else row_data_size + row_padding
            )

            slice_data_size = height * effective_row_stride
            effective_slice_stride = slice_data_size + slice_stride

            total_header_size = (
                header_size + skip_slices * effective_slice_stride
            )

            raw_stream = None
            raw_source_name = self.current_file

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
                raw_stream = BytesIO(raw_bytes)
                raw_size = len(raw_bytes)
                raw_source_name = (
                    f"DICOM tag (0x{self.dicom_selected_tag.group:04X}"
                    f"{self.dicom_selected_tag.element:04X})"
                )
            else:
                raw_size = os.path.getsize(self.current_file)
                raw_stream = open(self.current_file, "rb")

            available_data_size = raw_size - total_header_size - footer_size
            if available_data_size <= 0:
                self.status_text.append(
                    "Error: Not enough data after header and footer"
                )
                if hasattr(raw_stream, "close"):
                    raw_stream.close()
                return

            max_slices = int(
                (available_data_size + slice_stride) / effective_slice_stride
            )
            final_depth = min(depth, max_slices)

            if final_depth <= 0:
                self.status_text.append(
                    "Error: Not enough data for specified parameters"
                )
                if hasattr(raw_stream, "close"):
                    raw_stream.close()
                return

            image_slices = []
            with raw_stream as f:
                for slice_idx in range(final_depth):
                    slice_position = (
                        total_header_size + slice_idx * effective_slice_stride
                    )

                    if effective_row_stride == row_data_size:
                        f.seek(slice_position)
                        slice_bytes = f.read(height * row_data_size)
                        if len(slice_bytes) < height * row_data_size:
                            break
                        slice_data = np.frombuffer(slice_bytes, dtype=dtype)
                    else:
                        row_data_list = []
                        for row_idx in range(height):
                            row_position = (
                                slice_position + row_idx * effective_row_stride
                            )
                            f.seek(row_position)
                            row_bytes = f.read(row_data_size)
                            if len(row_bytes) < row_data_size:
                                break
                            row_data_list.append(
                                np.frombuffer(row_bytes, dtype=dtype)
                            )
                        if len(row_data_list) != height:
                            break
                        slice_data = np.concatenate(row_data_list)

                    if components == 1:
                        slice_data = slice_data.reshape((height, width))
                    else:
                        slice_data = slice_data.reshape(
                            (height, width, components)
                        )

                    image_slices.append(slice_data)

            if not image_slices:
                self.status_text.append("Error: No valid slices could be read")
                return

            self.image_data = np.array(image_slices)
            final_depth = len(image_slices)

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
                f"{width}x{height}x{final_depth} (header={header_size})"
            )

        except Exception as e:
            self.status_text.append(f"Error loading image: {str(e)}")

    def apply_orientation_ops(self):
        if not self.orientation_ops or self.image_data is None:
            return

        for op in self.orientation_ops:
            if op[0] == "flip":
                _, axis = op
                axis_map = {"z": 0, "y": 1, "x": 2}
                self.image_data = np.flip(self.image_data, axis=axis_map[axis])
                if axis == "z":
                    self.current_slice = (
                        self.image_data.shape[0] - 1 - self.current_slice
                    )
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
        slice_data = self.apply_crop_to_slice(slice_data)

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

        D, H, W = self.image_data.shape[:3]
        t = 0.0 if D <= 1 else self.current_slice / (D - 1)

        cp = self.corner_positions
        corners = ["TL", "TR", "BR", "BL"]

        H_display = H - self.crop_top - self.crop_bottom
        W_display = W - self.crop_left - self.crop_right

        positions = [
            (4, 12, "left", "top"),
            (W_display - 4, 12, "right", "top"),
            (W_display - 4, H_display - 4, "right", "bottom"),
            (4, H_display - 4, "left", "bottom"),
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
        enhanced = image.astype(np.float64)
        enhanced = enhanced + self.brightness

        if self.contrast != 1.0:
            middle = np.mean(enhanced)
            enhanced = middle + (enhanced - middle) * self.contrast

        if self.gamma != 1.0:
            min_val = np.min(enhanced)
            max_val = np.max(enhanced)
            if max_val > min_val:
                normalized = (enhanced - min_val) / (max_val - min_val)
                gamma_corrected = np.power(normalized, self.gamma)
                enhanced = min_val + gamma_corrected * (max_val - min_val)

        max_val = (
            np.iinfo(image.dtype).max
            if np.issubdtype(image.dtype, np.integer)
            else 1.0
        )
        enhanced = np.clip(enhanced, 0, max_val)

        return enhanced.astype(image.dtype)

    def apply_zoom_and_pan(self):
        if self.image_data is None:
            return

        slice_data = self.warp_slice(self.current_slice)
        slice_data = self.apply_crop_to_slice(slice_data)
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
        if self.image_data is not None:
            d, h, w = self.image_data.shape[:3]
        else:
            w = self.width_slider.value()
            h = self.height_slider.value()
            d = self.depth_slider.value()

        self.corner_positions = np.zeros((8, 3), dtype=np.float64)
        for idx in range(8):
            ix, iy, iz = (idx >> 0) & 1, (idx >> 1) & 1, (idx >> 2) & 1
            self.corner_positions[idx] = [
                ix * (w - 1),
                iy * (h - 1),
                iz * (d - 1),
            ]

        self.setup_corner_slider_ranges()
        self.update_slice_display()

    def setup_corner_slider_ranges(self):
        if self.image_data is not None:
            d, h, w = self.image_data.shape[:3]
        else:
            w = self.width_slider.value()
            h = self.height_slider.value()
            d = self.depth_slider.value()

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

        self.corner_positions[1] = [2 * center[0] - c000[0], c000[1], c000[2]]
        self.corner_positions[3] = [2 * center[0] - c010[0], c010[1], c010[2]]
        self.corner_positions[4] = [c000[0], c000[1], 2 * center[2] - c000[2]]
        self.corner_positions[5] = [
            self.corner_positions[1][0],
            self.corner_positions[1][1],
            2 * center[2] - self.corner_positions[1][2],
        ]
        self.corner_positions[6] = [c010[0], c010[1], 2 * center[2] - c010[2]]
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

    def warp_slice(self, z_idx):
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

        X, Y, Z = self.apply_curve_deformation(X, Y, Z, D, H, W)

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

    def apply_curve_deformation(self, X, Y, Z, D, H, W):
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

        if abs(self.curve_x_pos) > 1e-6 or abs(self.curve_x_neg) > 1e-6:
            curve_x = np.where(
                x_norm >= 0.5,
                self.curve_x_pos * (x_norm - 0.5) * 2.0,
                self.curve_x_neg * (0.5 - x_norm) * 2.0,
            )
            Y += curve_x * (H - 1) * 0.5 * np.sin(np.pi * y_norm)
            Z += curve_x * (D - 1) * 0.5 * np.sin(np.pi * z_norm)

        if abs(self.curve_y_pos) > 1e-6 or abs(self.curve_y_neg) > 1e-6:
            curve_y = np.where(
                y_norm >= 0.5,
                self.curve_y_pos * (y_norm - 0.5) * 2.0,
                self.curve_y_neg * (0.5 - y_norm) * 2.0,
            )
            X += curve_y * (W - 1) * 0.5 * np.sin(np.pi * x_norm)
            Z += curve_y * (D - 1) * 0.5 * np.sin(np.pi * z_norm)

        if abs(self.curve_z_pos) > 1e-6 or abs(self.curve_z_neg) > 1e-6:
            curve_z = np.where(
                z_norm >= 0.5,
                self.curve_z_pos * (z_norm - 0.5) * 2.0,
                self.curve_z_neg * (0.5 - z_norm) * 2.0,
            )
            X += curve_z * (W - 1) * 0.5 * np.sin(np.pi * x_norm)
            Y += curve_z * (H - 1) * 0.5 * np.sin(np.pi * y_norm)

        return X, Y, Z

    def apply_crop_to_slice(self, slice_data):
        if not (
            self.crop_top
            or self.crop_bottom
            or self.crop_left
            or self.crop_right
        ):
            return slice_data

        H, W = slice_data.shape[:2]

        if (
            self.crop_top + self.crop_bottom >= H
            or self.crop_left + self.crop_right >= W
        ):
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

        H = self.image_data.shape[1]
        W = self.image_data.shape[2]

        if self.crop_top + self.crop_bottom >= H:
            QMessageBox.warning(
                self,
                "Invalid Crop",
                "Vertical crop values exceed image height",
            )
            return
        if self.crop_left + self.crop_right >= W:
            QMessageBox.warning(
                self,
                "Invalid Crop",
                "Horizontal crop values exceed image width",
            )
            return

        if self.image_data.ndim == 3:
            self.image_data = self.image_data[
                :,
                self.crop_top : H - self.crop_bottom,
                self.crop_left : W - self.crop_right,
            ]
        else:
            self.image_data = self.image_data[
                :,
                self.crop_top : H - self.crop_bottom,
                self.crop_left : W - self.crop_right,
                :,
            ]

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
            spacing_x = self.spacing_x_spin.value()
            spacing_y = self.spacing_y_spin.value()
            spacing_z = self.spacing_z_spin.value()
            pixel_type = self.pixel_type_combo.currentText()
            endianness = self.endianness_combo.currentText()

            type_map = {
                "8 bit unsigned": "uchar",
                "8 bit signed": "signed char",
                "16 bit unsigned": "ushort",
                "16 bit signed": "short",
                "float": "float",
                "double": "double",
                "24 bit RGB": "uchar",
            }

            base_name = os.path.splitext(self.current_file)[0]
            header_file = base_name + ".nhdr"

            dtype, byte_size, components = self.get_pixel_info()

            with open(header_file, "w") as f:
                f.write("NRRD0004\n")
                f.write(f"type: {type_map[pixel_type]}\n")
                f.write("space: left-posterior-superior\n")

                if components > 1:
                    f.write("dimension: 4\n")
                    f.write(f"sizes: {components} {width} {height} {depth}\n")
                    f.write(
                        f"space directions: none ({spacing_x},0,0) "
                        f"(0,{spacing_y},0) (0,0,{spacing_z})\n"
                    )
                    f.write("kinds: vector domain domain domain\n")
                else:
                    f.write("dimension: 3\n")
                    f.write(f"sizes: {width} {height} {depth}\n")
                    f.write(
                        f"space directions: ({spacing_x},0,0) "
                        f"(0,{spacing_y},0) (0,0,{spacing_z})\n"
                    )
                    f.write("kinds: domain domain domain\n")

                f.write(
                    f"endian: {'little' if endianness == 'Little endian' else 'big'}\n"
                )
                f.write("encoding: raw\n")
                f.write("space origin: (0,0,0)\n")
                if header_size > 0:
                    f.write(f"byte skip: {header_size}\n")
                f.write(f"data file: {os.path.basename(self.current_file)}\n")

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
            spacing_x = self.spacing_x_spin.value()
            spacing_y = self.spacing_y_spin.value()
            spacing_z = self.spacing_z_spin.value()
            pixel_type = self.pixel_type_combo.currentText()
            endianness = self.endianness_combo.currentText()

            original_header_bytes = self.get_original_header_bytes()
            original_header_b64 = base64.b64encode(
                original_header_bytes
            ).decode("ascii")

            if export_data.ndim == 4:
                depth, height, width, components = export_data.shape
            else:
                depth, height, width = export_data.shape
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

            with open(file_path, "w", newline="\n") as f:
                f.write("NRRD0004\n")
                f.write(f"type: {type_map[pixel_type]}\n")
                f.write("space: left-posterior-superior\n")

                if components > 1:
                    f.write("dimension: 4\n")
                    f.write(f"sizes: {components} {width} {height} {depth}\n")
                    f.write(
                        f"space directions: none ({spacing_x},0,0) "
                        f"(0,{spacing_y},0) (0,0,{spacing_z})\n"
                    )
                    f.write("kinds: vector domain domain domain\n")
                else:
                    f.write("dimension: 3\n")
                    f.write(f"sizes: {width} {height} {depth}\n")
                    f.write(
                        f"space directions: ({spacing_x},0,0) "
                        f"(0,{spacing_y},0) (0,0,{spacing_z})\n"
                    )
                    f.write("kinds: domain domain domain\n")

                f.write(
                    f"endian: {'little' if endianness == 'Little endian' else 'big'}\n"
                )
                f.write("encoding: raw\n")
                f.write("space origin: (0,0,0)\n")

                if original_header_bytes:
                    f.write(f"raw_header_size:={len(original_header_bytes)}\n")
                    f.write(f"raw_header_base64:={original_header_b64}\n")
                    f.write(
                        f"raw_header_source:={os.path.basename(self.current_file)}\n"
                    )

                f.write("\n")

            base_dtype_map = {
                "8 bit unsigned": np.uint8,
                "8 bit signed": np.int8,
                "16 bit unsigned": np.uint16,
                "16 bit signed": np.int16,
                "float": np.float32,
                "double": np.float64,
                "24 bit RGB": np.uint8,
            }

            data_to_save = export_data.astype(
                base_dtype_map[pixel_type], copy=False
            )
            if endianness == "Big endian" and data_to_save.dtype.itemsize > 1:
                data_to_save = data_to_save.byteswap().newbyteorder()

            if components > 1:
                data_to_save = np.moveaxis(data_to_save, -1, 0)

            with open(file_path, "ab") as f:
                f.write(data_to_save.tobytes())

            self.status_text.append(f"Saved NRRD file: {file_path}")
            QMessageBox.information(
                self, "Success", f"NRRD file saved: {file_path}"
            )

        except Exception as e:
            self.status_text.append(f"Error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed: {str(e)}")

    def build_export_volume(self):
        needs_warp = not self.are_corners_identity()
        needs_crop = (
            self.crop_top
            or self.crop_bottom
            or self.crop_left
            or self.crop_right
        )

        if not needs_warp and not needs_crop:
            return self.image_data

        D = self.image_data.shape[0]
        slices = []
        for z in range(D):
            slice_data = self.warp_slice(z)
            slice_data = self.apply_crop_to_slice(slice_data)
            slices.append(slice_data)
        return np.stack(slices, axis=0)

    def are_corners_identity(self):
        if self.image_data is not None:
            d, h, w = self.image_data.shape[:3]
        else:
            w = self.width_slider.value()
            h = self.height_slider.value()
            d = self.depth_slider.value()

        default_corners = np.zeros((8, 3), dtype=np.float64)
        for idx in range(8):
            ix, iy, iz = (idx >> 0) & 1, (idx >> 1) & 1, (idx >> 2) & 1
            default_corners[idx] = [ix * (w - 1), iy * (h - 1), iz * (d - 1)]

        return np.array_equal(self.corner_positions, default_corners)

    def get_current_config(self):
        return {
            "current_file": getattr(self, "current_file", DEFAULT_IMAGE_FILE),
            "pixel_type": self.pixel_type_combo.currentText(),
            "endianness": self.endianness_combo.currentText(),
            "header_size": self.header_size_slider.value(),
            "footer_size": self.footer_size_slider.value(),
            "width": self.width_slider.value(),
            "height": self.height_slider.value(),
            "row_stride": self.row_stride_slider.value(),
            "row_padding": self.row_padding_slider.value(),
            "depth": self.depth_slider.value(),
            "skip_slices": self.skip_slices_slider.value(),
            "slice_stride": self.slice_stride_slider.value(),
            "spacing_x": self.spacing_x_spin.value(),
            "spacing_y": self.spacing_y_spin.value(),
            "spacing_z": self.spacing_z_spin.value(),
            "brightness": self.brightness_slider.value(),
            "contrast": self.contrast_slider.value(),
            "gamma": self.gamma_slider.value(),
            "window_min": self.window_min_slider.value(),
            "window_max": self.window_max_slider.value(),
            "curve_x_pos": self.curve_x_pos_slider.value(),
            "curve_x_neg": self.curve_x_neg_slider.value(),
            "curve_y_pos": self.curve_y_pos_slider.value(),
            "curve_y_neg": self.curve_y_neg_slider.value(),
            "curve_z_pos": self.curve_z_pos_slider.value(),
            "curve_z_neg": self.curve_z_neg_slider.value(),
            "crop_top": self.crop_top_slider.value(),
            "crop_bottom": self.crop_bottom_slider.value(),
            "crop_left": self.crop_left_slider.value(),
            "crop_right": self.crop_right_slider.value(),
            "use_corner_symmetry": self.use_corner_symmetry,
            "show_corner_notes": self.show_corner_notes,
            "corner_positions": (
                self.corner_positions.tolist()
                if self.corner_positions is not None
                else None
            ),
            "selected_corner_index": self.selected_corner_index,
            "orientation_ops": self.orientation_ops,
            "header_end_marker": self.header_end_marker,
            "use_header_offset": self.use_header_offset,
            "header_offset": self.header_offset,
        }

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
            self.dicom_ds = pydicom.dcmread(self.current_file, force=True)
            self.ob_tags = []

            for elem in self.dicom_ds.iterall():
                if elem.value is None:
                    continue
                try:
                    length = len(elem.value)
                except TypeError:
                    continue
                if length <= 100:
                    continue
                self.ob_tags.append((elem.name, length, elem.tag))

            self.dicom_tag_combo.clear()
            self.dicom_tag_combo.addItem("-- No tag selected --")
            for name, length, tag in self.ob_tags:
                self.dicom_tag_combo.addItem(
                    f"{name} (0x{tag.group:04X}{tag.element:04X}) - {length} bytes"
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
        header_size = self.header_size_slider.value()
        if header_size <= 0:
            return b""

        if (
            self.dicom_selected_tag is not None
            and self.dicom_ds is not None
            and self.dicom_selected_tag in self.dicom_ds
        ):
            elem = self.dicom_ds[self.dicom_selected_tag]
            if not isinstance(elem.value, (bytes, bytearray)):
                return b""
            raw_bytes = bytes(elem.value)
            return raw_bytes[:header_size]

        if not os.path.exists(self.current_file):
            return b""

        with open(self.current_file, "rb") as f:
            return f.read(header_size)

    def find_header_end(self, data: bytes) -> int:
        if not self.header_end_marker or not data:
            return 0

        import re

        marker_text = self.header_end_marker
        marker_bytes = marker_text.encode("utf-8")

        pos_exact = data.find(marker_bytes)
        if pos_exact != -1:
            end_pos = pos_exact + len(marker_bytes)
            while end_pos < len(data) and data[end_pos : end_pos + 1] in (
                b"\n",
                b"\r",
            ):
                end_pos += 1
            return end_pos

        marker_lines = [
            line for line in marker_text.splitlines() if line.strip()
        ]
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
                while end_pos < len(data) and data[end_pos : end_pos + 1] in (
                    b"\n",
                    b"\r",
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
            while end_pos < len(data) and data[end_pos : end_pos + 1] in (
                b"\n",
                b"\r",
            ):
                end_pos += 1
            return end_pos

        return 0

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


def main():
    parser = argparse.ArgumentParser(description="3D Ultrasound Scrapper")
    parser.add_argument(
        "--config", "-c", type=str, help="Path to configuration JSON file"
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = RawImageGuessQt(config_file=args.config)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
