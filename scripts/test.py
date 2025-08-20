import os
import sys
import json
import argparse

import numpy as np
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Constants
DEFAULT_IMAGE_FILE = "sample_image.raw"


class RawImageGuessQt(QMainWindow):
    def __init__(self, config_file=None):
        super().__init__()
        self.image_data = None
        self.current_slice = 0

        # View control
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.mouse_pressed = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        # Image enhancement
        self.brightness = 0.0
        self.contrast = 1.0
        self.gamma = 1.0
        self.vmin = None
        self.vmax = None

        # Orientation history (persist flips/rotations)
        self.orientation_ops = []
        self._last_load_reason = None

        # 3D corner warp state (8 corners, each with X/Y/Z)
        # idx mapping:
        # 0: (0,0,0)=C000 (X-,Y-,Z-)
        # 1: (1,0,0)=C100 (X+,Y-,Z-)
        # 2: (0,1,0)=C010 (X-,Y+,Z-)
        # 3: (1,1,0)=C110 (X+,Y+,Z-)
        # 4: (0,0,1)=C001 (X-,Y-,Z+)
        # 5: (1,0,1)=C101 (X+,Y-,Z+)
        # 6: (0,1,1)=C011 (X-,Y+,Z+)
        # 7: (1,1,1)=C111 (X+,Y+,Z+)
        self.corner_positions = None  # shape (8, 3)
        self.selected_corner_index = 0

        # Corner symmetry controls
        self.use_corner_symmetry = True
        self.master_corners = [0, 2]  # C000 and C010 are master corners

        # Corner notes overlay
        self.show_corner_notes = True

        # Volume stretch state
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.scale_z = 1.0

        # Curve deformation state
        self.curve_x_pos = 0.0  # Curve along X axis in positive direction
        self.curve_x_neg = 0.0  # Curve along X axis in negative direction
        self.curve_y_pos = 0.0  # Curve along Y axis in positive direction
        self.curve_y_neg = 0.0  # Curve along Y axis in negative direction
        self.curve_z_pos = 0.0  # Curve along Z axis in positive direction
        self.curve_z_neg = 0.0  # Curve along Z axis in negative direction

        self.init_ui()

        # Load config file if provided
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)

    def init_ui(self):
        self.setWindowTitle("3D Ultrasound Scrapper")
        self.setGeometry(100, 100, 1400, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        controls_panel = self.create_controls_panel()
        main_layout.addWidget(controls_panel, 1)

        image_panel = self.create_image_panel()
        main_layout.addWidget(image_panel, 2)

        self.load_default_parameters()

    def create_controls_panel(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        panel = QWidget()
        layout = QVBoxLayout(panel)

        # File selection
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)

        self.file_path_label = QLabel(f"File: {DEFAULT_IMAGE_FILE}")
        file_layout.addWidget(self.file_path_label)

        file_button_layout = QHBoxLayout()
        self.browse_button = QPushButton("Browse File")
        self.browse_button.clicked.connect(self.browse_file)
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.on_manual_load_clicked)
        file_button_layout.addWidget(self.browse_button)
        file_button_layout.addWidget(self.load_button)
        file_layout.addLayout(file_button_layout)
        layout.addWidget(file_group)

        # Image parameters
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
            self.on_parameter_changed
        )
        params_layout.addWidget(self.pixel_type_combo, 0, 1, 1, 3)

        params_layout.addWidget(QLabel("Endianness:"), 1, 0)
        self.endianness_combo = QComboBox()
        self.endianness_combo.addItems(["Little endian", "Big endian"])
        self.endianness_combo.currentTextChanged.connect(
            self.on_parameter_changed
        )
        params_layout.addWidget(self.endianness_combo, 1, 1, 1, 3)

        params_layout.addWidget(QLabel("Header Size:"), 2, 0)
        self.header_size_slider = self.create_slider(
            0, 1000000, 224999, self.on_parameter_changed
        )
        params_layout.addWidget(self.header_size_slider, 2, 1)
        self.header_size_label = QLabel("0")
        params_layout.addWidget(self.header_size_label, 2, 2)
        header_controls = self.create_slider_controls(self.header_size_slider)
        params_layout.addLayout(header_controls, 2, 3)

        params_layout.addWidget(QLabel("Footer Size:"), 3, 0)
        self.footer_size_slider = self.create_slider(
            0, 1000000, 0, self.on_parameter_changed
        )
        params_layout.addWidget(self.footer_size_slider, 3, 1)
        self.footer_size_label = QLabel("0")
        params_layout.addWidget(self.footer_size_label, 3, 2)
        footer_controls = self.create_slider_controls(self.footer_size_slider)
        params_layout.addLayout(footer_controls, 3, 3)

        params_layout.addWidget(QLabel("Width:"), 4, 0)
        self.width_slider = self.create_slider(
            1, 2000, 424, self.on_parameter_changed
        )
        params_layout.addWidget(self.width_slider, 4, 1)
        self.width_label = QLabel("424")
        params_layout.addWidget(self.width_label, 4, 2)
        width_controls = self.create_slider_controls(self.width_slider)
        params_layout.addLayout(width_controls, 4, 3)

        params_layout.addWidget(QLabel("Height:"), 5, 0)
        self.height_slider = self.create_slider(
            1, 2000, 127, self.on_parameter_changed
        )
        params_layout.addWidget(self.height_slider, 5, 1)
        self.height_label = QLabel("127")
        params_layout.addWidget(self.height_label, 5, 2)
        height_controls = self.create_slider_controls(self.height_slider)
        params_layout.addLayout(height_controls, 5, 3)

        params_layout.addWidget(QLabel("Row Stride:"), 6, 0)
        self.row_stride_slider = self.create_slider(
            0, 1000, 0, self.on_parameter_changed
        )
        params_layout.addWidget(self.row_stride_slider, 6, 1)
        self.row_stride_label = QLabel("0 (auto)")
        params_layout.addWidget(self.row_stride_label, 6, 2)
        row_stride_controls = self.create_slider_controls(
            self.row_stride_slider
        )
        params_layout.addLayout(row_stride_controls, 6, 3)

        params_layout.addWidget(QLabel("Row Padding:"), 7, 0)
        self.row_padding_slider = self.create_slider(
            0, 1000, 0, self.on_parameter_changed
        )
        params_layout.addWidget(self.row_padding_slider, 7, 1)
        self.row_padding_label = QLabel("0")
        params_layout.addWidget(self.row_padding_label, 7, 2)
        row_padding_controls = self.create_slider_controls(
            self.row_padding_slider
        )
        params_layout.addLayout(row_padding_controls, 7, 3)

        params_layout.addWidget(QLabel("Depth:"), 8, 0)
        self.depth_slider = self.create_slider(
            1, 600, 317, self.on_parameter_changed
        )
        params_layout.addWidget(self.depth_slider, 8, 1)
        self.depth_label = QLabel("317")
        params_layout.addWidget(self.depth_label, 8, 2)
        depth_controls = self.create_slider_controls(self.depth_slider)
        params_layout.addLayout(depth_controls, 8, 3)

        params_layout.addWidget(QLabel("Skip Slices:"), 9, 0)
        self.skip_slices_slider = self.create_slider(
            0, 100, 0, self.on_parameter_changed
        )
        params_layout.addWidget(self.skip_slices_slider, 9, 1)
        self.skip_slices_label = QLabel("0")
        params_layout.addWidget(self.skip_slices_label, 9, 2)
        skip_controls = self.create_slider_controls(self.skip_slices_slider)
        params_layout.addLayout(skip_controls, 9, 3)

        params_layout.addWidget(QLabel("Slice Padding:"), 10, 0)
        self.slice_stride_slider = self.create_slider(
            0, 1000, 432, self.on_parameter_changed
        )
        params_layout.addWidget(self.slice_stride_slider, 10, 1)
        self.slice_stride_label = QLabel("432")
        params_layout.addWidget(self.slice_stride_label, 10, 2)
        stride_controls = self.create_slider_controls(self.slice_stride_slider)
        params_layout.addLayout(stride_controls, 10, 3)

        layout.addWidget(params_group)

        # Spacing group
        spacing_group = QGroupBox("Image Spacing")
        spacing_layout = QGridLayout(spacing_group)

        spacing_layout.addWidget(QLabel("X Spacing:"), 0, 0)
        self.spacing_x_spin = QDoubleSpinBox()
        self.spacing_x_spin.setRange(0.01, 10.0)
        self.spacing_x_spin.setValue(1.0)
        self.spacing_x_spin.setSingleStep(0.1)
        self.spacing_x_spin.valueChanged.connect(self.on_parameter_changed)
        spacing_layout.addWidget(self.spacing_x_spin, 0, 1)

        spacing_layout.addWidget(QLabel("Y Spacing:"), 1, 0)
        self.spacing_y_spin = QDoubleSpinBox()
        self.spacing_y_spin.setRange(0.01, 10.0)
        self.spacing_y_spin.setValue(2.6)
        self.spacing_y_spin.setSingleStep(0.1)
        self.spacing_y_spin.valueChanged.connect(self.on_parameter_changed)
        spacing_layout.addWidget(self.spacing_y_spin, 1, 1)

        spacing_layout.addWidget(QLabel("Z Spacing:"), 2, 0)
        self.spacing_z_spin = QDoubleSpinBox()
        self.spacing_z_spin.setRange(0.01, 10.0)
        self.spacing_z_spin.setValue(1.0)
        self.spacing_z_spin.setSingleStep(0.1)
        self.spacing_z_spin.valueChanged.connect(self.on_parameter_changed)
        spacing_layout.addWidget(self.spacing_z_spin, 2, 1)

        layout.addWidget(spacing_group)

        # Orientation group
        orientation_group = QGroupBox("Orientation")
        orientation_layout = QGridLayout(orientation_group)

        flip_x_btn = QPushButton("Flip X")
        flip_x_btn.clicked.connect(lambda: self.flip_axis("x"))
        orientation_layout.addWidget(flip_x_btn, 0, 0)

        flip_y_btn = QPushButton("Flip Y")
        flip_y_btn.clicked.connect(lambda: self.flip_axis("y"))
        orientation_layout.addWidget(flip_y_btn, 0, 1)

        flip_z_btn = QPushButton("Flip Z")
        flip_z_btn.clicked.connect(lambda: self.flip_axis("z"))
        orientation_layout.addWidget(flip_z_btn, 0, 2)

        rot_x_p_btn = QPushButton("Rot X +90")
        rot_x_p_btn.clicked.connect(lambda: self.rotate_axis("x", +1))
        orientation_layout.addWidget(rot_x_p_btn, 1, 0)

        rot_x_n_btn = QPushButton("Rot X -90")
        rot_x_n_btn.clicked.connect(lambda: self.rotate_axis("x", -1))
        orientation_layout.addWidget(rot_x_n_btn, 1, 1)

        rot_y_p_btn = QPushButton("Rot Y +90")
        rot_y_p_btn.clicked.connect(lambda: self.rotate_axis("y", +1))
        orientation_layout.addWidget(rot_y_p_btn, 2, 0)

        rot_y_n_btn = QPushButton("Rot Y -90")
        rot_y_n_btn.clicked.connect(lambda: self.rotate_axis("y", -1))
        orientation_layout.addWidget(rot_y_n_btn, 2, 1)

        rot_z_p_btn = QPushButton("Rot Z +90")
        rot_z_p_btn.clicked.connect(lambda: self.rotate_axis("z", +1))
        orientation_layout.addWidget(rot_z_p_btn, 3, 0)

        rot_z_n_btn = QPushButton("Rot Z -90")
        rot_z_n_btn.clicked.connect(lambda: self.rotate_axis("z", -1))
        orientation_layout.addWidget(rot_z_n_btn, 3, 1)

        layout.addWidget(orientation_group)

        # 3D Corner Deform group
        corner_group = QGroupBox("3D Corner Deform (Realtime 8-corner warp)")
        corner_layout = QGridLayout(corner_group)

        self.corner_notes_checkbox = QCheckBox("Show corner notes")
        self.corner_notes_checkbox.setChecked(True)
        self.corner_notes_checkbox.toggled.connect(
            self.on_corner_notes_toggled
        )
        corner_layout.addWidget(self.corner_notes_checkbox, 0, 0, 1, 2)

        self.corner_symmetry_checkbox = QCheckBox("Use symmetry (C000+C010 control all)")
        self.corner_symmetry_checkbox.setChecked(True)
        self.corner_symmetry_checkbox.toggled.connect(
            self.on_corner_symmetry_toggled
        )
        corner_layout.addWidget(self.corner_symmetry_checkbox, 0, 2, 1, 2)

        corner_layout.addWidget(QLabel("Corner:"), 1, 0)
        self.corner_combo = QComboBox()
        # Will be populated by update_corner_combo_items()
        self.corner_combo.currentIndexChanged.connect(
            self.on_corner_selection_changed
        )
        corner_layout.addWidget(self.corner_combo, 1, 1, 1, 3)

        # X slider
        corner_layout.addWidget(QLabel("Corner X:"), 2, 0)
        self.corner_x_slider = self.create_slider(
            -1000, 4000, 0, self.on_corner_slider_changed
        )
        corner_layout.addWidget(self.corner_x_slider, 2, 1)
        self.corner_x_label = QLabel("0")
        corner_layout.addWidget(self.corner_x_label, 2, 2)
        corner_layout.addLayout(
            self.create_slider_controls(self.corner_x_slider), 2, 3
        )

        # Y slider
        corner_layout.addWidget(QLabel("Corner Y:"), 3, 0)
        self.corner_y_slider = self.create_slider(
            -1000, 4000, 0, self.on_corner_slider_changed
        )
        corner_layout.addWidget(self.corner_y_slider, 3, 1)
        self.corner_y_label = QLabel("0")
        corner_layout.addWidget(self.corner_y_label, 3, 2)
        corner_layout.addLayout(
            self.create_slider_controls(self.corner_y_slider), 3, 3
        )

        # Z slider
        corner_layout.addWidget(QLabel("Corner Z:"), 4, 0)
        self.corner_z_slider = self.create_slider(
            -1000, 4000, 0, self.on_corner_slider_changed
        )
        corner_layout.addWidget(self.corner_z_slider, 4, 1)
        self.corner_z_label = QLabel("0")
        corner_layout.addWidget(self.corner_z_label, 4, 2)
        corner_layout.addLayout(
            self.create_slider_controls(self.corner_z_slider), 4, 3
        )

        reset_corners_btn = QPushButton("Reset corners")
        reset_corners_btn.clicked.connect(self.reset_corners_to_default)
        corner_layout.addWidget(reset_corners_btn, 5, 0, 1, 2)

        # Help text for symmetry
        help_label = QLabel("Symmetry: C000 & C010 control all 8 corners via center mirroring")
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #666; font-size: 9px;")
        corner_layout.addWidget(help_label, 6, 0, 1, 4)

        layout.addWidget(corner_group)

        # Stretch group
        deform_group = QGroupBox("Stretch (resample)")
        deform_layout = QGridLayout(deform_group)

        deform_layout.addWidget(QLabel("Scale X:"), 0, 0)
        self.scale_x_spin = QDoubleSpinBox()
        self.scale_x_spin.setRange(0.1, 5.0)
        self.scale_x_spin.setSingleStep(0.1)
        self.scale_x_spin.setValue(1.0)
        deform_layout.addWidget(self.scale_x_spin, 0, 1)

        deform_layout.addWidget(QLabel("Scale Y:"), 1, 0)
        self.scale_y_spin = QDoubleSpinBox()
        self.scale_y_spin.setRange(0.1, 5.0)
        self.scale_y_spin.setSingleStep(0.1)
        self.scale_y_spin.setValue(1.0)
        deform_layout.addWidget(self.scale_y_spin, 1, 1)

        deform_layout.addWidget(QLabel("Scale Z:"), 2, 0)
        self.scale_z_spin = QDoubleSpinBox()
        self.scale_z_spin.setRange(0.1, 5.0)
        self.scale_z_spin.setSingleStep(0.1)
        self.scale_z_spin.setValue(1.0)
        deform_layout.addWidget(self.scale_z_spin, 2, 1)

        self.preserve_size_checkbox = QCheckBox(
            "Preserve physical size (adjust spacings)"
        )
        self.preserve_size_checkbox.setChecked(True)
        deform_layout.addWidget(self.preserve_size_checkbox, 3, 0, 1, 2)

        apply_scale_btn = QPushButton("Apply stretch (resample)")
        apply_scale_btn.clicked.connect(self.apply_stretch)
        deform_layout.addWidget(apply_scale_btn, 4, 0, 1, 2)

        layout.addWidget(deform_group)

        # Curve Deformation group
        curve_group = QGroupBox("Curve Deformation (Realtime)")
        curve_layout = QGridLayout(curve_group)

        # X axis curves
        curve_layout.addWidget(QLabel("X+ Curve:"), 0, 0)
        self.curve_x_pos_slider = self.create_slider(
            -100, 100, 0, self.on_curve_changed
        )
        curve_layout.addWidget(self.curve_x_pos_slider, 0, 1)
        self.curve_x_pos_label = QLabel("0")
        curve_layout.addWidget(self.curve_x_pos_label, 0, 2)
        curve_layout.addLayout(
            self.create_slider_controls(self.curve_x_pos_slider), 0, 3
        )

        curve_layout.addWidget(QLabel("X- Curve:"), 1, 0)
        self.curve_x_neg_slider = self.create_slider(
            -100, 100, 0, self.on_curve_changed
        )
        curve_layout.addWidget(self.curve_x_neg_slider, 1, 1)
        self.curve_x_neg_label = QLabel("0")
        curve_layout.addWidget(self.curve_x_neg_label, 1, 2)
        curve_layout.addLayout(
            self.create_slider_controls(self.curve_x_neg_slider), 1, 3
        )

        # Y axis curves
        curve_layout.addWidget(QLabel("Y+ Curve:"), 2, 0)
        self.curve_y_pos_slider = self.create_slider(
            -100, 100, 0, self.on_curve_changed
        )
        curve_layout.addWidget(self.curve_y_pos_slider, 2, 1)
        self.curve_y_pos_label = QLabel("0")
        curve_layout.addWidget(self.curve_y_pos_label, 2, 2)
        curve_layout.addLayout(
            self.create_slider_controls(self.curve_y_pos_slider), 2, 3
        )

        curve_layout.addWidget(QLabel("Y- Curve:"), 3, 0)
        self.curve_y_neg_slider = self.create_slider(
            -100, 100, 0, self.on_curve_changed
        )
        curve_layout.addWidget(self.curve_y_neg_slider, 3, 1)
        self.curve_y_neg_label = QLabel("0")
        curve_layout.addWidget(self.curve_y_neg_label, 3, 2)
        curve_layout.addLayout(
            self.create_slider_controls(self.curve_y_neg_slider), 3, 3
        )

        # Z axis curves
        curve_layout.addWidget(QLabel("Z+ Curve:"), 4, 0)
        self.curve_z_pos_slider = self.create_slider(
            -100, 100, 0, self.on_curve_changed
        )
        curve_layout.addWidget(self.curve_z_pos_slider, 4, 1)
        self.curve_z_pos_label = QLabel("0")
        curve_layout.addWidget(self.curve_z_pos_label, 4, 2)
        curve_layout.addLayout(
            self.create_slider_controls(self.curve_z_pos_slider), 4, 3
        )

        curve_layout.addWidget(QLabel("Z- Curve:"), 5, 0)
        self.curve_z_neg_slider = self.create_slider(
            -100, 100, 0, self.on_curve_changed
        )
        curve_layout.addWidget(self.curve_z_neg_slider, 5, 1)
        self.curve_z_neg_label = QLabel("0")
        curve_layout.addWidget(self.curve_z_neg_label, 5, 2)
        curve_layout.addLayout(
            self.create_slider_controls(self.curve_z_neg_slider), 5, 3
        )

        # Reset button
        reset_curves_btn = QPushButton("Reset Curves")
        reset_curves_btn.clicked.connect(self.reset_curves)
        curve_layout.addWidget(reset_curves_btn, 6, 0, 1, 2)

        # Help text
        help_curve_label = QLabel("Curve values bend the volume. + curves forward, - curves backward.")
        help_curve_label.setWordWrap(True)
        help_curve_label.setStyleSheet("color: #666; font-size: 9px;")
        curve_layout.addWidget(help_curve_label, 7, 0, 1, 4)

        layout.addWidget(curve_group)

        # Image enhancement
        enhancement_group = QGroupBox("Image Enhancement")
        enhancement_layout = QGridLayout(enhancement_group)

        enhancement_layout.addWidget(QLabel("Brightness:"), 0, 0)
        self.brightness_slider = self.create_slider(
            -100, 100, 0, self.on_enhancement_changed
        )
        enhancement_layout.addWidget(self.brightness_slider, 0, 1)
        self.brightness_label = QLabel("0")
        enhancement_layout.addWidget(self.brightness_label, 0, 2)
        enhancement_layout.addLayout(
            self.create_slider_controls(self.brightness_slider), 0, 3
        )

        enhancement_layout.addWidget(QLabel("Contrast:"), 1, 0)
        self.contrast_slider = self.create_slider(
            10, 300, 100, self.on_enhancement_changed
        )
        enhancement_layout.addWidget(self.contrast_slider, 1, 1)
        self.contrast_label = QLabel("100%")
        enhancement_layout.addWidget(self.contrast_label, 1, 2)
        enhancement_layout.addLayout(
            self.create_slider_controls(self.contrast_slider), 1, 3
        )

        enhancement_layout.addWidget(QLabel("Gamma:"), 2, 0)
        self.gamma_slider = self.create_slider(
            10, 300, 100, self.on_enhancement_changed
        )
        enhancement_layout.addWidget(self.gamma_slider, 2, 1)
        self.gamma_label = QLabel("1.0")
        enhancement_layout.addWidget(self.gamma_label, 2, 2)
        enhancement_layout.addLayout(
            self.create_slider_controls(self.gamma_slider), 2, 3
        )

        enhancement_layout.addWidget(QLabel("Window Min:"), 3, 0)
        self.window_min_slider = self.create_slider(
            0, 100, 0, self.on_enhancement_changed
        )
        enhancement_layout.addWidget(self.window_min_slider, 3, 1)
        self.window_min_label = QLabel("Auto")
        enhancement_layout.addWidget(self.window_min_label, 3, 2)
        enhancement_layout.addLayout(
            self.create_slider_controls(self.window_min_slider), 3, 3
        )

        enhancement_layout.addWidget(QLabel("Window Max:"), 4, 0)
        self.window_max_slider = self.create_slider(
            0, 100, 100, self.on_enhancement_changed
        )
        enhancement_layout.addWidget(self.window_max_slider, 4, 1)
        self.window_max_label = QLabel("Auto")
        enhancement_layout.addWidget(self.window_max_label, 4, 2)
        enhancement_layout.addLayout(
            self.create_slider_controls(self.window_max_slider), 4, 3
        )

        reset_enhancement_btn = QPushButton("Reset Enhancement")
        reset_enhancement_btn.clicked.connect(self.reset_enhancement)
        enhancement_layout.addWidget(reset_enhancement_btn, 5, 0, 1, 4)

        layout.addWidget(enhancement_group)

        # Auto update and actions
        self.auto_update_checkbox = QCheckBox("Auto Update")
        self.auto_update_checkbox.setChecked(True)
        layout.addWidget(self.auto_update_checkbox)

        self.update_button = QPushButton("Update Image")
        self.update_button.clicked.connect(self.load_image)
        layout.addWidget(self.update_button)

        self.generate_header_button = QPushButton("Generate NRRD Header")
        self.generate_header_button.clicked.connect(self.generate_nrrd_header)
        layout.addWidget(self.generate_header_button)

        self.save_nrrd_button = QPushButton("Save as NRRD")
        self.save_nrrd_button.clicked.connect(self.save_as_nrrd)
        layout.addWidget(self.save_nrrd_button)

        # Configuration controls
        config_group = QGroupBox("Configuration")
        config_layout = QHBoxLayout(config_group)

        self.save_config_button = QPushButton("Save Config")
        self.save_config_button.clicked.connect(self.save_config)
        config_layout.addWidget(self.save_config_button)

        self.load_config_button = QPushButton("Load Config")
        self.load_config_button.clicked.connect(self.load_config_dialog)
        config_layout.addWidget(self.load_config_button)

        layout.addWidget(config_group)

        # Offset controls
        offset_group = QGroupBox("Offset Controls")
        offset_layout = QGridLayout(offset_group)

        offset_layout.addWidget(QLabel("Column:"), 0, 0)
        col_sub_btn = QPushButton("-")
        col_sub_btn.clicked.connect(
            lambda: self.offset_header("sub", "column")
        )
        col_add_btn = QPushButton("+")
        col_add_btn.clicked.connect(
            lambda: self.offset_header("add", "column")
        )
        offset_layout.addWidget(col_sub_btn, 0, 1)
        offset_layout.addWidget(col_add_btn, 0, 2)

        offset_layout.addWidget(QLabel("Row:"), 1, 0)
        row_sub_btn = QPushButton("-")
        row_sub_btn.clicked.connect(lambda: self.offset_header("sub", "row"))
        row_add_btn = QPushButton("+")
        row_add_btn.clicked.connect(lambda: self.offset_header("add", "row"))
        offset_layout.addWidget(row_sub_btn, 1, 1)
        offset_layout.addWidget(row_add_btn, 1, 2)

        offset_layout.addWidget(QLabel("Slice:"), 2, 0)
        slice_sub_btn = QPushButton("-")
        slice_sub_btn.clicked.connect(
            lambda: self.offset_header("sub", "slice")
        )
        slice_add_btn = QPushButton("+")
        slice_add_btn.clicked.connect(
            lambda: self.offset_header("add", "slice")
        )
        offset_layout.addWidget(slice_sub_btn, 2, 1)
        offset_layout.addWidget(slice_add_btn, 2, 2)

        layout.addWidget(offset_group)

        # Status text
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        layout.addWidget(QLabel("Status:"))
        layout.addWidget(self.status_text)

        scroll_area.setWidget(panel)
        return scroll_area

    def create_image_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Slice selection
        slice_layout = QHBoxLayout()
        slice_layout.addWidget(QLabel("Slice:"))
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.valueChanged.connect(self.update_slice_display)
        slice_layout.addWidget(self.slice_slider)
        self.slice_label = QLabel("0/0")
        slice_layout.addWidget(self.slice_label)
        slice_controls = self.create_slider_controls(self.slice_slider)
        slice_layout.addLayout(slice_controls)
        layout.addLayout(slice_layout)

        # Zoom controls
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

        # Image display
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("3D Ultrasound Scrapper Preview")
        self.ax.axis("off")

        # Mouse events
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)

        return panel

    def create_slider(self, min_val, max_val, default_val, callback):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        slider.valueChanged.connect(callback)
        return slider

    def create_slider_controls(self, slider):
        controls_layout = QVBoxLayout()

        plus_btn = QPushButton("+")
        plus_btn.setMaximumSize(30, 20)
        plus_btn.clicked.connect(lambda: self.adjust_slider(slider, 1))
        controls_layout.addWidget(plus_btn)

        minus_btn = QPushButton("-")
        minus_btn.setMaximumSize(30, 20)
        minus_btn.clicked.connect(lambda: self.adjust_slider(slider, -1))
        controls_layout.addWidget(minus_btn)

        return controls_layout

    def adjust_slider(self, slider, direction):
        current_value = slider.value()
        step = max(1, (slider.maximum() - slider.minimum()) // 100)
        new_value = current_value + (direction * step)
        new_value = max(slider.minimum(), min(slider.maximum(), new_value))
        slider.setValue(new_value)

    def debug_log(self, message):
        try:
            print(f"[DEBUG] {message}")
        except Exception:
            pass
        try:
            self.status_text.append(f"[DEBUG] {message}")
        except Exception:
            pass

    def load_default_parameters(self):
        self.current_file = DEFAULT_IMAGE_FILE
        self.update_slider_labels()
        self.update_enhancement_labels()
        self.update_curve_labels()
        self.reset_corners_to_default()
        self.setup_corner_slider_ranges()
        self.update_corner_combo_items()

    def update_slider_labels(self):
        self.header_size_label.setText(str(self.header_size_slider.value()))
        self.footer_size_label.setText(str(self.footer_size_slider.value()))
        self.width_label.setText(str(self.width_slider.value()))
        self.height_label.setText(str(self.height_slider.value()))

        row_stride_value = self.row_stride_slider.value()
        if row_stride_value == 0:
            self.row_stride_label.setText("0 (auto)")
        else:
            self.row_stride_label.setText(str(row_stride_value))

        self.row_padding_label.setText(str(self.row_padding_slider.value()))
        self.depth_label.setText(str(self.depth_slider.value()))
        self.skip_slices_label.setText(str(self.skip_slices_slider.value()))

        stride_value = self.slice_stride_slider.value()
        self.slice_stride_label.setText(str(stride_value))

    def update_enhancement_labels(self):
        self.brightness_label.setText(str(self.brightness_slider.value()))
        self.contrast_label.setText(f"{self.contrast_slider.value()}%")
        gamma_val = self.gamma_slider.value() / 100.0
        self.gamma_label.setText(f"{gamma_val:.1f}")

        if self.window_min_slider.value() == 0:
            self.window_min_label.setText("Auto")
        else:
            self.window_min_label.setText(f"{self.window_min_slider.value()}%")

        if self.window_max_slider.value() == 100:
            self.window_max_label.setText("Auto")
        else:
            self.window_max_label.setText(f"{self.window_max_slider.value()}%")

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Raw Image File", "", "All Files (*)"
        )
        if file_path:
            self.current_file = file_path
            self.file_path_label.setText(
                f"File: {os.path.basename(file_path)}"
            )

    def on_parameter_changed(self):
        self.update_slider_labels()
        sender = getattr(self, 'sender', lambda: None)()
        sender_name = None
        sender_value = None
        if sender is self.spacing_x_spin:
            sender_name = 'spacing_x'
            sender_value = self.spacing_x_spin.value()
        elif sender is self.spacing_y_spin:
            sender_name = 'spacing_y'
            sender_value = self.spacing_y_spin.value()
        elif sender is self.spacing_z_spin:
            sender_name = 'spacing_z'
            sender_value = self.spacing_z_spin.value()
        else:
            try:
                sender_name = sender.objectName() or sender.__class__.__name__
            except Exception:
                sender_name = 'unknown'
        if self.auto_update_checkbox.isChecked():
            self._last_load_reason = f"auto_update from {sender_name}"
            self.load_image()

    def on_manual_load_clicked(self):
        self._last_load_reason = "manual load button"
        self.load_image()

    def on_enhancement_changed(self):
        self.update_enhancement_labels()
        self.update_enhancement_parameters()
        if self.image_data is not None:
            self.update_slice_display()

    def update_enhancement_parameters(self):
        self.brightness = self.brightness_slider.value()
        self.contrast = self.contrast_slider.value() / 100.0
        self.gamma = self.gamma_slider.value() / 100.0

        if self.image_data is not None:
            current_slice = self.image_data[self.current_slice]
            if len(current_slice.shape) == 2:
                data_min = float(np.min(current_slice))
                data_max = float(np.max(current_slice))
                data_range = data_max - data_min

                if self.window_min_slider.value() == 0:
                    self.vmin = None
                else:
                    self.vmin = (
                        data_min
                        + (self.window_min_slider.value() / 100.0) * data_range
                    )

                if self.window_max_slider.value() == 100:
                    self.vmax = None
                else:
                    self.vmax = (
                        data_min
                        + (self.window_max_slider.value() / 100.0) * data_range
                    )

    def reset_enhancement(self):
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)
        self.gamma_slider.setValue(100)
        self.window_min_slider.setValue(0)
        self.window_max_slider.setValue(100)

    def update_curve_labels(self):
        """Update curve slider labels"""
        self.curve_x_pos_label.setText(str(self.curve_x_pos_slider.value()))
        self.curve_x_neg_label.setText(str(self.curve_x_neg_slider.value()))
        self.curve_y_pos_label.setText(str(self.curve_y_pos_slider.value()))
        self.curve_y_neg_label.setText(str(self.curve_y_neg_slider.value()))
        self.curve_z_pos_label.setText(str(self.curve_z_pos_slider.value()))
        self.curve_z_neg_label.setText(str(self.curve_z_neg_slider.value()))

    def on_curve_changed(self):
        """Handle curve slider changes"""
        self.update_curve_labels()
        self.update_curve_parameters()
        if self.image_data is not None:
            self.update_slice_display()

    def update_curve_parameters(self):
        """Update internal curve parameters from sliders"""
        self.curve_x_pos = self.curve_x_pos_slider.value() / 100.0
        self.curve_x_neg = self.curve_x_neg_slider.value() / 100.0
        self.curve_y_pos = self.curve_y_pos_slider.value() / 100.0
        self.curve_y_neg = self.curve_y_neg_slider.value() / 100.0
        self.curve_z_pos = self.curve_z_pos_slider.value() / 100.0
        self.curve_z_neg = self.curve_z_neg_slider.value() / 100.0

    def reset_curves(self):
        """Reset all curve deformations to zero"""
        self.curve_x_pos_slider.setValue(0)
        self.curve_x_neg_slider.setValue(0)
        self.curve_y_pos_slider.setValue(0)
        self.curve_y_neg_slider.setValue(0)
        self.curve_z_pos_slider.setValue(0)
        self.curve_z_neg_slider.setValue(0)

    def are_curves_active(self):
        """Check if any curve deformations are active"""
        return (abs(self.curve_x_pos) > 1e-6 or abs(self.curve_x_neg) > 1e-6 or
                abs(self.curve_y_pos) > 1e-6 or abs(self.curve_y_neg) > 1e-6 or
                abs(self.curve_z_pos) > 1e-6 or abs(self.curve_z_neg) > 1e-6)

    def on_corner_notes_toggled(self, checked):
        self.show_corner_notes = checked
        self.update_slice_display()

    def on_corner_selection_changed(self, idx):
        if self.use_corner_symmetry:
            # Map combo index to actual corner index for master corners
            self.selected_corner_index = self.master_corners[idx]
        else:
            self.selected_corner_index = idx
        self.update_corner_slider_states()
        self.sync_corner_sliders_from_positions()
        self.update_slice_display()

    def on_corner_slider_changed(self):
        x = self.corner_x_slider.value()
        y = self.corner_y_slider.value()
        z = self.corner_z_slider.value()
        self.corner_x_label.setText(str(x))
        self.corner_y_label.setText(str(y))
        self.corner_z_label.setText(str(z))

        if self.corner_positions is not None:
            self.corner_positions[self.selected_corner_index] = [x, y, z]

            # Apply symmetry if enabled and we're editing a master corner
            if self.use_corner_symmetry and self.selected_corner_index in self.master_corners:
                self.apply_corner_symmetry()
                # No need to sync sliders again since we're editing the current corner

        if self.image_data is not None:
            self.update_slice_display()

    def setup_corner_slider_ranges(self):
        if self.image_data is not None:
            d, h, w = (
                self.image_data.shape[0],
                self.image_data.shape[1],
                self.image_data.shape[2],
            )
        else:
            w = self.width_slider.value()
            h = self.height_slider.value()
            d = self.depth_slider.value()

        ext = max(w, h, d)
        x_min, x_max = -ext, (w - 1) + ext
        y_min, y_max = -ext, (h - 1) + ext
        z_min, z_max = -ext, (d - 1) + ext

        for slider, mn, mx in [
            (self.corner_x_slider, x_min, x_max),
            (self.corner_y_slider, y_min, y_max),
            (self.corner_z_slider, z_min, z_max),
        ]:
            slider.blockSignals(True)
            slider.setMinimum(int(mn))
            slider.setMaximum(int(mx))
            slider.blockSignals(False)

        self.sync_corner_sliders_from_positions()

    def sync_corner_sliders_from_positions(self):
        if self.corner_positions is None:
            return
        idx = self.selected_corner_index
        x, y, z = self.corner_positions[idx].astype(int)
        self.corner_x_slider.blockSignals(True)
        self.corner_y_slider.blockSignals(True)
        self.corner_z_slider.blockSignals(True)
        self.corner_x_slider.setValue(int(x))
        self.corner_y_slider.setValue(int(y))
        self.corner_z_slider.setValue(int(z))
        self.corner_x_slider.blockSignals(False)
        self.corner_y_slider.blockSignals(False)
        self.corner_z_slider.blockSignals(False)
        self.corner_x_label.setText(str(int(x)))
        self.corner_y_label.setText(str(int(y)))
        self.corner_z_label.setText(str(int(z)))

    def idx_to_ijk(self, idx):
        ix = (idx >> 0) & 1
        iy = (idx >> 1) & 1
        iz = (idx >> 2) & 1
        return ix, iy, iz

    def reset_corners_to_default(self):
        if self.image_data is not None:
            d, h, w = (
                self.image_data.shape[0],
                self.image_data.shape[1],
                self.image_data.shape[2],
            )
        else:
            w = self.width_slider.value()
            h = self.height_slider.value()
            d = self.depth_slider.value()

        cps = np.zeros((8, 3), dtype=np.float64)
        for idx in range(8):
            ix, iy, iz = self.idx_to_ijk(idx)
            cps[idx, 0] = ix * (w - 1)
            cps[idx, 1] = iy * (h - 1)
            cps[idx, 2] = iz * (d - 1)

        self.corner_positions = cps
        self.setup_corner_slider_ranges()
        self.update_slice_display()

    def update_corner_combo_items(self):
        """Update corner combo box items based on symmetry setting"""
        self.corner_combo.clear()
        if self.use_corner_symmetry:
            # Only show master corners when symmetry is enabled
            items = [
                "C000 (X-,Y-,Z-) [Master]",
                "C010 (X-,Y+,Z-) [Master]",
            ]
            self.corner_combo.addItems(items)
            # Map combo index to actual corner index
            if self.selected_corner_index not in self.master_corners:
                self.selected_corner_index = 0  # Default to C000
            combo_idx = self.master_corners.index(self.selected_corner_index)
            self.corner_combo.setCurrentIndex(combo_idx)
        else:
            # Show all corners when symmetry is disabled
            items = [
                "C000 (X-,Y-,Z-)",
                "C100 (X+,Y-,Z-)",
                "C010 (X-,Y+,Z-)",
                "C110 (X+,Y+,Z-)",
                "C001 (X-,Y-,Z+)",
                "C101 (X+,Y-,Z+)",
                "C011 (X-,Y+,Z+)",
                "C111 (X+,Y+,Z+)",
            ]
            self.corner_combo.addItems(items)
            self.corner_combo.setCurrentIndex(self.selected_corner_index)

    def on_corner_symmetry_toggled(self, checked):
        """Handle toggling of corner symmetry"""
        self.use_corner_symmetry = checked
        self.update_corner_combo_items()
        if checked:
            # Apply symmetry to all corners based on current master corners
            self.apply_corner_symmetry()
        self.update_corner_slider_states()
        self.sync_corner_sliders_from_positions()
        self.update_slice_display()

    def update_corner_slider_states(self):
        """Enable/disable corner sliders based on symmetry state and selected corner"""
        if self.use_corner_symmetry:
            # Only enable sliders if we're editing a master corner
            is_master = self.selected_corner_index in self.master_corners
            self.corner_x_slider.setEnabled(is_master)
            self.corner_y_slider.setEnabled(is_master)
            self.corner_z_slider.setEnabled(is_master)
        else:
            # Enable all sliders when symmetry is disabled
            self.corner_x_slider.setEnabled(True)
            self.corner_y_slider.setEnabled(True)
            self.corner_z_slider.setEnabled(True)

    def apply_corner_symmetry(self):
        """Apply symmetry transformations to all corners based on master corners C000 and C010"""
        if self.corner_positions is None:
            return

        # Get volume dimensions for center calculation
        if self.image_data is not None:
            d, h, w = (
                self.image_data.shape[0],
                self.image_data.shape[1],
                self.image_data.shape[2],
            )
        else:
            w = self.width_slider.value()
            h = self.height_slider.value()
            d = self.depth_slider.value()

        # Calculate center point
        center_x = (w - 1) / 2.0
        center_y = (h - 1) / 2.0
        center_z = (d - 1) / 2.0

        # Get master corner positions
        c000 = self.corner_positions[0]  # C000 (X-,Y-,Z-)
        c010 = self.corner_positions[2]  # C010 (X-,Y+,Z-)

        # Apply symmetry transformations:
        # Mirror C000 and C010 across center in X direction to get C100 and C110
        self.corner_positions[1] = [2*center_x - c000[0], c000[1], c000[2]]  # C100
        self.corner_positions[3] = [2*center_x - c010[0], c010[1], c010[2]]  # C110

        # Mirror all Z- plane corners across center in Z direction to get Z+ plane
        self.corner_positions[4] = [c000[0], c000[1], 2*center_z - c000[2]]  # C001
        self.corner_positions[5] = [self.corner_positions[1][0], self.corner_positions[1][1], 2*center_z - self.corner_positions[1][2]]  # C101
        self.corner_positions[6] = [c010[0], c010[1], 2*center_z - c010[2]]  # C011
        self.corner_positions[7] = [self.corner_positions[3][0], self.corner_positions[3][1], 2*center_z - self.corner_positions[3][2]]  # C111

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
            if dtype == np.uint16:
                dtype = ">u2"
            elif dtype == np.int16:
                dtype = ">i2"
            elif dtype == np.float32:
                dtype = ">f4"
            elif dtype == np.float64:
                dtype = ">f8"

        return dtype, byte_size, components

    def load_image(self):
        if not os.path.exists(self.current_file):
            self.status_text.append(
                f"Error: File {self.current_file} not found"
            )
            return

        try:
            width = self.width_slider.value()
            height = self.height_slider.value()
            depth = self.depth_slider.value()
            header_size = self.header_size_slider.value()
            footer_size = self.footer_size_slider.value()
            skip_slices = self.skip_slices_slider.value()
            row_stride = self.row_stride_slider.value()
            row_padding = self.row_padding_slider.value()
            slice_stride = self.slice_stride_slider.value()

            dtype, byte_size, components = self.get_pixel_info()

            row_data_size = width * byte_size * components
            if row_stride > 0:
                effective_row_stride = row_stride
            else:
                effective_row_stride = row_data_size + row_padding

            slice_data_size = height * effective_row_stride
            effective_slice_stride = slice_data_size + slice_stride

            total_header_size = (
                header_size + skip_slices * effective_slice_stride
            )

            file_size = os.path.getsize(self.current_file)
            available_data_size = file_size - total_header_size - footer_size
            if available_data_size <= 0:
                self.status_text.append(
                    "Error: Not enough data after header and footer"
                )
                return

            max_slices = int(
                (available_data_size + slice_stride) / effective_slice_stride
            )
            final_depth = min(depth, max_slices)

            if final_depth <= 0:
                self.status_text.append(
                    "Error: Not enough data for specified parameters"
                )
                return

            image_slices = []
            with open(self.current_file, "rb") as f:
                for slice_idx in range(final_depth):
                    slice_position = (
                        total_header_size + slice_idx * effective_slice_stride
                    )

                    if effective_row_stride == row_data_size:
                        f.seek(slice_position)
                        slice_bytes = f.read(height * row_data_size)
                        if len(slice_bytes) < height * row_data_size:
                            self.status_text.append(
                                f"Warning: Incomplete slice {slice_idx}, "
                                "truncating depth"
                            )
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
                                self.status_text.append(
                                    "Warning: Incomplete row "
                                    f"{row_idx} in slice {slice_idx}"
                                )
                                break
                            row_data = np.frombuffer(row_bytes, dtype=dtype)
                            row_data_list.append(row_data)
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

            # Update slice controls
            self.slice_slider.setMaximum(final_depth - 1)
            self.current_slice = 0
            self.slice_slider.setValue(0)

            # Reset view
            self.reset_zoom()

            # Reapply saved orientation (flips/rotations) after reload
            self.apply_saved_orientation()

            # Reset corners after orientation applied
            self.reset_corners_to_default()
            self.setup_corner_slider_ranges()

            # Apply symmetry if enabled
            if self.use_corner_symmetry:
                self.apply_corner_symmetry()

            # Update display
            self.update_slice_display()

            stride_info = ""
            if row_stride > 0:
                stride_info += f", row stride: {effective_row_stride}"
            elif row_padding > 0:
                stride_info += f", row padding: {row_padding}"
            if slice_stride > 0:
                stride_info += (
                    f", slice padding: {slice_stride} "
                    f"(stride: {effective_slice_stride})"
                )
            self.status_text.append(
                f"Loaded image: {width}x{height}x{final_depth}{stride_info}"
            )
            self._last_load_reason = None

        except Exception as e:
            self.status_text.append(f"Error loading image: {str(e)}")

    def update_slice_display(self):
        if self.image_data is None:
            return

        self.current_slice = self.slice_slider.value()
        max_slice = self.image_data.shape[0] - 1
        self.slice_label.setText(f"{self.current_slice}/{max_slice}")

        # Always warp in realtime
        if self.corner_positions is not None:
            slice_data = self.warp_slice_with_corners(self.current_slice)
        else:
            slice_data = self.image_data[self.current_slice].copy()

        # Apply enhancement for grayscale
        if len(slice_data.shape) == 2:
            slice_data = self.apply_image_enhancement(slice_data)

        # Display
        self.ax.clear()
        if len(slice_data.shape) == 3:
            self.ax.imshow(slice_data)
        else:
            self.ax.imshow(
                slice_data, cmap="gray", vmin=self.vmin, vmax=self.vmax
            )

        # Aspect from spacing
        try:
            aspect_ratio = float(self.spacing_y_spin.value()) / float(
                self.spacing_x_spin.value()
            )
            if aspect_ratio > 0:
                self.ax.set_aspect(aspect_ratio)
        except Exception:
            pass

        z_pos = self.current_slice * float(self.spacing_z_spin.value())
        symmetry_text = " (Symmetry: C000+C010→All)" if self.use_corner_symmetry else ""
        title = f"Slice {self.current_slice} (z={z_pos:.3f}) - Realtime 3D Corner Warp{symmetry_text}"
        self.ax.set_title(title)
        self.ax.axis("off")

        # Overlay: notes for corners
        self.draw_corner_notes()

        self.apply_zoom_and_pan()
        self.canvas.draw()

    def draw_corner_notes(self):
        if not self.show_corner_notes:
            return
        if self.corner_positions is None or self.image_data is None:
            return

        D = self.image_data.shape[0]
        H = self.image_data.shape[1]
        W = self.image_data.shape[2]
        t = 0.0 if D <= 1 else float(self.current_slice) / float(D - 1)

        cp = self.corner_positions
        cnames = [
            "C000",
            "C100",
            "C010",
            "C110",
            "C001",
            "C101",
            "C011",
            "C111",
        ]

        def lerp_pair(i_minus, i_plus, tval):
            return (1.0 - tval) * cp[i_minus] + tval * cp[i_plus]

        corners_info = [
            (4, 12, "left", "top", (0, 4), "TL"),
            (W - 4, 12, "right", "top", (1, 5), "TR"),
            (W - 4, H - 4, "right", "bottom", (3, 7), "BR"),
            (4, H - 4, "left", "bottom", (2, 6), "BL"),
        ]

        pair_color = (0.2, 1.0, 1.0)
        near0_color = (0.3, 1.0, 0.3)
        near1_color = (1.0, 0.4, 1.0)

        near_tol = 0.025
        is_near_0 = t <= near_tol
        is_near_1 = (1.0 - t) <= near_tol

        for sx, sy, ha, va, (i0, i1), tag in corners_info:
            p = lerp_pair(i0, i1, t)

            # Add master corner indicators when symmetry is enabled
            symmetry_indicator = ""
            if self.use_corner_symmetry:
                if i0 in self.master_corners or i1 in self.master_corners:
                    symmetry_indicator = " [M]"  # Master
                else:
                    symmetry_indicator = " [A]"  # Auto

            txt = (
                f"{tag} {cnames[i0]}↔{cnames[i1]}{symmetry_indicator}\n"
                f"3D @ t={t:.2f}: "
                f"X={p[0]:.1f}, Y={p[1]:.1f}, Z={p[2]:.1f}"
            )
            self.ax.text(
                sx,
                sy,
                txt,
                color=pair_color,
                ha=ha,
                va=va,
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    fc=(0.0, 0.0, 0.0, 0.35),
                    ec=(0.0, 0.0, 0.0, 0.6),
                ),
            )

        marker_size = 8
        edge_w = 1.2

        if is_near_0:
            pts = [
                (0, 0, cnames[0]),
                (W - 1, 0, cnames[1]),
                (W - 1, H - 1, cnames[3]),
                (0, H - 1, cnames[2]),
            ]
            for x, y, name in pts:
                self.ax.plot(
                    x,
                    y,
                    marker="o",
                    markersize=marker_size,
                    color=near0_color,
                    markeredgecolor="black",
                    markeredgewidth=edge_w,
                )
                self.ax.text(
                    x + (6 if x == 0 else -6),
                    y + (10 if y == 0 else -10),
                    name,
                    color=near0_color,
                    fontsize=9,
                    ha="left" if x == 0 else "right",
                    va="top" if y == 0 else "bottom",
                    bbox=dict(
                        boxstyle="round,pad=0.2", fc=(0, 0, 0, 0.3), ec="none"
                    ),
                )

        if is_near_1:
            pts = [
                (0, 0, cnames[4]),
                (W - 1, 0, cnames[5]),
                (W - 1, H - 1, cnames[7]),
                (0, H - 1, cnames[6]),
            ]
            for x, y, name in pts:
                self.ax.plot(
                    x,
                    y,
                    marker="o",
                    markersize=marker_size,
                    color=near1_color,
                    markeredgecolor="black",
                    markeredgewidth=edge_w,
                )
                self.ax.text(
                    x + (6 if x == 0 else -6),
                    y + (10 if y == 0 else -10),
                    name,
                    color=near1_color,
                    fontsize=9,
                    ha="left" if x == 0 else "right",
                    va="top" if y == 0 else "bottom",
                    bbox=dict(
                        boxstyle="round,pad=0.2", fc=(0, 0, 0, 0.3), ec="none"
                    ),
                )

        sel = self.selected_corner_index
        sel_tag_map = {
            0: "TL",
            1: "TR",
            3: "BR",
            2: "BL",
            4: "TL",
            5: "TR",
            7: "BR",
            6: "BL",
        }
        sel_tag = sel_tag_map.get(sel, None)
        if sel_tag is not None:
            if sel_tag == "TL":
                sx, sy, ha, va = 4, 12, "left", "top"
            elif sel_tag == "TR":
                sx, sy, ha, va = W - 4, 12, "right", "top"
            elif sel_tag == "BR":
                sx, sy, ha, va = W - 4, H - 4, "right", "bottom"
            else:
                sx, sy, ha, va = 4, H - 4, "left", "bottom"
            corner_names = ['C000','C100','C010','C110','C001','C101','C011','C111']
            selected_text = f"<< Selected {corner_names[sel]} >>"
            if self.use_corner_symmetry:
                if sel in self.master_corners:
                    selected_text += " [MASTER]"
                else:
                    selected_text += " [Auto]"

            self.ax.text(
                sx,
                sy + (14 if va == "top" else -14),
                selected_text,
                color="yellow",
                ha=ha,
                va=va,
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    fc=(0.2, 0.2, 0.0, 0.45),
                    ec=(0.0, 0.0, 0.0, 0.6),
                ),
            )

    def apply_image_enhancement(self, image):
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

        enhanced = np.clip(
            enhanced,
            0,
            (
                np.iinfo(image.dtype).max
                if np.issubdtype(image.dtype, np.integer)
                else 1.0
            ),
        )

        return enhanced.astype(image.dtype)

    def apply_zoom_and_pan(self):
        if self.image_data is None:
            return

        height, width = self.image_data[self.current_slice].shape[:2]

        center_x = width / 2 + self.pan_x
        center_y = height / 2 + self.pan_y

        half_width = width / (2 * self.zoom_factor)
        half_height = height / (2 * self.zoom_factor)

        xlim = [center_x - half_width, center_x + half_width]
        ylim = [center_y + half_height, center_y - half_height]

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

    def zoom_in(self):
        self.zoom_factor *= 1.5
        self.update_zoom_display()

    def zoom_out(self):
        self.zoom_factor /= 1.5
        if self.zoom_factor < 0.1:
            self.zoom_factor = 0.1
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
            self.zoom_factor /= 1.2
            if self.zoom_factor < 0.1:
                self.zoom_factor = 0.1

        self.update_zoom_display()

    def on_mouse_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            self.mouse_pressed = True
            self.last_mouse_x = event.xdata
            self.last_mouse_y = event.ydata

    def on_mouse_move(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if not self.mouse_pressed:
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

    def flip_axis(self, axis, record=True):
        if self.image_data is None:
            return
        axis_map = {"z": 0, "y": 1, "x": 2}
        if axis not in axis_map:
            return
        target_axis = axis_map[axis]
        self.image_data = np.flip(self.image_data, axis=target_axis)
        if axis == "z":
            self.current_slice = (
                self.image_data.shape[0] - 1 - self.current_slice
            )
            self.slice_slider.setValue(self.current_slice)
        self.slice_slider.setMaximum(self.image_data.shape[0] - 1)
        # Reset corners (orientation changed)
        self.reset_corners_to_default()
        self.update_slice_display()
        if record:
            self.orientation_ops.append(("flip", axis))

    def rotate_axis(self, axis, direction, record=True):
        if self.image_data is None:
            return
        k = 1 if direction > 0 else 3
        if axis == "z":
            axes = (1, 2)
        elif axis == "x":
            axes = (0, 1)
        elif axis == "y":
            axes = (0, 2)
        else:
            return

        self.image_data = np.rot90(self.image_data, k=k, axes=axes)

        if axis == "z":
            self._swap_spacing("x", "y")
        elif axis == "x":
            self._swap_spacing("y", "z")
        elif axis == "y":
            self._swap_spacing("x", "z")

        self.slice_slider.setMaximum(self.image_data.shape[0] - 1)
        self.current_slice = min(
            self.current_slice, self.image_data.shape[0] - 1
        )
        self.slice_slider.setValue(self.current_slice)
        self.reset_corners_to_default()
        self.update_slice_display()
        if record:
            self.orientation_ops.append(("rotate", axis, int(direction)))

    def _swap_spacing(self, a, b):
        spin_map = {
            "x": self.spacing_x_spin,
            "y": self.spacing_y_spin,
            "z": self.spacing_z_spin,
        }
        sa = spin_map[a]
        sb = spin_map[b]
        va, vb = sa.value(), sb.value()
        sa.blockSignals(True)
        sb.blockSignals(True)
        sa.setValue(vb)
        sb.setValue(va)
        sa.blockSignals(False)
        sb.blockSignals(False)
        self.debug_log(f"Spacing after swap: {a}={sa.value()} {b}={sb.value()}")

    def apply_saved_orientation(self):
        """Reapply recorded flips/rotations to current volume after reload."""
        if self.image_data is None or not self.orientation_ops:
            return

        # Decide whether to swap spacings on reapply
        reason = self._last_load_reason or ""
        skip_spacing_swaps = reason.startswith("auto_update from spacing_") or reason.startswith("auto_update from spacing") or reason.startswith("auto_update from spacing-")

        for op in self.orientation_ops:
            if not op:
                continue
            if op[0] == "flip":
                _, axis = op
                axis_map = {"z": 0, "y": 1, "x": 2}
                if axis not in axis_map:
                    continue
                target_axis = axis_map[axis]
                self.debug_log(f"Reapply op: flip {axis}")
                self.image_data = np.flip(self.image_data, axis=target_axis)
                if axis == "z":
                    self.current_slice = (
                        self.image_data.shape[0] - 1 - self.current_slice
                    )
            elif op[0] == "rotate":
                _, axis, direction = op
                k = 1 if int(direction) > 0 else 3
                if axis == "z":
                    axes = (1, 2)
                    if not skip_spacing_swaps:
                        self._swap_spacing("x", "y")
                elif axis == "x":
                    axes = (0, 1)
                    if not skip_spacing_swaps:
                        self._swap_spacing("y", "z")
                elif axis == "y":
                    axes = (0, 2)
                    if not skip_spacing_swaps:
                        self._swap_spacing("x", "z")
                else:
                    continue
                self.debug_log(f"Reapply op: rotate {axis} dir={int(direction)}")
                self.image_data = np.rot90(self.image_data, k=k, axes=axes)

        # Update slice controls after applying all ops
        self.slice_slider.setMaximum(self.image_data.shape[0] - 1)
        self.current_slice = min(self.current_slice, self.image_data.shape[0] - 1)
        self.slice_slider.setValue(self.current_slice)

    def offset_header(self, operation, mode):
        width = self.width_slider.value()
        height = self.height_slider.value()
        dtype, byte_size, components = self.get_pixel_info()

        if mode == "column":
            offset = 1
        elif mode == "row":
            offset = width
        elif mode == "slice":
            offset = width * height
        else:
            return

        offset *= byte_size * components

        current_value = self.header_size_slider.value()
        if operation == "sub":
            new_value = max(0, current_value - offset)
        else:
            new_value = min(
                self.header_size_slider.maximum(), current_value + offset
            )

        self.header_size_slider.setValue(new_value)

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
            footer_size = self.footer_size_slider.value()
            skip_slices = self.skip_slices_slider.value()
            row_stride = self.row_stride_slider.value()
            row_padding = self.row_padding_slider.value()
            slice_stride = self.slice_stride_slider.value()
            spacing_x = self.spacing_x_spin.value()
            spacing_y = self.spacing_y_spin.value()
            spacing_z = self.spacing_z_spin.value()

            pixel_type = self.pixel_type_combo.currentText()
            endianness = self.endianness_combo.currentText()

            base_name = os.path.splitext(self.current_file)[0]
            header_file = base_name + ".nhdr"

            type_map = {
                "8 bit unsigned": "uchar",
                "8 bit signed": "signed char",
                "16 bit unsigned": "ushort",
                "16 bit signed": "short",
                "float": "float",
                "double": "double",
                "24 bit RGB": "uchar",
            }

            dtype, byte_size, components = self.get_pixel_info()

            row_data_size = width * byte_size * components
            if row_stride > 0:
                effective_row_stride = row_stride
            else:
                effective_row_stride = row_data_size + row_padding

            slice_data_size = height * effective_row_stride
            effective_slice_stride = slice_data_size + slice_stride

            total_header_size = (
                header_size + skip_slices * effective_slice_stride
            )

            file_size = os.path.getsize(self.current_file)
            available_data_size = file_size - total_header_size - footer_size
            max_slices = int(
                (available_data_size + slice_stride) / effective_slice_stride
            )
            final_depth = min(depth, max_slices)

            with open(header_file, "w") as f:
                f.write("NRRD0004\n")
                f.write("# Complete NRRD file format specification at:\n")
                f.write("# http://teem.sourceforge.net/nrrd/format.html\n")
                f.write(f"type: {type_map[pixel_type]}\n")
                f.write("space: left-posterior-superior\n")

                if components > 1:
                    f.write("dimension: 4\n")
                    f.write(
                        f"sizes: {components} {width} {height} "
                        f"{final_depth}\n"
                    )
                    f.write(
                        "space directions: none "
                        f"({spacing_x},0,0) (0,{spacing_y},0) "
                        f"(0,0,{spacing_z})\n"
                    )
                    f.write("kinds: vector domain domain domain\n")
                else:
                    f.write("dimension: 3\n")
                    f.write(f"sizes: {width} {height} {final_depth}\n")
                    f.write(
                        f"space directions: ({spacing_x},0,0) "
                        f"(0,{spacing_y},0) (0,0,{spacing_z})\n"
                    )
                    f.write("kinds: domain domain domain\n")

                f.write(
                    f"endian: "
                    f"{'little' if endianness == 'Little endian' else 'big'}\n"
                )
                f.write("encoding: raw\n")
                f.write("space origin: (0,0,0)\n")

                if total_header_size > 0:
                    f.write(f"byte skip: {total_header_size}\n")

                if row_stride > 0:
                    f.write(
                        f"# Custom field: row stride = "
                        f"{effective_row_stride} bytes\n"
                    )
                elif row_padding > 0:
                    f.write(
                        f"# Custom field: row padding = "
                        f"{row_padding} bytes\n"
                    )
                if slice_stride > 0:
                    f.write(
                        "# Custom field: slice padding = "
                        f"{slice_stride} bytes "
                        f"(stride: {effective_slice_stride})\n"
                    )
                if footer_size > 0:
                    f.write(
                        f"# Custom field: footer size = "
                        f"{footer_size} bytes\n"
                    )

                f.write(f"data file: {os.path.basename(self.current_file)}\n")

            self.status_text.append(f"Generated NRRD header: {header_file}")
            QMessageBox.information(
                self, "Success", f"NRRD header generated: {header_file}"
            )

        except Exception as e:
            self.status_text.append(f"Error generating header: {str(e)}")
            QMessageBox.critical(
                self, "Error", f"Failed to generate header: {str(e)}"
            )

    def save_as_nrrd(self):
        if self.image_data is None:
            QMessageBox.warning(self, "Warning", "No image data to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save NRRD File", "", "NRRD Files (*.nrrd);;All Files (*)"
        )
        if not file_path:
            return

        try:
            spacing_x = self.spacing_x_spin.value()
            spacing_y = self.spacing_y_spin.value()
            spacing_z = self.spacing_z_spin.value()
            pixel_type = self.pixel_type_combo.currentText()
            endianness = self.endianness_combo.currentText()

            # Prepare data to save: current realtime-warped volume
            export_data = self.build_warped_volume_for_export()

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

            with open(file_path, "w") as f:
                f.write("NRRD0004\n")
                f.write("# Complete NRRD file format specification at:\n")
                f.write("# http://teem.sourceforge.net/nrrd/format.html\n")
                f.write(f"type: {type_map[pixel_type]}\n")
                f.write("space: left-posterior-superior\n")

                if components > 1:
                    f.write("dimension: 4\n")
                    f.write(f"sizes: {components} {width} {height} {depth}\n")
                    f.write(
                        f"space directions: none "
                        f"({spacing_x},0,0) (0,{spacing_y},0) "
                        f"(0,0,{spacing_z})\n"
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
                    f"endian: "
                    f"{'little' if endianness == 'Little endian' else 'big'}\n"
                )
                f.write("encoding: raw\n")
                f.write("space origin: (0,0,0)\n")
                f.write("\n")

            with open(file_path, "ab") as f:
                base_dtype_map = {
                    "8 bit unsigned": np.uint8,
                    "8 bit signed": np.int8,
                    "16 bit unsigned": np.uint16,
                    "16 bit signed": np.int16,
                    "float": np.float32,
                    "double": np.float64,
                    "24 bit RGB": np.uint8,
                }
                base_dtype = base_dtype_map[pixel_type]

                data_to_save = export_data.astype(base_dtype, copy=False)
                if (
                    endianness == "Big endian"
                    and data_to_save.dtype.itemsize > 1
                ):
                    data_to_save = data_to_save.byteswap().newbyteorder()

                if components > 1:
                    data_to_save = np.moveaxis(data_to_save, -1, 0)

                f.write(data_to_save.tobytes())

            self.status_text.append(f"Saved NRRD file: {file_path}")
            QMessageBox.information(
                self, "Success", f"NRRD file saved: {file_path}"
            )

        except Exception as e:
            self.status_text.append(f"Error saving NRRD file: {str(e)}")
            QMessageBox.critical(
                self, "Error", f"Failed to save NRRD file: {str(e)}"
            )

    def build_warped_volume_for_export(self):
        # If corners are exactly identity, skip the resample for speed
        if self.corner_positions is None or self.image_data is None:
            return self.image_data

        if self.are_corners_identity():
            return self.image_data

        D = self.image_data.shape[0]
        warped_slices = []
        for z in range(D):
            warped_slices.append(self.warp_slice_with_corners(z))
        return np.stack(warped_slices, axis=0)

    def are_corners_identity(self):
        # Check if corners match axis-aligned volume bounds
        if self.image_data is not None:
            d, h, w = (
                self.image_data.shape[0],
                self.image_data.shape[1],
                self.image_data.shape[2],
            )
        else:
            w = self.width_slider.value()
            h = self.height_slider.value()
            d = self.depth_slider.value()

        cps_default = np.zeros((8, 3), dtype=np.float64)
        for idx in range(8):
            ix, iy, iz = self.idx_to_ijk(idx)
            cps_default[idx, 0] = ix * (w - 1)
            cps_default[idx, 1] = iy * (h - 1)
            cps_default[idx, 2] = iz * (d - 1)

        return np.array_equal(self.corner_positions, cps_default)

    def apply_stretch(self):
        if self.image_data is None:
            return

        sx = self.scale_x_spin.value()
        sy = self.scale_y_spin.value()
        sz = self.scale_z_spin.value()
        preserve = self.preserve_size_checkbox.isChecked()

        if (
            abs(sx - 1.0) < 1e-9
            and abs(sy - 1.0) < 1e-9
            and abs(sz - 1.0) < 1e-9
        ):
            self.status_text.append("Stretch: scale factors are 1.0 (no-op).")
            return

        self.status_text.append(
            f"Stretching volume with scale (X,Y,Z)=({sx:.3f},{sy:.3f},{sz:.3f})"
        )
        QApplication.processEvents()

        try:
            from scipy import ndimage
        except ImportError:
            ndimage = None

        arr = self.image_data
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

        if preserve:
            self.spacing_x_spin.setValue(self.spacing_x_spin.value() / sx)
            self.spacing_y_spin.setValue(self.spacing_y_spin.value() / sy)
            self.spacing_z_spin.setValue(self.spacing_z_spin.value() / sz)

        self.slice_slider.setMaximum(self.image_data.shape[0] - 1)
        self.current_slice = min(
            self.current_slice, self.image_data.shape[0] - 1
        )
        self.slice_slider.setValue(self.current_slice)
        self.reset_zoom()
        self.reset_corners_to_default()
        self.update_slice_display()
        self.status_text.append("Stretch complete.")

    def apply_curve_transformation(self, X, Y, Z, D, H, W):
        """Apply curve deformations to coordinate arrays"""
        if not self.are_curves_active():
            return X, Y, Z

        # Normalize coordinates to [0,1] range for curve calculations
        x_norm = X / (W - 1) if W > 1 else np.zeros_like(X)
        y_norm = Y / (H - 1) if H > 1 else np.zeros_like(Y)
        z_norm = Z / (D - 1) if D > 1 else np.zeros_like(Z)

        # Apply X-axis curves (bend along YZ plane)
        if abs(self.curve_x_pos) > 1e-6 or abs(self.curve_x_neg) > 1e-6:
            # Curve strength varies across the volume
            curve_factor_x = np.zeros_like(x_norm)

            # Positive X curve affects the positive X half
            if abs(self.curve_x_pos) > 1e-6:
                mask_pos = x_norm >= 0.5
                intensity = (x_norm - 0.5) * 2.0  # 0 to 1 from center to edge
                curve_factor_x[mask_pos] += self.curve_x_pos * intensity[mask_pos]

            # Negative X curve affects the negative X half
            if abs(self.curve_x_neg) > 1e-6:
                mask_neg = x_norm <= 0.5
                intensity = (0.5 - x_norm) * 2.0  # 0 to 1 from center to edge
                curve_factor_x[mask_neg] += self.curve_x_neg * intensity[mask_neg]

            # Apply curve by offsetting Y and Z coordinates
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
        # Warp slice z_idx using 3D trilinear mapping defined by 8 corners.
        # Output dims equal to original slice dims.
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

        cp = self.corner_positions  # (8,3)
        c000 = cp[0]
        c100 = cp[1]
        c010 = cp[2]
        c110 = cp[3]
        c001 = cp[4]
        c101 = cp[5]
        c011 = cp[6]
        c111 = cp[7]

        X = (
            c000[0] * s0 * t0 * p0
            + c100[0] * s1 * t0 * p0
            + c010[0] * s0 * t1 * p0
            + c110[0] * s1 * t1 * p0
            + c001[0] * s0 * t0 * p1
            + c101[0] * s1 * t0 * p1
            + c011[0] * s0 * t1 * p1
            + c111[0] * s1 * t1 * p1
        )
        Y = (
            c000[1] * s0 * t0 * p0
            + c100[1] * s1 * t0 * p0
            + c010[1] * s0 * t1 * p0
            + c110[1] * s1 * t1 * p0
            + c001[1] * s0 * t0 * p1
            + c101[1] * s1 * t0 * p1
            + c011[1] * s0 * t1 * p1
            + c111[1] * s1 * t1 * p1
        )
        Z = (
            c000[2] * s0 * t0 * p0
            + c100[2] * s1 * t0 * p0
            + c010[2] * s0 * t1 * p0
            + c110[2] * s1 * t1 * p0
            + c001[2] * s0 * t0 * p1
            + c101[2] * s1 * t0 * p1
            + c011[2] * s0 * t1 * p1
            + c111[2] * s1 * t1 * p1
        )

        # Apply curve deformations after corner transformation
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
                out = ndimage.map_coordinates(
                    vol, [Z, Y, X], order=1, mode="constant", cval=0.0
                )
                return out
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
                if np.issubdtype(self.image_data.dtype, np.integer):
                    info = np.iinfo(self.image_data.dtype)
                    out = np.clip(out, info.min, info.max).astype(
                        self.image_data.dtype
                    )
                else:
                    out = out.astype(self.image_data.dtype)
                return out

    def get_current_config(self):
        """Get current configuration as a dictionary"""
        config = {
            # File parameters
            "current_file": getattr(self, 'current_file', DEFAULT_IMAGE_FILE),

            # Image parameters
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

            # Spacing parameters
            "spacing_x": self.spacing_x_spin.value(),
            "spacing_y": self.spacing_y_spin.value(),
            "spacing_z": self.spacing_z_spin.value(),

            # Enhancement parameters
            "brightness": self.brightness_slider.value(),
            "contrast": self.contrast_slider.value(),
            "gamma": self.gamma_slider.value(),
            "window_min": self.window_min_slider.value(),
            "window_max": self.window_max_slider.value(),

            # Scale parameters
            "scale_x": self.scale_x_spin.value(),
            "scale_y": self.scale_y_spin.value(),
            "scale_z": self.scale_z_spin.value(),
            "preserve_size": self.preserve_size_checkbox.isChecked(),

            # Curve parameters
            "curve_x_pos": self.curve_x_pos_slider.value(),
            "curve_x_neg": self.curve_x_neg_slider.value(),
            "curve_y_pos": self.curve_y_pos_slider.value(),
            "curve_y_neg": self.curve_y_neg_slider.value(),
            "curve_z_pos": self.curve_z_pos_slider.value(),
            "curve_z_neg": self.curve_z_neg_slider.value(),

            # Corner parameters
            "use_corner_symmetry": self.use_corner_symmetry,
            "show_corner_notes": self.show_corner_notes,
            "corner_positions": self.corner_positions.tolist() if self.corner_positions is not None else None,
            "selected_corner_index": self.selected_corner_index,

            # UI parameters
            "auto_update": self.auto_update_checkbox.isChecked(),

            # Orientation history
            "orientation_ops": self.orientation_ops,
        }
        self.debug_log(
            f"get_current_config spacing: x={config['spacing_x']}, y={config['spacing_y']}, z={config['spacing_z']} ops={len(config['orientation_ops'])}"
        )
        return config

    def apply_config(self, config):
        """Apply configuration to UI controls"""
        try:
            self.debug_log("apply_config start")
            # Block signals during bulk updates
            widgets = [
                self.pixel_type_combo, self.endianness_combo,
                self.header_size_slider, self.footer_size_slider,
                self.width_slider, self.height_slider,
                self.row_stride_slider, self.row_padding_slider,
                self.depth_slider, self.skip_slices_slider,
                self.slice_stride_slider, self.spacing_x_spin,
                self.spacing_y_spin, self.spacing_z_spin,
                self.brightness_slider, self.contrast_slider,
                self.gamma_slider, self.window_min_slider,
                self.window_max_slider, self.scale_x_spin,
                self.scale_y_spin, self.scale_z_spin,
                self.preserve_size_checkbox, self.curve_x_pos_slider,
                self.curve_x_neg_slider, self.curve_y_pos_slider,
                self.curve_y_neg_slider, self.curve_z_pos_slider,
                self.curve_z_neg_slider, self.corner_notes_checkbox,
                self.corner_symmetry_checkbox, self.auto_update_checkbox
            ]

            for widget in widgets:
                widget.blockSignals(True)

            # File parameters
            if "current_file" in config:
                self.current_file = config["current_file"]
                self.file_path_label.setText(f"File: {os.path.basename(self.current_file)}")

            # Image parameters
            if "pixel_type" in config:
                idx = self.pixel_type_combo.findText(config["pixel_type"])
                if idx >= 0:
                    self.pixel_type_combo.setCurrentIndex(idx)

            if "endianness" in config:
                idx = self.endianness_combo.findText(config["endianness"])
                if idx >= 0:
                    self.endianness_combo.setCurrentIndex(idx)

            # Slider parameters
            slider_configs = [
                ("header_size", self.header_size_slider),
                ("footer_size", self.footer_size_slider),
                ("width", self.width_slider),
                ("height", self.height_slider),
                ("row_stride", self.row_stride_slider),
                ("row_padding", self.row_padding_slider),
                ("depth", self.depth_slider),
                ("skip_slices", self.skip_slices_slider),
                ("slice_stride", self.slice_stride_slider),
                ("brightness", self.brightness_slider),
                ("contrast", self.contrast_slider),
                ("gamma", self.gamma_slider),
                ("window_min", self.window_min_slider),
                ("window_max", self.window_max_slider),
                ("curve_x_pos", self.curve_x_pos_slider),
                ("curve_x_neg", self.curve_x_neg_slider),
                ("curve_y_pos", self.curve_y_pos_slider),
                ("curve_y_neg", self.curve_y_neg_slider),
                ("curve_z_pos", self.curve_z_pos_slider),
                ("curve_z_neg", self.curve_z_neg_slider),
            ]

            for key, slider in slider_configs:
                if key in config:
                    slider.setValue(int(config[key]))

            # Spinbox parameters
            spinbox_configs = [
                ("spacing_x", self.spacing_x_spin),
                ("spacing_y", self.spacing_y_spin),
                ("spacing_z", self.spacing_z_spin),
                ("scale_x", self.scale_x_spin),
                ("scale_y", self.scale_y_spin),
                ("scale_z", self.scale_z_spin),
            ]

            for key, spinbox in spinbox_configs:
                if key in config:
                    spinbox.setValue(float(config[key]))

            # Checkbox parameters
            checkbox_configs = [
                ("preserve_size", self.preserve_size_checkbox),
                ("show_corner_notes", self.corner_notes_checkbox),
                ("use_corner_symmetry", self.corner_symmetry_checkbox),
                ("auto_update", self.auto_update_checkbox),
            ]

            for key, checkbox in checkbox_configs:
                if key in config:
                    checkbox.setChecked(bool(config[key]))

            # Corner parameters
            if "corner_positions" in config and config["corner_positions"] is not None:
                self.corner_positions = np.array(config["corner_positions"])

            if "selected_corner_index" in config:
                self.selected_corner_index = int(config["selected_corner_index"])

            # Orientation history
            if "orientation_ops" in config:
                self.orientation_ops = list(config["orientation_ops"])  # ensure list type

            # Restore signals
            for widget in widgets:
                widget.blockSignals(False)

            # Update labels and UI state
            self.update_slider_labels()
            self.update_enhancement_labels()
            self.update_curve_labels()
            self.update_corner_combo_items()
            if self.corner_positions is not None:
                self.setup_corner_slider_ranges()
                self.sync_corner_sliders_from_positions()

            # Apply corner symmetry if enabled
            if self.use_corner_symmetry:
                self.apply_corner_symmetry()

            self.status_text.append("Configuration loaded successfully")

        except Exception as e:
            self.status_text.append(f"Error applying configuration: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to apply configuration: {str(e)}")

    def save_config(self):
        """Save current configuration to a JSON file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "", "JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return

        try:
            config = self.get_current_config()
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=4)
            self.status_text.append(f"Configuration saved to: {file_path}")
            QMessageBox.information(self, "Success", f"Configuration saved to: {file_path}")

        except Exception as e:
            self.status_text.append(f"Error saving configuration: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save configuration: {str(e)}")

    def load_config_dialog(self):
        """Load configuration from a JSON file via dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", "JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return

        self.load_config(file_path)

    def load_config(self, file_path):
        """Load configuration from a JSON file"""
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            self.apply_config(config)
            self.status_text.append(f"Configuration loaded from: {file_path}")

        except Exception as e:
            self.status_text.append(f"Error loading configuration: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load configuration: {str(e)}")

    def main(self):
        pass


def main():
    parser = argparse.ArgumentParser(description='3D Ultrasound Scrapper')
    parser.add_argument('--config', '-c', type=str, help='Path to configuration JSON file')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = RawImageGuessQt(config_file=args.config)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
