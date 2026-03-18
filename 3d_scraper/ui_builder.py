from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

DEFAULT_IMAGE_FILE = "sample_image.raw"


class RawImageGuessUiMixin:
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

        orientation_group = QGroupBox("Orientation")
        orientation_layout = QGridLayout(orientation_group)
        self.add_orientation_buttons(orientation_layout)
        layout.addWidget(orientation_group)

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

        self.generate_header_button = QPushButton("Generate NRRD Header")
        self.generate_header_button.clicked.connect(self.generate_nrrd_header)
        layout.addWidget(self.generate_header_button)

        self.save_nrrd_button = QPushButton("Save as NRRD")
        self.save_nrrd_button.clicked.connect(self.save_as_nrrd)
        layout.addWidget(self.save_nrrd_button)

        config_group = QGroupBox("Configuration")
        config_layout = QHBoxLayout(config_group)
        save_config_btn = QPushButton("Save Config")
        save_config_btn.clicked.connect(self.save_config)
        config_layout.addWidget(save_config_btn)
        load_config_btn = QPushButton("Load Config")
        load_config_btn.clicked.connect(self.load_config_dialog)
        config_layout.addWidget(load_config_btn)
        layout.addWidget(config_group)

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
