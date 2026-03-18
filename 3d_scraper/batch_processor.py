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

from shared.config_parsing import config_to_params
from shared.geometry import build_export_volume
from shared.nrrd_io import save_nrrd as save_nrrd_shared
from shared.raw_io import read_raw_volume
from shared.types import NrrdParams, VolumeParams
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
        self.current_file = None
        self.params = config_to_params(config)

    def set_input_file(self, file_path):
        self.current_file = file_path

    def process_file(self):
        """Process a single file using the configuration"""
        if not os.path.exists(self.current_file):
            return None

        try:
            p = self.params

            volume = read_raw_volume(
                self.current_file,
                p.volume,
                p.raw_layout,
            )
            if volume is None:
                return None

            from shared.geometry import apply_orientation
            from shared.types import OrientationParams

            volume = apply_orientation(volume, p.orientation)

            volume = build_export_volume(volume, p.warp, p.curve, p.crop)

            return volume
        except Exception as e:
            print(f"Error processing {self.current_file}: {str(e)}")
            return None

    def save_nrrd(self, data, output_path):
        """Save data as NRRD file"""
        if data is None:
            return

        if data.ndim == 4:
            depth, height, width = data.shape[:3]
        else:
            depth, height, width = data.shape

        p = self.params
        volume_params = VolumeParams(
            width=width,
            height=height,
            depth=depth,
            pixel_type=p.volume.pixel_type,
            endianness=p.volume.endianness,
            spacing_x=p.volume.spacing_x,
            spacing_y=p.volume.spacing_y,
            spacing_z=p.volume.spacing_z,
        )
        nrrd_params = NrrdParams()

        save_nrrd_shared(
            output_path,
            data,
            volume_params,
            nrrd_params,
        )

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
        p = self.params
        scale_x = p.volume.spacing_x if p.volume.spacing_x > 0 else 1.0
        scale_y = p.volume.spacing_y if p.volume.spacing_y > 0 else 1.0

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
