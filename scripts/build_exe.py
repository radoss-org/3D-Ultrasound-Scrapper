#!/usr/bin/env python3
"""
Build script for 3D Ultrasound Scrapper executables
Creates standalone .exe files using PyInstaller
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_pyinstaller():
    """Check if PyInstaller is installed"""
    try:
        subprocess.run([sys.executable, "-m", "PyInstaller", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_pyinstaller():
    """Install PyInstaller"""
    print("\nPyInstaller not found. Installing...")
    cmd = [sys.executable, "-m", "pip", "install", "pyinstaller"]
    return run_command(cmd, "Installing PyInstaller")

def clean_build_dirs():
    """Clean previous build directories"""
    dirs_to_clean = ["build", "dist", "__pycache__"]

    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            print(f"Cleaning {dir_name}...")
            shutil.rmtree(dir_name)

    # Clean spec files
    for spec_file in Path(".").glob("*.spec"):
        print(f"Removing {spec_file}...")
        spec_file.unlink()

def build_main_app():
    """Build the main 3D Ultrasound Scrapper application"""
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--windowed",
        "--name", "3D_Ultrasound_Scrapper",
        "--icon", "NONE",  # You can add an .ico file path here if you have one
        "--hidden-import", "scipy",
        "--hidden-import", "scipy.ndimage",
        "--hidden-import", "matplotlib.backends.backend_qt5agg",
        "--hidden-import", "PyQt5.QtCore",
        "--hidden-import", "PyQt5.QtWidgets",
        "--hidden-import", "PyQt5.QtGui",
        "test.py"
    ]

    return run_command(cmd, "Building main application")

def build_batch_processor():
    """Build the batch processor application"""
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--windowed",
        "--name", "3D_Ultrasound_Scrapper_Batch",
        "--icon", "NONE",  # You can add an .ico file path here if you have one
        "--hidden-import", "scipy",
        "--hidden-import", "scipy.ndimage",
        "--hidden-import", "PyQt5.QtCore",
        "--hidden-import", "PyQt5.QtWidgets",
        "--hidden-import", "PyQt5.QtGui",
        "batch_processor.py"
    ]

    return run_command(cmd, "Building batch processor")

def create_dist_folder():
    """Create a clean distribution folder"""
    dist_folder = Path("../dist")

    if dist_folder.exists():
        shutil.rmtree(dist_folder)

    dist_folder.mkdir()

    # Copy executables
    dist_path = Path("dist")
    if dist_path.exists():
        for exe_file in dist_path.glob("*.exe"):
            dest_file = dist_folder / exe_file.name
            shutil.copy2(exe_file, dest_file)
            print(f"Copied {exe_file.name} to {dist_folder}")

    return dist_folder

def main():
    """Main build process"""
    print("=" * 60)
    print("3D Ultrasound Scrapper - Build Script")
    print("=" * 60)

    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    print(f"Script location: {script_dir}")

    # Change to the script directory so PyInstaller can find the Python files
    original_cwd = Path.cwd()
    os.chdir(script_dir)
    print(f"Working directory: {script_dir}")

    try:
        # Check if required files exist in the script directory
        required_files = ["test.py", "batch_processor.py"]
        missing_files = [f for f in required_files if not os.path.exists(f)]

        if missing_files:
            print(f"✗ Missing required files in {script_dir}: {', '.join(missing_files)}")
            print("Make sure test.py and batch_processor.py are in the scripts directory")
            sys.exit(1)

        # Check and install PyInstaller if needed
        if not check_pyinstaller():
            if not install_pyinstaller():
                print("✗ Failed to install PyInstaller")
                sys.exit(1)
        else:
            print("✓ PyInstaller is available")

        # Clean previous builds
        clean_build_dirs()

        # Build applications
        success = True

        if not build_main_app():
            success = False

        if not build_batch_processor():
            success = False

        if success:
            # Create final distribution
            dist_folder = create_dist_folder()

            print("\n" + "=" * 60)
            print("✓ BUILD COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Executables are available in: {dist_folder.absolute()}")
            print("\nFiles created:")
            for file in dist_folder.iterdir():
                if file.is_file():
                    size_mb = file.stat().st_size / (1024 * 1024)
                    print(f"  - {file.name} ({size_mb:.1f} MB)")

            print("\nUsage:")
            print("  - Run 3D_Ultrasound_Scrapper.exe for the main application")
            print("  - Run 3D_Ultrasound_Scrapper_Batch.exe for batch processing")
            print("  - Use example_config.json as a starting template")

        else:
            print("\n" + "=" * 60)
            print("✗ BUILD FAILED")
            print("=" * 60)
            print("Check the error messages above for details.")
            sys.exit(1)

    finally:
        # Always return to the original directory
        os.chdir(original_cwd)

if __name__ == "__main__":
    main()