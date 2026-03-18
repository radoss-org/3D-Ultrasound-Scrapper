import argparse
import sys

from PyQt5.QtWidgets import QApplication
from raw_image_guess_window import RawImageGuessQt


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
