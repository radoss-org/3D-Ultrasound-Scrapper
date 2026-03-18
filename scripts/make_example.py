import numpy as np

OUTPUT_FILE = "red_sphere_300x300x300.raw"

WIDTH = 300
HEIGHT = 300
DEPTH = 300
RADIUS = 100

SPHERE_COLOR = np.array([255, 0, 0], dtype=np.uint8)
BACKGROUND_COLOR = np.array([255, 255, 255], dtype=np.uint8)


def main():
    volume = np.full(
        (DEPTH, HEIGHT, WIDTH, 3),
        BACKGROUND_COLOR,
        dtype=np.uint8,
    )

    center_x = (WIDTH - 1) / 2.0
    center_y = (HEIGHT - 1) / 2.0
    center_z = (DEPTH - 1) / 2.0

    z, y, x = np.ogrid[:DEPTH, :HEIGHT, :WIDTH]
    distance_sq = (
        (x - center_x) ** 2 + (y - center_y) ** 2 + (z - center_z) ** 2
    )

    mask = distance_sq <= RADIUS**2
    volume[mask] = SPHERE_COLOR

    with open(OUTPUT_FILE, "wb") as f:
        f.write(volume.tobytes())

    print(f"Wrote {OUTPUT_FILE}")
    print(f"Volume shape: {volume.shape}")
    print("Use these settings in your program:")
    print("Pixel Type: 24 bit RGB")
    print("Endianness: Little endian")
    print("Header Size: 0")
    print("Footer Size: 0")
    print(f"Width: {WIDTH}")
    print(f"Height: {HEIGHT}")
    print(f"Depth: {DEPTH}")


if __name__ == "__main__":
    main()
