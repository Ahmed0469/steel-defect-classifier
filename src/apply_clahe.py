"""
Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to every
image in a folder structure (one subfolder per class) and saves the result
into a mirrored output folder.

clipLimit=1.5 was chosen after comparing 1.5 vs 2.0: 2.0 over-sharpened
rolled-in_scale's texture enough to make it resemble scratches, causing a
new confusion that didn't exist before CLAHE was applied. 1.5 preserved the
gains on crazing/inclusion/scratches without introducing that regression.

Usage:
    python src/apply_clahe.py --input Data/train --output Data/train_clahe
    python src/apply_clahe.py --input Data/test  --output Data/test_clahe
"""

import argparse
import cv2
from pathlib import Path


def apply_clahe_to_folder(input_dir: str, output_dir: str, clip_limit: float = 1.5, tile_grid_size: int = 8):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))

    count = 0
    for class_folder in input_path.iterdir():
        if not class_folder.is_dir():
            continue

        out_class_folder = output_path / class_folder.name
        out_class_folder.mkdir(parents=True, exist_ok=True)

        for img_file in class_folder.iterdir():
            if not img_file.is_file():
                continue

            gray = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"Skipping unreadable file: {img_file}")
                continue

            enhanced = clahe.apply(gray)
            cv2.imwrite(str(out_class_folder / img_file.name), enhanced)
            count += 1

    print(f"Processed {count} images -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input folder (class-per-subfolder)")
    parser.add_argument("--output", required=True, help="Output folder for CLAHE'd images")
    parser.add_argument("--clip_limit", type=float, default=1.5)
    parser.add_argument("--tile_grid_size", type=int, default=8)
    args = parser.parse_args()

    apply_clahe_to_folder(args.input, args.output, args.clip_limit, args.tile_grid_size)
