from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def draw_icon(size: int) -> Image.Image:
    scale = size / 400
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (0, 0, size - 1, size - 1),
        radius=80 * scale,
        fill="#12110f",
    )

    draw.polygon(
        [(round(x * scale), round(y * scale)) for x, y in (
            (43, 20), (107, 20), (107, 320), (276, 320), (276, 380), (43, 380)
        )],
        fill="#efe7d8",
    )
    draw.polygon(
        [(round(x * scale), round(y * scale)) for x, y in (
            (125, 20), (243, 129), (356, 20), (356, 380), (298, 380),
            (298, 288), (213, 288), (259, 243), (298, 243), (298, 161),
            (164, 303), (125, 303)
        )],
        fill="#315a9a",
    )
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    icon = draw_icon(256)
    icon.save(args.output_dir / "lm-atelier.png", format="PNG")
    icon.save(
        args.output_dir / "lm-atelier.ico",
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )


if __name__ == "__main__":
    main()
