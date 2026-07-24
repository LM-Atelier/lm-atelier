from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def draw_icon(size: int) -> Image.Image:
    scale = size / 256
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (0, 0, size - 1, size - 1),
        radius=62 * scale,
        fill="#1b1b27",
        outline="#655b89",
        width=max(round(4 * scale), 1),
    )

    frame = [(57, 199), (116, 57), (140, 57), (199, 199)]
    draw.line(
        [(round(x * scale), round(y * scale)) for x, y in frame],
        fill="#d9d4ff",
        width=max(round(15 * scale), 1),
        joint="curve",
    )
    draw.line(
        [(round(78 * scale), round(153 * scale)), (round(178 * scale), round(153 * scale))],
        fill="#d9d4ff",
        width=max(round(15 * scale), 1),
    )

    thread = [(48, 103), (83, 80), (117, 145), (156, 121), (189, 98), (216, 72)]
    draw.line(
        [(round(x * scale), round(y * scale)) for x, y in thread],
        fill="#aaa1ff",
        width=max(round(13 * scale), 1),
        joint="curve",
    )
    radius = 12 * scale
    for x, y in ((48, 103), (216, 72)):
        draw.ellipse(
            (
                round(x * scale - radius),
                round(y * scale - radius),
                round(x * scale + radius),
                round(y * scale + radius),
            ),
            fill="#8074e8",
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
