#!/usr/bin/env python3
"""Optional one-image smoke test for the released HiDDeN checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from watermark_attribution.hidden import (  # noqa: E402
    load_hidden_checkpoint,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Embed and decode one watermark in a deterministic synthetic "
            "image. Install requirements-hidden.txt first."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="path to 64bitsRes099.pth",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="inference device",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="optional natural image; otherwise a synthetic image is used",
    )
    parser.add_argument("--seed", type=int, default=0, help="message seed")
    return parser.parse_args()


def synthetic_image(device: torch.device) -> torch.Tensor:
    """Create one smooth RGB image normalized to [-1, 1]."""
    axis = torch.linspace(-1.0, 1.0, 128, device=device)
    vertical, horizontal = torch.meshgrid(axis, axis, indexing="ij")
    blue = torch.sin(horizontal * torch.pi) * torch.cos(vertical * torch.pi)
    return torch.stack((horizontal, vertical, blue)).unsqueeze(0)


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    """Center-crop and normalize an image to the checkpoint's input format."""
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required for --image; install requirements-hidden.txt"
        ) from error

    with Image.open(path) as source:
        image = source.convert("RGB")
        edge = min(image.size)
        left = (image.width - edge) // 2
        top = (image.height - edge) // 2
        image = image.crop((left, top, left + edge, top + edge))
        image = image.resize((128, 128), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float32)

    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    return (tensor.to(device) / 127.5) - 1.0


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model = load_hidden_checkpoint(args.checkpoint, args.device)

    generator = torch.Generator().manual_seed(args.seed)
    message = torch.randint(
        0,
        2,
        (1, model.config.message_length),
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    image = (
        load_image(args.image, device)
        if args.image is not None
        else synthetic_image(device)
    )

    with torch.inference_mode():
        encoded, decoded_values = model(image, message)
        decoded = decoded_values.round().clamp(0, 1)

    bit_accuracy = torch.mean((decoded == message).float()).item()
    image_mse = torch.mean((encoded - image) ** 2).item()
    print(f"device:       {device}")
    print(f"bit accuracy: {bit_accuracy:.4f}")
    print(f"image MSE:    {image_mse:.6f}")


if __name__ == "__main__":
    main()
