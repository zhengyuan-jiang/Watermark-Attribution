"""Inference-only HiDDeN model used by the optional checkpoint demo.

The encoder is adapted from ando-khachatryan/HiDDeN. The ResNet decoder is
adapted from kuangliu/pytorch-cifar. Both upstream projects use the MIT
License; their notices are included in THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional


@dataclass(frozen=True)
class HiDDeNConfig:
    """Architecture settings for the released 64-bit checkpoint."""

    height: int = 128
    width: int = 128
    message_length: int = 64
    encoder_blocks: int = 4
    encoder_channels: int = 64
    decoder_channels: int = 64


class ConvBNRelu(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, stride: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                channels_in,
                channels_out,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class Encoder(nn.Module):
    """Embed a bit-string watermark into a normalized RGB image."""

    def __init__(self, config: HiDDeNConfig):
        super().__init__()
        self.height = config.height
        self.width = config.width
        self.message_length = config.message_length

        layers: list[nn.Module] = [ConvBNRelu(3, config.encoder_channels)]
        for _ in range(config.encoder_blocks - 1):
            layers.append(
                ConvBNRelu(config.encoder_channels, config.encoder_channels)
            )
        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(
            config.encoder_channels + 3 + config.message_length,
            config.encoder_channels,
        )
        self.final_layer = nn.Conv2d(config.encoder_channels, 3, kernel_size=1)

    def forward(
        self,
        image: torch.Tensor,
        message: torch.Tensor,
    ) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1:] != (3, self.height, self.width):
            raise ValueError(
                f"image must have shape (batch, 3, {self.height}, {self.width})"
            )
        if message.ndim != 2 or message.shape[0] != image.shape[0]:
            raise ValueError("message must have shape (batch, message_length)")
        if message.shape[1] != self.message_length:
            raise ValueError(
                f"message must contain {self.message_length} bits per image"
            )

        expanded_message = message.unsqueeze(-1).unsqueeze(-1)
        expanded_message = expanded_message.expand(
            -1,
            -1,
            self.height,
            self.width,
        )
        encoded_features = self.conv_layers(image)
        combined = torch.cat((expanded_message, encoded_features, image), dim=1)
        return self.final_layer(self.after_concat_layer(combined))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = functional.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        outputs = outputs + self.shortcut(inputs)
        return functional.relu(outputs)


class ResNetDecoder(nn.Module):
    """Decode a 64-bit watermark from an RGB image."""

    def __init__(self, config: HiDDeNConfig):
        super().__init__()
        self.in_planes = config.decoder_channels
        self.conv1 = nn.Conv2d(
            3,
            config.decoder_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(config.decoder_channels)
        self.layer1 = self._make_layer(
            config.decoder_channels,
            blocks=2,
            stride=1,
        )
        self.layer2 = self._make_layer(
            config.decoder_channels * 2,
            blocks=2,
            stride=2,
        )
        self.layer3 = self._make_layer(
            config.decoder_channels * 4,
            blocks=2,
            stride=2,
        )
        self.layer4 = self._make_layer(
            config.decoder_channels * 8,
            blocks=2,
            stride=2,
        )
        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(
            config.decoder_channels * 8,
            config.message_length,
        )

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers: list[nn.Module] = []
        for block_stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, block_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        outputs = functional.relu(self.bn1(self.conv1(image)))
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = self.avgpooling(outputs)
        outputs = torch.flatten(outputs, 1)
        return self.linear(outputs)


class HiDDeNInference(nn.Module):
    """Encoder and decoder pair without training-only components."""

    def __init__(self, config: HiDDeNConfig | None = None):
        super().__init__()
        self.config = config or HiDDeNConfig()
        self.encoder = Encoder(self.config)
        self.decoder = ResNetDecoder(self.config)

    def forward(
        self,
        image: torch.Tensor,
        message: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(image, message)
        return encoded, self.decoder(encoded)


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve ``auto``, ``cpu``, or ``cuda`` to a PyTorch device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device)


def load_hidden_checkpoint(
    checkpoint_path: str | Path,
    device: str = "auto",
) -> HiDDeNInference:
    """Load the released encoder/decoder weights for inference."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    resolved_device = resolve_device(device)
    checkpoint = torch.load(
        path,
        map_location=resolved_device,
        weights_only=True,
    )
    required_keys = {"enc-model", "dec-model"}
    if not isinstance(checkpoint, dict) or not required_keys <= checkpoint.keys():
        raise ValueError("checkpoint does not contain enc-model and dec-model")

    model = HiDDeNInference().to(resolved_device)
    model.encoder.load_state_dict(checkpoint["enc-model"])
    model.decoder.load_state_dict(checkpoint["dec-model"])
    model.eval()
    return model
