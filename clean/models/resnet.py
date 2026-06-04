from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
        activation: bool = True,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = ConvBNAct(out_channels, out_channels, kernel_size=3, activation=False)
        self.downsample = downsample
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv2(self.conv1(x))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.act(out + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()
        width = out_channels
        self.conv1 = ConvBNAct(in_channels, width, kernel_size=1, padding=0)
        self.conv2 = ConvBNAct(width, width, kernel_size=3, stride=stride)
        self.conv3 = ConvBNAct(width, out_channels * self.expansion, kernel_size=1, padding=0, activation=False)
        self.downsample = downsample
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv3(self.conv2(self.conv1(x)))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.act(out + identity)


class ResNetClassifier(nn.Module):
    """
    ResNet-D image classifier for 2D spectra.

    Inputs are expected as tensors shaped (batch, channels, height, width), like the
    single-resolution YOLO spectrogram inputs. The default configuration is ResNet-50D.
    """

    def __init__(
        self,
        num_classes: int,
        input_canals: int = 1,
        block: type[BasicBlock] | type[Bottleneck] = Bottleneck,
        layers: Sequence[int] = (3, 4, 6, 3),
        stem_channels: int = 64,
        dropout: float = 0.0,
        zero_init_residual: bool = True,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.input_canals = int(input_canals)
        self.in_channels = int(stem_channels)

        stem_mid = stem_channels // 2
        self.stem = nn.Sequential(
            ConvBNAct(self.input_canals, stem_mid, kernel_size=3, stride=2),
            ConvBNAct(stem_mid, stem_mid, kernel_size=3, stride=1),
            ConvBNAct(stem_mid, stem_channels, kernel_size=3, stride=1),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(512 * block.expansion, self.num_classes)

        self._init_weights(zero_init_residual=zero_init_residual)

        if device is not None:
            self.to(torch.device(device))

    def _make_layer(
        self,
        block: type[BasicBlock] | type[Bottleneck],
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        expanded_channels = out_channels * block.expansion
        downsample = None
        if stride != 1 or self.in_channels != expanded_channels:
            downsample_layers = []
            if stride != 1:
                downsample_layers.append(nn.AvgPool2d(kernel_size=2, stride=stride, ceil_mode=True, count_include_pad=False))
                conv_stride = 1
            else:
                conv_stride = 1
            downsample_layers.append(
                ConvBNAct(self.in_channels, expanded_channels, kernel_size=1, stride=conv_stride, padding=0, activation=False)
            )
            downsample = nn.Sequential(*downsample_layers)

        layers = [block(self.in_channels, out_channels, stride=stride, downsample=downsample)]
        self.in_channels = expanded_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self, zero_init_residual: bool) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck):
                    nn.init.zeros_(module.conv3.bn.weight)
                elif isinstance(module, BasicBlock):
                    nn.init.zeros_(module.conv2.bn.weight)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.stem(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self(x), dim=1)

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels.long())


def resnet18_classifier(num_classes: int, input_canals: int = 1, **kwargs) -> ResNetClassifier:
    return ResNetClassifier(num_classes, input_canals=input_canals, block=BasicBlock, layers=(2, 2, 2, 2), **kwargs)


def resnet34_classifier(num_classes: int, input_canals: int = 1, **kwargs) -> ResNetClassifier:
    return ResNetClassifier(num_classes, input_canals=input_canals, block=BasicBlock, layers=(3, 4, 6, 3), **kwargs)


def resnet50d_classifier(num_classes: int, input_canals: int = 1, **kwargs) -> ResNetClassifier:
    return ResNetClassifier(num_classes, input_canals=input_canals, block=Bottleneck, layers=(3, 4, 6, 3), **kwargs)


def resnet50_classifier(num_classes: int, input_canals: int = 1, **kwargs) -> ResNetClassifier:
    return resnet50d_classifier(num_classes, input_canals=input_canals, **kwargs)


def resnet101d_classifier(num_classes: int, input_canals: int = 1, **kwargs) -> ResNetClassifier:
    return ResNetClassifier(num_classes, input_canals=input_canals, block=Bottleneck, layers=(3, 4, 23, 3), **kwargs)


def labels_from_yolo_targets(targets: torch.Tensor, batch_size: int, ignore_empty: bool = False) -> torch.Tensor:
    """
    Build image-level labels from collated YOLO targets shaped [N, >=2].

    Each image must contain exactly one class label. Empty images or images with
    multiple different classes raise an error unless ignore_empty=True, in which
    case empty images receive label -1 and must be filtered before loss.
    """
    labels = torch.full((batch_size,), -1, dtype=torch.long, device=targets.device)
    for image_idx in range(batch_size):
        mask = targets[:, 0].long() == image_idx if targets.numel() else torch.zeros(0, dtype=torch.bool, device=targets.device)
        image_classes = targets[mask, 1].long().unique()
        if image_classes.numel() == 0:
            if ignore_empty:
                continue
            raise ValueError(f"Image {image_idx} has no class target.")
        if image_classes.numel() > 1:
            raise ValueError(f"Image {image_idx} has multiple class targets: {image_classes.tolist()}.")
        labels[image_idx] = image_classes[0]
    return labels


ResNet = ResNetClassifier


__all__ = [
    "BasicBlock",
    "Bottleneck",
    "ResNet",
    "ResNetClassifier",
    "labels_from_yolo_targets",
    "resnet18_classifier",
    "resnet34_classifier",
    "resnet50_classifier",
    "resnet50d_classifier",
    "resnet101d_classifier",
]
