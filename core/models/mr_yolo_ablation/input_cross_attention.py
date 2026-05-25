from typing import List, Tuple

import torch
import torch.nn as nn

from ...nn.convs import Conv
from ..yolov11 import YOLOv11
from .fusion import InterResolutionCrossAttentionFusion


class MRYOLOInputCrossAttentionAblation(YOLOv11):
    """
    MR-YOLO ablation that fuses raw multi-resolution spectra before YOLOv11.
    """

    def __init__(
        self,
        input_resolutions: List[Tuple[int, int]],
        output_dir: str,
        num_classes: int = 80,
        reg_max: int = 16,
        device: str = "cuda:0",
        in_ch: int = 1,
        width_mult: float = 0.25,
        fusion_mode: str = "deformable",
        center_resolution_index: int | None = None,
        fusion_d_model: int = 128,
        fusion_num_heads: int = 4,
        fusion_num_layers: int = 1,
        fusion_num_points: int = 4,
        fusion_ffn_ratio: float = 2.0,
        fusion_dropout: float = 0.0,
        debug: bool = False,
        encoder_channels: int | None = None,
    ):
        if not input_resolutions:
            raise ValueError("input_resolutions must contain at least one resolution.")

        num_resolutions = len(input_resolutions)
        if center_resolution_index is None:
            center_resolution_index = num_resolutions // 2
        if not 0 <= int(center_resolution_index) < num_resolutions:
            raise ValueError(
                f"center_resolution_index must be in [0, {num_resolutions - 1}], "
                f"got {center_resolution_index}."
            )
        center_resolution_index = int(center_resolution_index)
        center_hw = tuple(input_resolutions[center_resolution_index])

        super().__init__(
            output_dir=output_dir,
            num_classes=num_classes,
            strides=None,
            reg_max=reg_max,
            device=device,
            input_canals=in_ch,
            width_mult=width_mult,
            debug=debug,
            anisotropic=False,
            p3_size=(64, 64),
            input_hw=center_hw,
        )

        self.input_resolutions = list(input_resolutions)
        self.center_resolution_index = center_resolution_index
        self.in_ch = int(in_ch)
        self.last_encoded_inputs = []
        self.last_fused_input = None

        if encoder_channels is None:
            encoder_channels = max(1, int(64 * width_mult))
        encoder_channels = int(encoder_channels)
        self.input_encoders = nn.ModuleList(
            Conv(in_ch, encoder_channels, k=3, s=1)
            for _ in input_resolutions
        )
        self.input_fusion = InterResolutionCrossAttentionFusion(
            input_channels=[encoder_channels] * num_resolutions,
            out_channels=in_ch,
            d_model=fusion_d_model,
            num_heads=fusion_num_heads,
            num_layers=fusion_num_layers,
            num_points=fusion_num_points,
            ffn_ratio=fusion_ffn_ratio,
            dropout=fusion_dropout,
            fusion_mode=fusion_mode,
            center_resolution_index=center_resolution_index,
        )

        self.to(device)

    def _as_input_list(self, inputs) -> List[torch.Tensor]:
        if torch.is_tensor(inputs) and len(self.input_resolutions) == 1:
            return [inputs]
        if isinstance(inputs, tuple):
            return list(inputs)
        if isinstance(inputs, list):
            return inputs
        raise ValueError("inputs must be a list of tensors for multi-resolution inference.")

    def _validate_inputs(self, inputs: List[torch.Tensor]) -> None:
        if len(inputs) != len(self.input_resolutions):
            raise ValueError(
                f"Expected {len(self.input_resolutions)} inputs, got {len(inputs)}."
            )
        for index, (x, (height, width)) in enumerate(zip(inputs, self.input_resolutions)):
            if x.dim() != 4:
                raise ValueError(f"Input #{index} must be 4D, got {x.dim()}D.")
            if x.shape[1] != self.in_ch:
                raise ValueError(
                    f"Input #{index} has {x.shape[1]} channels, expected {self.in_ch}."
                )
            if tuple(x.shape[-2:]) != (height, width):
                raise ValueError(
                    f"Input #{index} has shape {tuple(x.shape[-2:])}, "
                    f"expected {(height, width)}."
                )

    def forward_features(self, inputs):
        inputs = self._as_input_list(inputs)
        self._validate_inputs(inputs)

        encoded = [encoder(x) for encoder, x in zip(self.input_encoders, inputs)]
        self.last_encoded_inputs = encoded

        fused = self.input_fusion(encoded)
        self.last_fused_input = fused
        self.debug_shape("input fusion", fused)

        return YOLOv11.forward_features(self, fused)
