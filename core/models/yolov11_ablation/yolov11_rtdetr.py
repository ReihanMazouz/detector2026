import torch

from ..Head.rtdetr import RTDETRHead
from ..Neck import RTDETRHybridEncoderNeck
from .yolov11_rtdetr_head import YOLOv11RTDETRHead


class YOLOv11RTDETR(YOLOv11RTDETRHead):
    """RT-DETR ablation with a YOLOv11 backbone, RT-DETR hybrid neck and RT-DETR head."""

    def __init__(
        self,
        output_dir,
        num_classes=80,
        strides=None,
        reg_max=16,
        device="cuda:0",
        input_canals=1,
        width_mult=0.25,
        debug=False,
        anisotropic=False,
        p3_size=(64, 64),
        input_hw=None,
        hidden_dim=256,
        neck_num_heads=8,
        neck_num_encoder_layers=1,
        neck_ffn_ratio=4.0,
        neck_dropout=0.0,
        neck_depth_mult=1.0,
        num_queries=300,
        num_decoder_layers=3,
        num_heads=8,
        num_decoder_points=4,
        use_deformable_attention=True,
        dim_feedforward=1024,
        dropout=0.0,
        learnt_init_query=False,
        matcher_num_threads=1,
    ):
        super().__init__(
            output_dir=output_dir,
            num_classes=num_classes,
            strides=strides,
            reg_max=reg_max,
            device=device,
            input_canals=input_canals,
            width_mult=width_mult,
            debug=debug,
            anisotropic=anisotropic,
            p3_size=p3_size,
            input_hw=input_hw,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            num_decoder_points=num_decoder_points,
            use_deformable_attention=use_deformable_attention,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            learnt_init_query=learnt_init_query,
            matcher_num_threads=matcher_num_threads,
        )

        c3 = int(256 * width_mult)
        c4 = int(512 * width_mult)
        c5 = int(1024 * width_mult)
        self.rtdetr_neck = RTDETRHybridEncoderNeck(
            in_channels=[c3, c4, c5],
            hidden_dim=hidden_dim,
            num_heads=neck_num_heads,
            num_encoder_layers=neck_num_encoder_layers,
            ffn_ratio=neck_ffn_ratio,
            dropout=neck_dropout,
            depth_mult=neck_depth_mult,
        )
        self.detect_one2one = RTDETRHead(
            in_channels=self.rtdetr_neck.out_channels,
            strides=self.strides,
            num_classes=self.num_classes,
            reg_max=self.reg_max,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            num_decoder_points=num_decoder_points,
            use_deformable_attention=use_deformable_attention,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            learnt_init_query=learnt_init_query,
        )
        self.detect_one2one.bias_init(image_size=self.input_hw if self.input_hw is not None else 1024)
        self.active_head = "one2one"
        self.criterion = self.criterion_one2one
        self._freeze_unused_yolo_neck_and_head()
        self.to(self.device)

    def _freeze_unused_yolo_neck_and_head(self):
        unused_modules = (
            self.head_c3_1,
            self.head_c3_2,
            self.down_p3,
            self.head_c3_3,
            self.down_p4,
            self.head_c3_4,
            self.detect,
        )
        for module in unused_modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward_backbone(self, x):
        x = self._prepare_input(x)

        x = self.conv1(x)
        self.debug_shape("conv1", x)
        x = self.conv2(x)
        self.debug_shape("conv2", x)

        f2 = self.c3_1(x)
        self.debug_shape("c3_1 (f2)", f2)
        x = self.conv3(f2)
        self.debug_shape("conv3", x)

        p3 = self.c3_2(x)
        self.debug_shape("c3_2 (p3)", p3)
        x = self.conv4(p3)
        self.debug_shape("conv4", x)

        p4 = self.c3_3(x)
        self.debug_shape("c3_3 (p4)", p4)
        x = self.conv5(p4)
        self.debug_shape("conv5", x)

        x = self.c3_4(x)
        x = self.sppf(x)
        p5 = self.attn(x)
        self.debug_shape("attn (p5)", p5)
        return p3, p4, p5

    def forward(self, x, head=None):
        self._last_image_hw = tuple(x.shape[-2:])
        image_size = self.input_hw if self.input_hw is not None else self._last_image_hw
        p3, p4, p5 = self.forward_backbone(x)
        p3, p4, p5 = self.rtdetr_neck(p3, p4, p5)
        return self.detect_one2one(p3, p4, p5, image_size=image_size)

    def training_forward(self, imgs):
        return self(imgs)

    def get_training_criterion(self):
        return self.criterion_one2one

    def fit(self, *args, **kwargs):
        return self._fit_one2one_rtdetr(*args, **kwargs)

    def train(self, mode=True):
        torch.nn.Module.train(self, mode)
        return self

    def load_yolov11_backbone_weights(self, weights_path: str, device="cpu", eval_mode=True):
        """Optionally initialize the YOLOv11 backbone from a standard YOLOv11 checkpoint."""
        state_dict = torch.load(weights_path, map_location=device)
        model_state = self.state_dict()
        backbone_prefixes = ("conv", "c3_", "sppf.", "attn.")
        clean_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(backbone_prefixes) and key in model_state and model_state[key].shape == value.shape:
                clean_state_dict[key] = value

        missing_keys, unexpected_keys = self.load_state_dict(clean_state_dict, strict=False)
        if eval_mode:
            self.eval()
        return missing_keys, unexpected_keys
