from ..yolov11 import YOLOv11


class YOLOv11NoNeck(YOLOv11):
    """YOLOv11 backbone with detection heads applied directly on P3, P4 and P5."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in ("upsample", "head_c3_1", "head_c3_2", "down_p3", "head_c3_3", "down_p4", "head_c3_4"):
            if hasattr(self, name):
                delattr(self, name)

    def forward_features(self, x):
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
        self.debug_shape("p3 direct", p3)

        x = self.conv4(p3)
        self.debug_shape("conv4", x)

        p4 = self.c3_3(x)
        self.debug_shape("p4 direct", p4)

        x = self.conv5(p4)
        self.debug_shape("conv5", x)

        x = self.c3_4(x)
        x = self.sppf(x)
        p5 = self.attn(x)
        self.debug_shape("p5 direct", p5)

        return p3, p4, p5
