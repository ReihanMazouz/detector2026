__all__ = [
    "BboxLoss",
    "DFLoss",
    "SNRYOLODetectionLoss",
    "YOLODetectionLoss",
    "YOLOOne2OneHungarianLoss",
]


def __getattr__(name):
    if name == "BboxLoss":
        from .bbox import BboxLoss

        return BboxLoss
    if name == "DFLoss":
        from .dfl import DFLoss

        return DFLoss
    if name == "SNRYOLODetectionLoss":
        from .snr_yolo_detection import SNRYOLODetectionLoss

        return SNRYOLODetectionLoss
    if name == "YOLODetectionLoss":
        from .yolo_detection import YOLODetectionLoss

        return YOLODetectionLoss
    if name == "YOLOOne2OneHungarianLoss":
        from .yolo_one2one_hungarian import YOLOOne2OneHungarianLoss

        return YOLOOne2OneHungarianLoss
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
