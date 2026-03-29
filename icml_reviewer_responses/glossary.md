# Glossary

- **CA (Channel Attention):** An attention mechanism that reweights feature channels to emphasize the most informative channel responses [1].
- **SA (Spatial Attention):** An attention mechanism that reweights spatial locations to emphasize informative regions in the feature map [1].
- **CBAM (Convolutional Block Attention Module):** An attention module that applies channel attention followed by spatial attention in sequence to refine intermediate feature maps [1].
- **SCSA (Spatial-Channel Synergistic Attention):** An attention module combining spatial and channel attention to jointly emphasize informative regions and feature channels [2].
- **SMSA (Shareable Multi-Semantic Spatial Attention):** The spatial-attention component of SCSA. It aggregates multi-semantic spatial information and provides spatial priors to guide channel recalibration [2].
- **PCSA (Progressive Channel-wise Self-Attention):** The channel-attention component of SCSA. It performs channel-wise self-attention and progressively uses the spatial priors produced by SMSA to recalibrate feature channels [2].
- **C2f:** A CSP-style convolutional block introduced in the Ultralytics YOLOv8 codebase. It splits the features, propagates part of them through successive bottlenecks, concatenates intermediate outputs, and fuses them efficiently [3, 4].
- **A2 (Area Attention):** A local attention mechanism that reduces the cost of global attention by reshaping the feature map into a small number of large spatial segments and computing attention within these areas instead of over the full map [5, 6].
- **A2C2f:** A `C2f`-based block augmented with `A2` area attention, used to increase receptive field and feature interaction at moderate computational cost in real-time detectors [5, 6].
- **C3k2:** A YOLO11 block derived from `C2f`, where the inner transformations can use `C3k` sub-blocks for efficient feature extraction while preserving a lightweight CSP-style structure [4, 7].
- **SPPF (Spatial Pyramid Pooling - Fast):** A pooling module that aggregates multi-scale contextual information using successive pooling operations with different receptive fields. It is a fast variant inspired by Spatial Pyramid Pooling [8].
- **C2PSA:** A convolutional block integrating partial self-attention mechanisms to enhance long-range dependency modeling while maintaining efficiency. Its design motivation is related to the broader use of self-attention in vision backbones [9].
- **Backbone:** The feature extraction network that processes the input and produces hierarchical feature maps.
- **FPN (Feature Pyramid Network):** A multi-scale feature aggregation structure that combines features from different levels of the backbone [10].
- **Head (Detection Head):** The final network component that predicts bounding boxes, objectness scores, and class probabilities from feature maps.

## References

- [1] Sanghyun Woo, Jongchan Park, Joon-Young Lee, and In So Kweon, *CBAM: Convolutional Block Attention Module*, ECCV, 2018.
- [2] Yunzhong Si, Huiying Xu, Xinzhong Zhu, Wenhao Zhang, Yao Dong, Yuxing Chen, and Hongbo Li, *SCSA: Exploring the Synergistic Effects Between Spatial and Channel Attention*, Neurocomputing, vol. 634, p. 129866, 2025.
- [3] Glenn Jocher, Ayush Chaurasia, and Jing Qiu, *Ultralytics YOLOv8*, version 8.0.0, 2023. Software. https://github.com/ultralytics/ultralytics
- [4] Ultralytics, *Reference for `ultralytics/nn/modules/block.py`*, official Ultralytics documentation sourced from the Ultralytics codebase. This reference documents both `C2f` and `C3k2`. https://github.com/ultralytics/ultralytics/blob/main/docs/en/reference/nn/modules/block.md
- [5] Yunjie Tian, Qixiang Ye, and David Doermann, *YOLO12: Attention-Centric Real-Time Object Detectors*, arXiv preprint arXiv:2502.12524, 2025.
- [6] Yunjie Tian, Qixiang Ye, and David Doermann, *YOLO12: Attention-Centric Real-Time Object Detectors*, software, 2025. https://github.com/sunsmarterjie/yolov12
- [7] Glenn Jocher and Jing Qiu, *Ultralytics YOLO11*, version 11.0.0, 2024. Software. https://github.com/ultralytics/ultralytics
- [8] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, *Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition*, IEEE TPAMI, 2015.
- [9] Aravind Srinivas et al., *BoTNet: Bottleneck Transformers for Visual Recognition*, CVPR, 2021.
- [10] Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie, *Feature Pyramid Networks for Object Detection*, CVPR, 2017.
