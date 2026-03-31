We thank the reviewer for their careful reading and constructive feedback.

**Weakness 1.**

We agree that evaluating on public datasets would strengthen the paper.

Regarding **CageDroneRF**, we have recently obtained access to the dataset. However, the remaining time before the deadline is too limited to conduct a thorough and reliable evaluation.

Concerning **RadDet**, although it is a relevant benchmark, it cannot be directly used in our framework. The dataset provides only PNG spectrograms rather than raw IQ signals, preventing the recomputation of multi-resolution representations required by our method. Additionally, the limited dynamic range of PNG encoding is not well-suited for low-SNR scenarios.

To address this limitation more meaningfully, we include in the revised version a **cross-domain evaluation** on real-world acoustic data: *Ocean Networks Canada (ONC) hydrophone dataset*, publicly available and annotated for marine mammal detection [1].

This dataset consists of real signals and provides a meaningful test of generalization beyond simulated RF data. A summary of the detection results is reported below:

| Model | Initial R@0.9P | Initial mAP50 | Initial mAP50:95 | Moderate Noise R@0.9P | Moderate Noise mAP50 | Moderate Noise mAP50:95 | Strong Noise R@0.9P | Strong Noise mAP50 | Strong Noise mAP50:95 | Very Strong Noise R@0.9P | Very Strong Noise mAP50 | Very Strong Noise mAP50:95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO (512) | 0.934 | 0.925 | 0.897 | 0.409 | 0.224 | 0.217 | 0.236 | 0.133 | 0.128 | 0.046 | 0.002 | 0.002 |
| YOLO (1024) | 0.921 | 0.903 | 0.876 | 0.374 | 0.408 | 0.371 | 0.242 | 0.268 | 0.248 | 0.126 | 0.102 | 0.097 |
| YOLO (2048) | 0.941 | 0.934 | 0.886 | 0.375 | 0.032 | 0.030 | 0.199 | 0.010 | 0.009 | 0.132 | 0.068 | 0.062 |
| YOLO (4096) | 0.854 | 0.833 | 0.696 | 0.280 | 0.017 | 0.016 | 0.192 | 0.147 | 0.130 | 0.109 | 0.000 | 0.000 |
| YOLO (8192) | 0.368 | 0.440 | 0.284 | 0.105 | 0.069 | 0.055 | 0.065 | 0.012 | 0.011 | 0.035 | 0.019 | 0.016 |
| **MRS-YOLO** | **0.962** | **0.956** | **0.931** | **0.591** | **0.589** | **0.563** | **0.463** | **0.458** | **0.431** | **0.247** | **0.008** | **0.007** |

The corresponding confusion matrices and qualitative examples are provided here: [acoustic_results_figures.md](https://github.com/ICMLanonymous2026/MRS_YOLO_ICML26/tree/main/icml_reviewer_responses/acoustic_results_figures.md).

**This additional experiment will represent a significant contribution to the revised version of the paper, as it demonstrates that our model is able to maintain superior performance in a fundamentally different setting involving real-world data.**

**Weakness 2.**

The corresponding visualization will be included in the revised version. It can be found here: [dataset_b_confusion_matrices.md](https://github.com/ICMLanonymous2026/MRS_YOLO_ICML26/tree/main/icml_reviewer_responses/dataset_b_confusion_matrices.md).

**Weakness 3.**

A detailed description of each waveform will be provided in the **appendix**.

**Key Question: Translation invariance and SCSA on spectrograms.**

We agree that spectrograms do not exhibit translation invariance, especially along the frequency axis, unlike natural images. However, the SCSA block does not inherently require such invariance in our setting.

Its use is primarily motivated by ablation results, but it can also be interpreted intuitively in our framework:

- The goal of SCSA is to improve the **fusion of multi-resolution feature spaces** after concatenation and before filtering.
- **Channel attention** is a natural choice here because reweighting channels before the subsequent channel filtering stage helps the model emphasize the most useful resolution-dependent features at the right moment in the pipeline.
- **Spatial attention**, applied beforehand, emphasizes relevant time-frequency regions. At this stage, all feature maps are aligned on a common spatial grid: a signal located at $(x,y)$ in one resolution remains localized at the same position across all resolutions. Therefore, applying spatial attention around this location is consistent and meaningful.

In this context, spatial attention acts as a **location-dependent weighting** and do not rely on any assumption of translation invariance.

**References.**

[1] K. S. J. Kanes, "Recycling data: An annotated marine acoustic data set that is publicly available for use in classifier development and marine mammal research," *The Journal of the Acoustical Society of America*, vol. 148, p. 2595, 2020. DOI: 10.1121/1.5147208.
