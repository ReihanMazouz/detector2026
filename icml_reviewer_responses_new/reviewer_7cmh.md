We thank the reviewer for their detailed and thoughtful evaluation. We address the main concerns below.

**W1. Lack of real-world evaluation**

We now include a cross-domain evaluation on a real acoustic dataset [1]. Due to space constraints, we refer the reviewer to our response to Reviewer 2hiN.  The corresponding results and qualitative examples are provided here: [acoustic_results_figures.md](https://github.com/ICMLanonymous2026/MRS_YOLO_ICML26/tree/main/icml_reviewer_responses/acoustic_results_figures.md).


**W2. Questionable necessity of the multi-resolution motivation**

From a signal detection perspective, this question can be further clarified. In classical detection theory, when the signal is unknown, the GLRT leads to a decision rule primarily based on amplitude or energy, with phase often marginalized out [2]. However, the "unknown signal hypothesis" does not hold in a learning-based setting, where neural networks implicitly learn priors from data. In this context, phase information can in principle become informative, although current neural architectures do not necessarily exploit it effectively in practice, especially in low-SNR regimes.

As a concise ablation summary, we compared three single-resolution input parameterizations with the same model YOLOv11n:

| Representation | mAP50:95 | mAP50 | Recall low SNR | Recall medium SNR | Recall high SNR | Params | FLOPs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Amplitude only | 0.3396 | 0.3999 | 0.4420 | 0.5599 | 0.6327 | 2.88M | 629.64M |
| Amplitude + phase | 0.3127 | 0.3833 | 0.4369 | 0.5551 | 0.6305 | 2.88M | 632.00M |
| Real + imaginary | 0.3077 | 0.3671 | 0.4308 | 0.5475 | 0.6239 | 2.88M | 632.00M |

This ablation indicates that, in our setting, explicitly providing the complex spectrum does not improve over amplitude alone. A more principled alternative would be to use complex-valued neural networks [3] to process the complex spectrum natively. In some settings, this can be beneficial: for example, a complex-valued version of YOLO for SAR object detection [4] reports improvements of approximately +1.4% mAP50 and +0.5% mAP50:95. However, these gains come at a significant computational cost, with FLOPs roughly doubling due to complex-valued operations. Taken together, these observations support our interpretation that the benefit of multi-resolution STFTs is not to recover information missing from a single complex spectrogram, but to provide a more useful inductive bias for robust detection. The corresponding table and training curves are provided here: [w2_ablation_figures.md](https://github.com/ICMLanonymous2026/MRS_YOLO_ICML26/tree/main/icml_reviewer_responses/w2_ablation_figures.md).

**W3. Potential over-engineering**

We agree that this ablation study was conducted exclusively on RF simulated datasets. As such, the concern regarding potential over-engineering and limited generalization is valid. To address this point, the revised version now includes evaluation on an additional **real-world acoustic dataset**.  This additional experiment provides evidence that the model is not solely tailored to simulated RF data, but can generalize to:

- Different signal modalities (RF vs acoustic),
- Different data generation processes (simulated vs real),
- Different noise conditions.

We therefore believe that, although the concern about over-engineering is legitimate, the new cross-domain evaluation supports the relevance of the proposed architectural choices beyond the initial experimental setting. The corresponding results are provided here: [acoustic_results_figures.md](https://github.com/ICMLanonymous2026/MRS_YOLO_ICML26/tree/main/icml_reviewer_responses/acoustic_results_figures.md).

**Conclusion**

We acknowledge that the concerns raised by the reviewer were valid given the initial version of the paper. However, with the addition of real-world evaluation and further clarifications, we hope that these issues have now been properly addressed.

[1] K. S. J. Kanes, "Recycling data: An annotated marine acoustic data set that is publicly available for use in classifier development and marine mammal research," *The Journal of the Acoustical Society of America*, vol. 148, p. 2595, 2020. DOI: 10.1121/1.5147208.

[2] H. Vincent Poor, *An Introduction to Signal Detection and Estimation*. Springer Science & Business Media, 2013.

[3] ChiYan Lee, Hideyuki Hasegawa, and Shangce Gao, “Complex-valued neural networks: A comprehensive survey,” *IEEE/CAA Journal of Automatica Sinica*, vol. 9, no. 8, pp. 1406-1426, 2022.

[4] Dandan Zhao, Zhe Zhang, Dongdong Lu, Xiaolan Qiu, Wei Li, Hang Li, and Yirong Wu, “CV-YOLO: A complex-valued convolutional neural network for oriented ship detection in single-polarization single-look complex SAR images,” *Remote Sensing*, vol. 17, no. 8, p. 1478, 2025.
