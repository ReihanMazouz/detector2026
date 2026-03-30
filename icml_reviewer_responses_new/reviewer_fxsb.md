# Response to Reviewer fxsb

We thank the reviewer for the thorough evaluation and constructive feedback. We address the main concerns below.

## W1. Lack of real-world evaluation and reproducibility

We now include a cross-domain evaluation on a real acoustic dataset. The corresponding results and qualitative examples are provided here: [acoustic_results_figures.md](./acoustic_results_figures.md).

## W2 and Key Question 1. Evaluation baselines (decision-level ensemble vs Oracle)

We agree that a decision-level ensemble (e.g., merging predictions from multiple single-resolution YOLO models via NMS) would constitute a more practical baseline. Related merged-prediction comparisons have been explored separately, but including them explicitly at this stage could compromise the double-blind review process. We will include this comparison in the revised version once anonymity constraints are lifted.

Instead, we clarify our design choice. We intentionally introduced the **Oracle-OR** baseline, despite its impracticality, to isolate the contribution of multi-resolution learning. The Oracle ensures that the limitation does not stem from post-processing, but rather from the inability of single-resolution models to capture complementary information.

As also noted by Reviewer hRjP, this Oracle baseline would benefit from additional clarification. We provide this clarification directly in the revised version.

## W3 and Key Question 2. Sensitivity to the number of resolutions

In practice, this choice is guided by domain expertise rather than a strict optimization. For Datasets A and B, prior experiments show that a window size of $2^8$ (256) performs well in the single-resolution setting. We therefore extend this configuration by adding neighboring scales, yielding $\{2^6, 2^7, 2^8, 2^9, 2^{10}\}$.

Dataset C provides partial validation: even with only two well-separated resolutions, we observe clear gains over single-resolution baselines on specific waveforms.

For the acoustic dataset, the choice is also domain-driven. Following dataset recommendations, we use the minimum and maximum window sizes ($2^{10}$ and $2^{14}$) and include intermediate powers of two to cover the range.

More generally, widely spaced resolutions can induce strong anisotropy and increase computational cost due to deeper downsampling. Conversely, intermediate resolutions improve representation continuity and, in a parallel setting, have limited impact on inference latency beyond additional STFT computations.

If time permits, we will include an **ablation study** on the number of resolutions in the revised version to further support this design choice.

## W4. Presentation and clarity

In the revised version, we will add a **glossary in the appendix** to improve clarity.

## Key Question 3: Computational overhead and latency

The appendix currently reports model forward time plus NMS, but not the preprocessing needed to compute multiple STFT spectrograms. This preprocessing costs about **0.46 ms on CPU** and **0.09 ms on an H100 GPU** per spectrogram.

The end-to-end comparison depends on the baseline:

- Compared with one YOLO on one spectrogram, `MRS-YOLO` has additional preprocessing and is therefore slower end-to-end.
- A more relevant comparison is against five single-resolution YOLO models. In that case, the runtimes become comparable:
  `MRS-YOLO: 0.792 ms + 5 x 0.09 ms`
  `5-model ensemble: 5 x 0.112 ms + 5 x 0.09 ms`
- The factor `5` is the number of resolutions and can be parallelized.

**In the revised version, we will make these trade-offs explicit and report end-to-end latency measurements.**

## W3 and Key Question 4. Originality, conceptual contribution, and societal impact

The contribution is primarily conceptual and architectural. The goal is not simply to aggregate several single-resolution detectors, but to learn a joint representation from complementary time-frequency resolutions within a single detection model.

We also agree that the societal impact discussion can be strengthened. A revised societal impact section will be included directly in the paper.
