We thank the reviewer for the thorough evaluation and constructive feedback. We address the main concerns below.

**W1. Lack of real-world evaluation and reproducibility**

We now include a cross-domain evaluation on a real acoustic dataset [1]. Due to space constraints, we refer the reviewer to our response to Reviewer 2hiN.  The corresponding results and qualitative examples are provided here: [acoustic_results_figures.md](https://github.com/ICMLanonymous2026/MRS_YOLO_ICML26/tree/main/icml_reviewer_responses/acoustic_results_figures.md).

**W2 and Key Question 1. Evaluation baselines (decision-level ensemble vs Oracle)**

We agree that a decision-level ensemble (e.g., merging predictions from multiple single-resolution YOLO models via NMS) would constitute a more practical baseline. The most direct way to address this would be to reference related recently accepted publication where such a strategy is explored, however, doing so at this stage could compromise the double-blind review process. We will include this comparison explicitly in the revised version once anonymity constraints are lifted.

Instead, we can clarify our design choice. We intentionally introduced the Oracle baseline to isolate the contribution of multi-resolution learning. The Oracle ensures that the limitation does not stem from post-processing, but rather from the inability of single-resolution models to capture complementary information.

As also noted by Reviewer hRjP, this Oracle baseline would benefit from additional clarification. The corresponding algorithm is available here: [oracle_algorithm.pdf](https://github.com/ICMLanonymous2026/MRS_YOLO_ICML26/blob/main/icml_reviewer_responses/oracle_algorithm.pdf).

**W3 and Key Question 2. Sensitivity to the number of resolutions**

In practice, this choice is guided by domain expertise rather than a strict optimization. For Datasets A and B, prior experiments show that a window size of $2^8$ (256) performs well in the single-resolution setting. We therefore extend this configuration by adding neighboring scales, yielding $\{2^6, 2^7, 2^8, 2^9, 2^{10}\}$.

Dataset C provides partial validation: even with only two well-separated resolutions, we observe clear gains over single-resolution baselines on specific waveforms.

For the acoustic dataset, the choice is also domain-driven. Following dataset recommendations, we use the minimum and maximum window sizes ($2^{10}$ and $2^{14}$) and include intermediate powers of two to cover the range.

More generally, widely spaced resolutions can induce strong anisotropy and increase computational cost due to deeper downsampling. Conversely, intermediate resolutions improve representation continuity and, in a parallel setting, have limited impact on inference latency beyond additional STFT computations.

If time permits, we will include an **ablation study** on the number of resolutions in the revised version to further support this design choice.

**W4. Presentation and clarity**

In the revised version, we will add a **glossary in the appendix** to improve clarity.

**Key Question 3: Computational overhead and latency**

Indeed, the appendix currently reports model forward time plus NMS, but not the preprocessing needed to compute multiple STFT spectrograms. This preprocessing costs about **0.46 ms on CPU** and **0.09 ms on an H100 GPU** per spectrogram.

**In the revised version, we will make these trade-offs explicit and report end-to-end latency measurements.**

**W3 and Key Question 4. Originality, conceptual contribution, and societal impact**

We agree that the societal impact discussion should be strengthened. While the method is motivated as a general approach for time-frequency detection and is relevant to benign applications such as acoustic monitoring and environmental sensing, RF signal detection technologies may also raise dual-use concerns. In particular, such methods could be used in surveillance, intelligence, or electronic-warfare contexts, and may therefore raise privacy, security, or governance issues if deployed without appropriate safeguards and oversight. We will include this discussion directly in the revised paper and clarify that the present work is methodological in nature rather than tied to a specific deployment scenario.
