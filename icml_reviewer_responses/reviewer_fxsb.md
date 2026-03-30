# Response to Reviewer fxsb

We thank the reviewer for the thorough evaluation and constructive feedback. We address the main concerns below.

## W1. Lack of real-world evaluation and reproducibility

Se référer à la réponse faite au reviewer 2hiN

## W2 and Key Question 1. Evaluation baselines (decision-level ensemble vs Oracle)

We agree that a decision-level ensemble (e.g., merging predictions from multiple single-resolution YOLO models via NMS) would constitute a more practical baseline. The most direct way to address this would be to reference a recent accepted work (ICASSP 2026) where such a strategy is explored, however, citing this work at this stage would compromise the double-blind review process.

Instead, we clarify our design choice. We intentionally introduced the **Oracle-OR** baseline, despite its impracticality, to isolate the contribution of multi-resolution learning. The Oracle ensures that the limitation does not stem from post-processing, but rather from the inability of single-resolution models to capture complementary information.

As also noted by Reviewer hRjP, this Oracle baseline would benefit from additional clarification. We provide further implementation details in our response to that reviewer.

In the revised version, we will reference the ICASSP 2026 work once anonymity constraints are lifted to include the **NMS-based merging** comparison.

## W3. Originality and conceptual contribution

à rédiger ...

## W3 and Key Question 2. Sensitivity to the number of resolutions

We agree that the choice of the number of resolutions requires further justification.

In practice, this choice is guided by domain expertise rather than a strict optimization. For Datasets A and B, prior experiments show that a window size of $2^8$ (256) performs well in the single-resolution setting. We therefore extend this configuration by adding neighboring scales, yielding $\{2^6, 2^7, 2^8, 2^9, 2^{10}\}$. Powers of two are standard in signal processing, ensuring efficient FFT computation and consistent frequency binning.

While heuristic, this design does not affect the core methodological contribution. Dataset C provides partial validation: even with only two well-separated resolutions, we observe clear gains over single-resolution baselines on specific waveforms.

For the acoustic dataset, the choice is also domain-driven. Following dataset recommendations, we use the minimum and maximum window sizes ($2^{10}$ and $2^{14}$) and include intermediate powers of two to cover the range.

More generally, widely spaced resolutions can induce strong anisotropy and increase computational cost due to deeper downsampling. Conversely, intermediate resolutions improve representation continuity and, in a parallel setting, have limited impact on inference latency beyond additional STFT computations.

If time permits, we will include an **ablation study** on the number of resolutions in the revised version to further support this design choice.

## W4. Presentation and clarity

We agree that the use of architecture-specific acronyms may hinder readability for a broader audience. In the revised version, we will add a **glossary in the appendix** to improve clarity. A draft is available here: [glossary.md](https://github.com/ReihanMazouz/detector2026/blob/main/icml_reviewer_responses/glossary.md).

## Key Question 3: Computational overhead and latency

The appendix currently reports model forward time plus NMS, but not the preprocessing needed to compute multiple STFT spectrograms. This preprocessing costs about **0.46 ms on CPU** and **0.09 ms on an H100 GPU** per spectrogram.

The end-to-end comparison depends on the deployment setting:

- **Single-resolution baseline:**  
  Compared with one YOLO operating on a single spectrogram, MRS-YOLO introduces extra preprocessing and may be up to about $8\times$ slower end-to-end. In such simple settings, the multi-resolution design is not always justified. This issue is amplified for highly anisotropic spectrograms, where isotropic models may also struggle to localize fine structures.

- **Multiple isotropic models baseline:**  
  A more relevant comparison is against several single-resolution YOLO models, each processing one spectrogram. In that case, the runtimes become comparable:

  $$
  \text{MRS-YOLO: } 0.792\,\text{ms} + 5 \times 0.09\,\text{ms}
  \quad \text{vs} \quad
  \text{Multi-model: } 5 \times 0.112\,\text{ms}  \text{(Oracle processing time =0)}+ 5 \times 0.09\,\text{ms}
  $$

  where the factor 5 is the number of resolutions and can be parallelized.

- **Multiple anisotropic models:**  
  If single-resolution models must handle highly anisotropic spectrograms, they may require deeper backbones, making their effective cost closer to that of MRS-YOLO.

**In the revised version, we will make these trade-offs explicit and report end-to-end latency measurements.**

A dedicated section reporting processing times has also been introduced for the acoustic dataset.

## Key Question 4: Societal impact

We agree that the societal impact discussion can be strengthened.

For readability, we moved the full proposed reformulation to the standalone note [societal_impact_reformulation.md](https://github.com/ReihanMazouz/detector2026/blob/main/icml_reviewer_responses/societal_impact_reformulation.md).
