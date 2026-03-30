# Response to Reviewer hRjP

We thank the reviewer for the detailed and technically insightful feedback.

## Q5 -> Q13: About the Oracle

We agree that the original description of the Oracle-OR lacked clarity. In the revised version, we will provide a full algorithmic definition (see the standalone file [oracle_or_algorithm.md](https://github.com/ReihanMazouz/detector2026/blob/main/icml_reviewer_responses/oracle_or_algorithm.md)).

### Formalization of the Oracle

Let $\{\mathcal{P}_r\}_{r=1}^{R}$ be the post-NMS prediction sets from the $R$ single-resolution models, and $\mathcal{G}=\{(b_g,c_g)\}$ the ground-truth set.

For each ground-truth instance $g=(b_g,c_g)$, we define the candidate set:

\[
\mathcal{M}_g = \left\{ p=(b_p,c_p,s_p)\in \bigcup_{r=1}^{R}\mathcal{P}_r \;:\; \mathrm{IoU}(b_p,b_g)\geq \tau \right\}.
\]

We then define the correctly classified subset:

\[
\mathcal{M}_g^{\mathrm{cls}} = \left\{ p\in \mathcal{M}_g \;:\; c_p = c_g \right\}.
\]

If $\mathcal{M}_g^{\mathrm{cls}}\neq\emptyset$, we retain this subset otherwise, we keep $\mathcal{M}_g$. The selected prediction is:

$$
p^*(g) = \arg\max_{p\in\mathcal{M}_g} \mathrm{IoU}(b_p,b_g).
$$

Predictions that do not intersect any ground-truth instance are treated as false alarms, with an additional redundancy filtering step applied to avoid counting overlapping false detections multiple times.

### Interpretation: recall and classification upper bounds

Under this definition, the oracle provides:

- A **recall upper bound**: a ground-truth instance is detected if *any* single-resolution model detects it.

- A **classification upper bound (conditional on detection)**: when evaluating classification accuracy (e.g., confusion matrices), only matched predictions are considered. Since the oracle retains a correctly classified prediction whenever one exists in $\mathcal{M}_g$, it achieves the best possible classification performance given the detections.

- **No upper bound on mAP**: although the oracle maximizes recall by construction, it aggregates false positives from all resolutions. An additional IoU-based suppression step removes part of the redundant detections, but the total number of false positives increases, which degrades precision.

\[
\mathrm{Precision}_{\mathrm{oracle}} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}_{\mathrm{oracle}}}.
\]

Since mAP depends on the full precision-recall curve and on the ranking induced by confidence scores (which remain unchanged), **the oracle is not guaranteed to maximize mAP**.

### Conclusion and discussion of Q5--Q13

We believe that the revised formulation and detailed specification of the Oracle-OR address most of the concerns raised in questions Q5 to Q13.

**Regarding Q5.**  
The reviewer correctly states that a fusion model should not outperform a true instance-level oracle unless it extracts additional information beyond what is present in individual models. This is precisely the phenomenon we aim to highlight. The proposed MRS-YOLO model does not simply select the best detection across resolutions; it jointly exploits multi-resolution representations.

In particular, the model benefits from observing the same signal under multiple time-frequency resolutions, which improves confidence calibration. This can reduce false alarms, or, at a fixed false alarm rate, increase detection probability. Moreover, the experiment on Dataset C explicitly demonstrates that the model can leverage inter-resolution correlations to discriminate between waveform patterns that are intentionally indistinguishable at a single resolution.

**Regarding Q11 and Q12.**  
We hope that the revised oracle definition clarifies the handling of duplicate detections, confidence scores, and aggregation. It is important to emphasize that the oracle, by construction, exhibits a higher false alarm rate than individual models.

This increased false alarm rate does not impact recall nor classification accuracy (as measured by confusion matrices), but it does affect $\mathrm{mAP}_{50}$ and $\mathrm{mAP}_{50:95}$. Therefore, mAP comparisons involving the oracle must be interpreted with caution.

Regarding alternative strategies such as applying a global NMS over merged predictions, we refer the reviewer to our response to Reviewer **fxsb**. Such approaches were explored (and are part of ongoing work accepted to ICASSP 2026, not cited here due to double-blind constraints). A key difficulty is that confidence scores across single-resolution models are not calibrated consistently. As a result, applying a global NMS tends to favor predictions with higher raw confidence, which may not correspond to the most reliable detections across resolutions.

In contrast, the proposed oracle avoids introducing additional bias from score aggregation and provides a controlled reference for analyzing recall and classification performance, while, as described by the reviewer, mAP comparisons must be interpreted carefully.

## Q4: STFT vs wavelets and role of the spectrogram

Time-frequency analysis is fundamentally constrained by the Heisenberg trade-off: time and frequency resolution cannot both be arbitrarily precise [1, 2]. Different transforms therefore correspond to different tilings of the time-frequency plane.

Wavelets induce a logarithmic tiling, with finer temporal resolution at high frequencies and finer frequency resolution at low frequencies [2]. This is well suited to some multi-scale signals, but not universally optimal: it implicitly favors signals whose relevant structures follow that hierarchy.

More generally, no time-frequency representation avoids the underlying trade-off; they differ only in how it is distributed [2]. This is why combining several representations is a natural way to capture complementary signal characteristics.

We therefore use multiple STFTs. This choice retains a harmonic basis with clear physical meaning in terms of local spectral content, while remaining computationally efficient thanks to FFT-based implementations.

- [1] Karlheinz Gröchenig, *Foundations of Time-Frequency Analysis*, Birkhäuser, 2001.
- [2] Stéphane Mallat, *A Wavelet Tour of Signal Processing: The Sparse Way*, 3rd edition, Academic Press, 2009.

## Responses to Q1--Q3 and Q14--Q15

### Q1: Why not raw-domain modeling?

Raw-domain approaches (e.g., I/Q) are possible but lack explicit spatial structure, making localization, especially along the frequency axis, more difficult. Time-frequency representations introduce a strong inductive bias that facilitates detection. In addition, raw I/Q data typically involve significantly higher data volumes, making lightweight detection frameworks less practical.

### Q2: Why five resolutions?

We refer the reviewer to our response to reviewer **7cmh**.

### Q3: Ground truth

Ground-truth boxes are generated from simulation. This will be clarified. We will also include a discussion based on additional results obtained on a real acoustic dataset with manualy annotated ground-truth boxes. The corresponding note is available here: [real_acoustic_data_results.md](https://github.com/ReihanMazouz/detector2026/blob/main/icml_reviewer_responses/real_acoustic_data_results.md).

### Q14 and Q15: Nature of the contribution

The contribution is primarily **architectural and representational**, centered on time-frequency detection. Rather than being application-specific (e.g., RF or acoustic), it is driven by the general problem of detecting structures in time-frequency representations. More generally, the approach can be applied to any setting where multiple complementary representations can be fused within a detection framework.