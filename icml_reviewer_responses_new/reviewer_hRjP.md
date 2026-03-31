We thank the reviewer for the detailed and technically insightful feedback.

## Q5 -> Q13: About the Oracle

The original description of the Oracle-OR lacked clarity. In the revised version, we will provide a full algorithmic definition together with its interpretation as a recall and classification upper bound; the corresponding standalone note is available here: [oracle_icml.pdf](https://github.com/ICMLanonymous2026/MRS_YOLO_ICML26/blob/main/icml_reviewer_responses/oracle_icml.pdf).

### Q5

The reviewer correctly states that a fusion model should not outperform a true instance-level oracle unless it extracts additional information beyond what is present in individual models. This is precisely the phenomenon we aim to highlight: the proposed model benefits from observing the same signal at multiple time-frequency resolutions, which improves confidence calibration. This can reduce false alarms, or, at a fixed false alarm rate (FAR), increase detection probability. Moreover, the experiment on Dataset C explicitly demonstrates that the model can leverage inter-resolution correlations to discriminate between waveform patterns that are intentionally indistinguishable at a single resolution.

### Q11 and Q12

It is important to emphasize that the oracle, by construction, exhibits a higher false alarm rate (FAR) than individual models. This increased FAR does not impact recall nor classification accuracy (as measured by confusion matrices without noise class), but it does indeed affect mAP.

As for alternative strategies, please refer to our response to Reviewer **fxsb**. Such approaches were explored but a key difficulty is that confidence scores across single-resolution models are not calibrated consistently. As a result, applying a global NMS tends to favor predictions with higher raw confidence, which may not correspond to the most reliable detections across resolutions.

In contrast, the proposed oracle avoids introducing additional bias from score aggregation and provides a controlled reference for analyzing recall and classification performance (upped bound), while mAP comparisons must be interpreted carefully.

## Q4: STFT vs wavelets and role of the spectrogram

Time-frequency analysis is constrained by the Heisenberg trade-off: time and frequency resolution cannot both be arbitrarily precise [1, 2]. Different transforms correspond to different tilings of the time-frequency plane.

Wavelets induce a logarithmic tiling, with finer temporal resolution at high frequencies and finer frequency resolution at low frequencies [2]. This is well suited to some multi-scale signals, but not universally optimal: it implicitly favors signals whose relevant structures follow that hierarchy.

More generally, no time-frequency representation avoids the underlying trade-off; they differ only in how it is distributed [2]. This is why combining several ones is a natural way to capture complementary signal characteristics.

We hence use multiple STFTs. The STFT provides a uniform tiling of the time-frequency plane, avoiding frequency-dependent bias and making it a flexible representation for detecting diverse and unknown signal structures, while remaining computationally efficient via FFT-based implementations. More generally, our approach is not restricted to STFT: it can be extended to other time-frequency representations if they better match the underlying structure of the signals of interest.

[1] Karlheinz Gröchenig, *Foundations of Time-Frequency Analysis*, Birkhäuser, 2001.

[2] Stéphane Mallat, *A Wavelet Tour of Signal Processing: The Sparse Way*, 3rd edition, Academic Press, 2009.

## Responses to Q1--Q3 and Q14--Q15

### Q1: Why not raw-domain modeling?

Raw-domain approaches are possible, but time-frequency representations make signals easier to separate from noise and interference. In practice, they improve effective SNR and signal separability, which is especially important in congested environments where several emissions may overlap. This provides a strong inductive bias for detection that is not explicit in raw I/Q data.

### Q2: Why five resolutions?

The number of resolutions was chosen as a domain-driven compromise between representational coverage and computational cost. In practice, we start from a window size that already performs well in the single-resolution setting and add neighboring scales to capture complementary time-frequency structures.

### Q3: Ground truth

Ground-truth boxes are generated from the known parameters used to generate the signals. This will be clarified.

### Q14 and Q15: Nature of the contribution

The contribution is primarily architectural, centered on time-frequency detection. Rather than being application-specific (e.g., RF or acoustic), it is driven by the general problem of detecting structures in time-frequency representations. More generally, the approach can be applied to any setting where multiple complementary representations can be fused within a detection framework.
