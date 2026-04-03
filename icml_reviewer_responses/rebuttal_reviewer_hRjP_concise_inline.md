We understand that the reviewer’s concern stems from the choice of multiple-resolution spectrograms over wavelets. Our interpretation is summarized below:

---

**Does the built-in multi-resolution nature of wavelets eliminate the need for multiple complementary representations, in particular multiple STFT resolutions?**  
**No.** Wavelets redistribute resolution within one representation, still under the Heisenberg uncertainty principle, [A Wavelet Tour of Signal Processing, chapter 2, section 3.2, pp. 30-32]. They do not provide multiple resolutions of the same time-frequency location, which is precisely what multiple STFT resolutions provide.

---

**Is the proposed methodology limited to STFT representations, or could it also be applied to multiple time-frequency representations, in particular multiple wavelet-based ones?**  
**Yes.** It can in principle be applied to any family of time-frequency representations, including multiple wavelet-based ones. We remain cautious on this point because, with well-calibrated STFTs, the same signal tends to appear at the same spatial location across branches once they reach the fusion stage. This is particularly consistent with our fusion design, which combines channel concatenation, attention mechanisms including spatial attention, and subsequent filtering. This exact alignment is not necessarily preserved with other families of transforms. That said, based on our expertise, we cautiously believe that the model can still learn such scale differences and not be fundamentally limited by this challenge.

---

**In the application settings considered in this paper, is the wavelet transform better suited than the STFT or than a set of STFTs with different resolutions?**  
**Not clearly.** In the datasets considered here, relevant structures may be distributed across both time and frequency without following a strictly scale-dependent organization, so the logarithmic tiling induced by wavelets is not necessarily aligned with the underlying signal structure. In addition, wavelet-based representations depend on the choice of the mother wavelet, when signals are largely unknown or when several signal classes with different characteristics must be analyzed, multiple mother wavelets may be needed, which brings us back to the multiple-representation problem addressed in this paper.

---

**Concluding remark.**  
We believe that the main source of confusion is that the term *multi-resolution* is used here in a different sense than in wavelet analysis. For wavelets, it refers to a single representation with location-dependent resolution. In our paper, it refers to using several complementary representations of the same signal so that the same local pattern can be observed under different resolutions.
