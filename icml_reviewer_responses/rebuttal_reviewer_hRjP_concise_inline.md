We understand that the reviewer’s concern stems from the choice of multiple-resolution spectrograms over wavelets. Our interpretation is summarized below:

---

**Does the built-in multi-resolution nature of wavelets eliminate the need for multiple complementary representations, in particular multiple STFT resolutions?**  
**No.** In a wavelet scalogram, the local resolution varies with frequency across the time-frequency plane, still under the Heisenberg uncertainty principle, [A Wavelet Tour of Signal Processing, chapter 2, section 3.2, pp. 30-32]. However, each given time-frequency location is still observed under only one local resolution, whereas multiple STFT resolutions provide several complementary views of that same local pattern.

---

**Is the proposed methodology limited to STFT representations, or could it also be applied to multiple time-frequency representations, in particular multiple wavelet-based ones?**  
**Yes.** It can in principle be applied to any family of time-frequency representations, including multiple wavelet-based ones. We remain cautious on this point because, with well-calibrated STFTs, the same signal tends to appear at the same spatial location across branches once they reach the fusion stage. This is particularly consistent with our fusion design, which combines channel concatenation, attention mechanisms including spatial attention, and subsequent filtering. This exact alignment is not necessarily preserved with other families of transforms. That said, based on our expertise, we cautiously believe that the model can still learn such scale differences and may not be fundamentally limited by this challenge.

---

**In the application settings considered in this paper, is the wavelet transform better suited than the STFT or than a set of STFTs with different resolutions?**  
**Not clearly.** In the datasets considered here, relevant structures may be distributed across both time and frequency without following a strictly scale-dependent organization, so the logarithmic tiling induced by wavelets is not necessarily aligned with the underlying signal structure. In addition, wavelet-based representations depend on design choices such as the mother wavelet and the transform parameters, all of which influence the resulting time-frequency structure. When signals are largely unknown or heterogeneous, as in our case, these choices are not straightforward to calibrate. By contrast, STFTs are more standard, more intuitive to interpret, and typically faster to compute thanks to optimized FFT implementations.

---

As the reviewer suggests, we propose to clarify these points more explicitly in the revised version.
