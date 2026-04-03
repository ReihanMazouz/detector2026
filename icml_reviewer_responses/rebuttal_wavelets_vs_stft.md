We agree that this point should be stated more carefully. Our approach is not *multi-resolution analysis* in the classical wavelet sense, and we will revise the wording to avoid that ambiguity. A more accurate description is that the method learns jointly from several complementary time-frequency representations within a single detector.

We also agree that multiple scalograms could in principle be used as parallel inputs in a similar way. This does not contradict our approach; rather, it suggests that the architecture is not restricted to STFTs.

Our choice of multiple STFTs is mainly motivated by practical and representational reasons:

- STFT branches preserve a common and uniform time-frequency geometry, which makes spatial alignment, fusion, and detection more straightforward.
- They remain computationally efficient through FFT-based implementations.
- They are easier to interpret in terms of signal duration and bandwidth.
- They avoid the additional design choice of selecting one or several mother wavelets, which becomes nontrivial when the signal class is unknown or heterogeneous.

Therefore, our claim is not to introduce a new form of classical multi-resolution analysis, but to show that jointly learning from several complementary time-frequency views within a one-stage detector is effective for this class of detection problems.
