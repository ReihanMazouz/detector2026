# Detailed Response to Reviewer 7cmh on W2

## W2. Questionable necessity of the multi-resolution motivation

We agree that, in theory, the complex-valued STFT representation preserves the full information content of the original signal. This naturally raises the question of whether the choice of STFT window size truly impacts the representation, and whether a single complex spectrogram should be sufficient.

From a signal detection perspective, this question can be further clarified. In classical detection theory, when dealing with an unknown signal embedded in noise, the Generalized Likelihood Ratio Test (GLRT) leads to a decision rule based primarily on the signal amplitude, without explicitly exploiting phase information, as detailed in *H. Poor, “An Introduction to Signal Detection and Estimation,” Springer, 1994*. This suggests that, under a fully unknown signal hypothesis, phase may not be necessary for detection. However, this assumption no longer holds in a learning-based framework. A trained neural network implicitly incorporates prior knowledge from the data distribution, relaxing the “unknown signal” hypothesis. In this context, phase information can become informative.

Following this reasoning, one may consider directly processing the complex spectrogram instead of introducing a multi-resolution framework. Two main strategies are commonly explored in the literature:

- **(i) Real-valued networks using real and imaginary parts as separate channels.**

  In this approach, the complex spectrogram is decomposed into its real and imaginary components, which are treated as two input channels for a standard CNN. This paradigm is widely adopted as it preserves compatibility with real-valued architectures and benefits from stable training procedures.

  For instance, in *“Complex Spectral Mapping With Attention Based Convolution Recurrent Neural Network for Speech Enhancement”*, the authors show that attention mechanisms can significantly improve performance by filtering out irrelevant features and enhancing informative patterns. Similarly, in *“Complex Spectrogram Enhancement by Convolutional Neural Network with Multi-Metrics Learning”*, real and imaginary spectrograms are explicitly treated as separate channels.

  However, these works also highlight an important limitation: while the real and imaginary parts exhibit structured patterns that can be exploited by CNNs, the phase itself remains difficult to learn directly, especially in low SNR regimes. In such conditions, the noisy phase can significantly deviate from the clean phase, making it challenging for neural networks to learn consistent phase relationships. As a result, even when reconstructing real and imaginary components, the benefit of explicitly modeling phase remains limited in practice.

  In our work, we also evaluated this approach by feeding amplitude and phase (or equivalently real and imaginary components) as separate channels. However, we did not observe significant performance improvements compared to magnitude-only representations (see Table~X), suggesting that simply exposing phase information is not sufficient for effective exploitation in our detection setting.

- **(ii) Complex-valued neural networks.**

  A more principled alternative consists in designing neural networks that operate directly in the complex domain. As discussed in *“Complex-Valued Neural Networks: A Comprehensive Survey” by ChiYan Lee and Hideyuki Hasegawa*, such models offer a theoretically appealing framework but introduce practical challenges. In particular, they suffer from numerical instability during training.

  In the context of object detection, *“CV-YOLO: A Complex-Valued Convolutional Neural Network for Oriented Ship Detection in Single-Polarization Single-Look Complex SAR Images” by Dandan Zhao et al.* proposes a complex-valued extension of YOLO for SAR image analysis. Their results indicate that complex-valued modeling can provide improvements (approximately +1.4% mAP50 and +0.5% mAP50:95) compared to real-valued models with a similar number of parameters. However, this gain comes at a significant computational cost, with FLOPs roughly doubling due to complex-valued operations.

Overall, these observations suggest that while complex representations are theoretically complete and potentially beneficial, current convolutional neural networks struggle to effectively exploit phase information in practice, especially in noisy conditions. By leveraging multiple time-frequency resolutions, we expose the network to complementary structures that are more readily exploitable than raw phase information. Therefore, the proposed approach does not contradict the theoretical completeness of the STFT, but instead introduces a **meaningful inductive bias** that improves learning efficiency and robustness in realistic detection scenarios.
