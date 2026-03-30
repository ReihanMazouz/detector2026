# Waveform Descriptions

## LFM

LFM stands for **Linear Frequency Modulation**. It is the classical linear chirp radar waveform, with instantaneous frequency varying linearly over time [1].

\[
s(t) = A e^{j\left(2\pi f_c t + k \pi t^2 + \varphi_0\right)}, \quad 0 < t \leq T_s
\]

<!-- ## QFM

\[
s(t) = A e^{j\left(2\pi f_c t + \frac{8B}{3T_s^2} \pi \left(t - \frac{T_s}{2}\right)^3\right)}, \quad 0 < t \leq T_s
\]
-->

## NLFM

NLFM stands for **Non-Linear Frequency Modulation**. Unlike LFM, the instantaneous frequency law is shaped nonlinearly to improve spectral containment or sidelobe behavior after pulse compression [1].

\[
s(t) = A e^{j\left(2\pi f_c t + \phi(t) + \varphi_0\right)}, \quad 0 < t \leq T_s
\]

\[
\phi(t) = -\pi f_c T_s - \frac{2\pi B T_s}{4\beta \tan(\beta)} \log\!\left[\cos\!\left(\frac{2\beta}{T_s}\left(t - \frac{T_s}{2}\right)\right)\right]
\]

\[
\beta = \arctan(\alpha), \quad \alpha \in [3.5,\,10]
\]

## BPSK

BPSK stands for **Binary Phase-Shift Keying**. In this context, it denotes a binary phase-coded waveform whose chips alternate between two phase states [1].

\[
s(t) = A e^{j(2\pi f_c t + \varphi(t) + \varphi_0)}, \quad 0 < t \leq T_s
\]

\[
\varphi(t) = \sum_{i=1}^{N_c} b_i \pi \cdot r\!\left(\frac{t - i\Delta t}{\Delta t}\right), 
\quad b_i \in \{0,1\}
\]

## Frank

The Frank code is a **polyphase coded waveform** constructed from an `N_p \times N_p` phase matrix. It is one of the standard polyphase pulse-compression codes described in radar waveform design references [2].

\[
s(t) = A e^{j(2\pi f_c t + \varphi(t) + \varphi_0)}, \quad 0 < t \leq T_s
\]

\[
\varphi(t) = \sum_{i=1}^{N_p^2} \varphi_{pq} \cdot r\!\left(\frac{t - i\Delta t}{\Delta t}\right)
\]

\[
\varphi_{pq} = \frac{2\pi}{N_p}(p-1)(q-1), \quad 1 \leq p,q \leq N_p
\]

## P1

P1 is a **polyphase radar code** derived from the Frank family, designed to provide structured phase variation across chips for pulse compression [2].

\[
s(t) = A e^{j(2\pi f_c t + \varphi(t) + \varphi_0)}, \quad 0 < t \leq T_s
\]

\[
\varphi(t) = \sum_{i=1}^{N_p^2} \varphi_{pq} \cdot r\!\left(\frac{t-i\Delta t}{\Delta t}\right)
\]

\[
\varphi_{pq} = -\frac{\pi}{N_p}(N_p-2q+1)\bigl((q-1)N_p + p - 1\bigr),
\quad 1 \leq p,q \leq N_p
\]

## P2

P2 is another **polyphase code** closely related to P1, with a different phase law chosen to preserve good ambiguity and sidelobe properties [2].

\[
s(t) = A e^{j(2\pi f_c t + \varphi(t) + \varphi_0)}, \quad 0 < t \leq T_s
\]

\[
\varphi(t) = \sum_{i=1}^{N_p^2} \varphi_{pq} \cdot r\!\left(\frac{t-i\Delta t}{\Delta t}\right)
\]

\[
\varphi_{pq} = -\frac{\pi}{2N_p}(2p-1-N_p)(2q-1-N_p),
\quad 1 \leq p,q \leq N_p
\]

## P3

P3 is a **polyphase pulse-compression code** with a quadratic phase progression across chips [2].

\[
s(t) = A e^{j(2\pi f_c t + \varphi(t) + \varphi_0)}, \quad 0 < t \leq T_s
\]

\[
\varphi(t) = \sum_{i=1}^{N_c} \varphi_i \cdot r\!\left(\frac{t - i\Delta t}{\Delta t}\right)
\]

\[
\varphi_i = \frac{\pi}{N_c}(i-1)^2
\]

## P4

P4 is another **polyphase code** in the same family, differing from P3 by an additional linear phase term [2].

\[
s(t) = A e^{j(2\pi f_c t + \varphi(t) + \varphi_0)}, \quad 0 < t \leq T_s
\]

\[
\varphi(t) = \sum_{i=1}^{N_c} \varphi_i \cdot r\!\left(\frac{t - i\Delta t}{\Delta t}\right)
\]

\[
\varphi_i = \frac{\pi}{N_c}(i-1)^2 - \pi(i-1)
\]

## T1

T1 is a **discrete polyphase waveform** obtained by quantizing an underlying continuous phase law into `M` phase states [2].

\[
s(t) = A e^{j(2\pi f_c t + \varphi_{T1}(t) + \varphi_0)}, \quad 0 < t \leq T_s
\]

Let \(K = N_c\) and let \(M\) denote the number of phase states.  
For the \(j\)-th segment, \(j=0,\dots,K-1\), define

\[
\phi_j(t) = \frac{2\pi}{M}
\left\lfloor
\left(Kt - jT_s\right)\frac{jM}{T_s}
\right\rfloor
\]

for \(t \in \left[\frac{jT_s}{K}, \frac{(j+1)T_s}{K}\right)\).  

\[
\varphi_{T1}(t) =
\begin{cases}
0, & \mathrm{mod}(\phi_j(t),2\pi) < \pi \\
\pi, & \mathrm{otherwise}
\end{cases}
\]

## T2

T2 is another **discrete polyphase waveform** based on a quantized piecewise phase construction [2].

\[
s(t) = A e^{j(2\pi f_c t + \varphi_{T2}(t) + \varphi_0)}, \quad 0 < t \leq T_s
\]

Let \(K = N_c\) and \(M\) be the number of phase states.  

\[
\phi_j(t) = \frac{2\pi}{M}
\left\lfloor
\left(Kt - jT_s\right)\left(\frac{2j-K+1}{T_s}\right)\frac{M}{2}
\right\rfloor
\]

\[
\varphi_{T2}(t) =
\begin{cases}
0, & \mathrm{mod}(\phi_j(t),2\pi) < \pi \\
\pi, & \mathrm{otherwise}
\end{cases}
\]

## T3

T3 is a **quantized polyphase waveform** derived from a quadratic phase law [2].

\[
s(t) = A e^{j(2\pi f_c t + \varphi_{T3}(t) + \varphi_0)}, \quad 0 < t \leq T_s
\]

\[
\phi(t) = \frac{2\pi}{M}
\left\lfloor
\frac{M \Delta F}{2T_s} t^2
\right\rfloor
\]

\[
\varphi_{T3}(t) =
\begin{cases}
0, & \mathrm{mod}(\phi(t),2\pi) < \pi \\
\pi, & \mathrm{otherwise}
\end{cases}
\]

## T4

T4 is a **quantized polyphase waveform** related to T3, with an additional linear phase term [2].

\[
s(t) = A e^{j(2\pi f_c t + \varphi_{T4}(t) + \varphi_0)}, \quad 0 < t \leq T_s
\]

\[
\phi(t) = \frac{2\pi}{M}
\left\lfloor
\frac{M \Delta F}{2T_s} t^2 - \frac{M \Delta F}{2} t
\right\rfloor
\]

\[
\varphi_{T4}(t) =
\begin{cases}
0, & \mathrm{mod}(\phi(t),2\pi) < \pi \\
\pi, & \mathrm{otherwise}
\end{cases}
\]

## QAM

\[
s(t) = A \sum_{i=1}^{N_c} a_i \, r\!\left(\frac{t-i\Delta t}{\Delta t}\right)e^{j(2\pi f_c t + \varphi_0)},
\quad 0 < t \leq T_s
\]

where \(a_i \in \mathcal{A}_M\) are complex-valued symbols drawn from an \(M\)-QAM constellation normalized to unit average power.

## FSK

FSK stands for **Frequency-Shift Keying**. It is a frequency-coded waveform in which the instantaneous frequency switches between a finite set of discrete tones from one symbol interval to the next [1].

\[
s(t) = A e^{j\left(2\pi \int_0^t f(\tau)\,d\tau + \varphi_0\right)}, \quad 0 < t \leq T_s
\]

\[
f(t) = f_c + \sum_{i=1}^{N_c} c_i \, r\!\left(\frac{t-(i-1)\Delta t}{\Delta t}\right)
\]

\[
\Delta t = \frac{T_s}{N_c}
\]

where \(c_i\) is the \(i\)-th frequency offset.

## LFSK

LFSK follows the same slot structure as FSK, with a local chirp added within each slot.

## OFDM

OFDM stands for **Orthogonal Frequency-Division Multiplexing**. It is a telecommunication waveform in which the transmitted information is distributed across many closely spaced orthogonal subcarriers. Each subcarrier carries a low-rate data stream, which makes the overall transmission efficient for broadband links and robust to frequency-selective channels [3].

## DSSS

DSSS stands for **Direct-Sequence Spread Spectrum**. It is a telecommunication waveform based on spread-spectrum principles: the information signal is multiplied by a higher-rate spreading code, which spreads its energy over a wider bandwidth. This improves robustness to interference and is a core mechanism in many wireless and military communication systems [4].

## Notation

- \(A\): signal amplitude  
- \(f_c\): carrier frequency  
- \(B\): bandwidth  
- \(T_s\): signal duration  
- \(\Delta t\): chip duration  
- \(\varphi_0\): initial phase  
- \(k = B/T_s\): LFM modulation slope  
- \(N_c\): number of chips  
- \(N_p\): number of phase steps  
- \(M\): number of phase states or QAM order  
- \(\Delta F\): frequency sweep parameter  
- \(c_i\): frequency offset of the \(i\)-th FSK code element  
- \(r(\cdot)\): rectangular function  
- \(\mathcal{A}_M\): normalized QAM constellation  

## References

- [1] Nadav Levanon and Eli Mozeson, *Radar Signals*, John Wiley & Sons, 2004.
- [2] Phillip E. Pace, *Detecting and Classifying Low Probability of Intercept Radar*, Artech House, 2009.
- [3] Richard van Nee and Ramjee Prasad, *OFDM for Wireless Multimedia Communications*, Artech House, 2000. Collection: Universal Personal Communications. ISBN: 0-89006-530-6.
- [4] Roger L. Peterson, Rodger E. Ziemer, and David E. Borth, *Introduction to Spread Spectrum Communications*, Prentice Hall, 1995.
