# Waveform Descriptions LaTeX Table

This file keeps the original LaTeX table source for direct reuse in the paper or appendix.

```latex
\begin{table}[h!]
\centering
\begin{tabular}{ll}
\toprule
\textbf{Modulation} & \textbf{Expression} \\
\midrule

\textbf{LFM} &
$ s(t) = A e^{j\left(2\pi f_c t + k \pi t^2 + \varphi_0\right)}, \quad 0 < t \leq T_s $ \\

\textbf{QFM} &
$ s(t) = A e^{j\left(2\pi f_c t + \frac{8B}{3T_s^2} \pi \left(t - \frac{T_s}{2}\right)^3\right)}, \quad 0 < t \leq T_s $ \\

\textbf{SFM} &
$ s(t) = A e^{j\left(2\pi f_c t - \frac{B}{2} T_s \cos\left(\frac{2\pi t}{T_s}\right)\right)}, \quad 0 < t \leq T_s $ \\

\textbf{Costas} &
$\begin{aligned}
s(t) &= \sum_{i=1}^{N_c} \phi_i(t - i\Delta t), \quad 0 < t \leq T_s \\
\phi_i(t) &= \mathrm{Rect}\!\left(\frac{t}{\Delta t}\right) A e^{j(2\pi f_i t + \varphi_0)}, \quad 1 \leq i \leq N_c
\end{aligned}$ \\

\textbf{BPSK} &
$\begin{aligned}
s(t) &= A e^{j(2\pi f_c t + \varphi(t) + \varphi_0)}, \quad 0 < t \leq T_s \\
\varphi(t) &= \sum_{i=1}^{N_c} b_i \pi \cdot r\!\left(\frac{t - i\Delta t}{\Delta t}\right), \quad b_i \in \{0,1\}
\end{aligned}$ \\

\textbf{Frank} &
$\begin{aligned}
s(t) &= A e^{j(2\pi f_c t + \varphi(t) + \varphi_0)}, \quad 0 < t \leq T_s \\
\varphi(t) &= \sum_{i=1}^{N_p^2} \varphi_{pq} \cdot r\!\left(\frac{t - i\Delta t}{\Delta t}\right) \\
\varphi_{pq} &= \frac{2\pi}{N_p} (p-1)(q-1), \quad 1 \leq p,q \leq N_p
\end{aligned}$ \\

\textbf{P3} &
$\begin{aligned}
s(t) &= A e^{j(2\pi f_c t + \varphi(t) + \varphi_0)}, \quad 0 < t \leq T_s \\
\varphi(t) &= \sum_{i=1}^{N_c} \varphi_i \cdot r\!\left(\frac{t - i\Delta t}{\Delta t}\right) \\
\varphi_i &= \frac{\pi}{N_c}(i-1)^2
\end{aligned}$ \\

\textbf{P4} &
$\begin{aligned}
s(t) &= A e^{j(2\pi f_c t + \varphi(t) + \varphi_0)}, \quad 0 < t \leq T_s \\
\varphi(t) &= \sum_{i=1}^{N_c} \varphi_i \cdot r\!\left(\frac{t - i\Delta t}{\Delta t}\right) \\
\varphi_i &= \frac{\pi}{N_c}(i-1)^2 - \pi(i-1)
\end{aligned}$ \\

\bottomrule
\end{tabular}

\vspace{0.5em}
\footnotesize{
Here, $A$, $f_c$, $B$, $T_s$, $\Delta t$, and $\varphi_0$ denote the signal amplitude, carrier frequency, bandwidth, signal duration, chip duration, and initial phase, respectively.
$k = B/T_s$ is the LFM modulation slope.
$N_c$ is the number of chips, and $N_p$ is the number of phase steps for the Frank code ($N_c = N_p^2$).
For phase-coded signals, $B = 1/\Delta t$.
$r(\cdot)$ denotes the rectangular function.
}
\end{table}
```
