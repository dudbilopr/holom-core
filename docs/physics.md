# Physics & Mathematical Formulation

All spatial coordinates follow row-column array ordering `(y, x)` centered so index `n // 2` sits on the optical axis. Lengths are in meters.

## 1. Scalar Diffraction

### Single-FFT Fresnel Transform (`holom.diffraction.fresnel`)

$$U(\xi, \eta, z) = \frac{e^{ikz}}{i\lambda z}\, e^{\frac{i\pi}{\lambda z}(\xi^2 + \eta^2)} \iint U(x, y, 0)\, e^{\frac{i\pi}{\lambda z}(x^2 + y^2)}\, e^{-\frac{i2\pi}{\lambda z}(x\xi + y\eta)}\, dx\, dy$$

with wave number $k = 2\pi / \lambda$, critical sampling distance $z_c = N \Delta x^2 / \lambda$, and output pixel pitch:

$$\Delta \xi = \frac{\lambda |z|}{N \Delta x}, \qquad \Delta \eta = \frac{\lambda |z|}{M \Delta y}$$

Forward propagation uses $z > 0$; numerical back-propagation of a recorded hologram uses $z < 0$.

### Angular Spectrum (`holom.diffraction.angular_spectrum`)

Propagates plane-wave components in spatial-frequency coordinates $(f_x, f_y)$:

$$H(f_x, f_y, z) = \exp\!\left(i 2\pi z \sqrt{\frac{1}{\lambda^2} - f_x^2 - f_y^2}\right)$$

with evanescent frequencies ($f_x^2 + f_y^2 \ge \lambda^{-2}$) suppressed to preserve numerical stability.

### Multi-Plane Grid Equalization (`holom.focus.fresnel_stack`)

Because the single-FFT Fresnel pixel pitch $\Delta\xi$ scales linearly with $|z|$, planes across an axial scan cannot be compared pixel-by-pixel directly. `fresnel_stack` resamples each intensity plane onto the physical grid of a reference distance $z_{\text{ref}}$ using coordinate scale $k = z_{\text{ref}} / z$ with NaN-safe boundary padding before computing `tenengrad` or `tamura` focus metrics.

## 2. Off-Axis Recording & Zero-Order Suppression

An interferogram between object wave $O$ and tilted reference wave $R$ is:

$$I(x, y) = |O|^2 + |R|^2 + O R^* + O^* R$$

Sampling without aliasing constrains the reference carrier tilt to:

$$\sin\theta_{\max} = \frac{\lambda}{2\Delta x}$$

Single-shot suppression filters implemented in `holom.filters`:
1. `subtract_mean`: subtracts global DC spatial average.
2. `highpass_fft`: masks low frequencies around DC in Fourier space.
3. `laplacian`: second-order discrete high-pass derivative.
4. `sideband_filter`: isolated circular spectral bandpass centered on the $+1$ or $-1$ carrier order.
5. `phase_shifting_4step`: exact extraction of $O R^*$ from four $\pi/2$-stepped interferograms:

$$O R^* = \frac{1}{4}\big[(I_0 - I_2) + i(I_3 - I_1)\big]$$
