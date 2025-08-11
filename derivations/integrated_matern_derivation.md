Recall the Matern kernel:

$$
k(s, t) =  k(u := |s - t|) = \sigma^2 \frac{2^{1 - \nu}}{\Gamma(\nu)} \left(\sqrt{2\nu}\frac{u}{\rho}\right)^\nu K_\nu\left(\frac{u}{\rho}\right)
$$

**I missed a factor of $\sqrt{2ν}$ in the Bessel argument. This has been corrected in the code but not in these notes.**

where $K_\nu$ is the modified Bessel function of the second kind, $\sigma^2$ is

For any radial kernel $k(s, t)$ = k(|s - t|) the integrated kernel,

$$
I(S, T) = \int_0^S \int_0^T k(s, t) \, ds \, dt,
$$

can be written as a sum of single integrals over $u := |s - t|$ (see handwritten
notes):

$$
\begin{align*}
I &= \int_0^{\Delta} (2m - u)k(u) du + \\
  &= \int_\Delta^{m} (S + T - 2u)k(u) du + \\
    &= \int_m^{M} (M - u)k(u) du
\end{align*}
$$

where,

- $m = \min(S, T)$
- $M = \max(S, T)$
- $\Delta = |S - T|$

Gradshteyn and Ryzhik (8th ed.) gives [6.561.4]

$$
\int_0^1 x^\nu K_\nu(a x) d x=2^{\nu-1} a^{-\nu} \pi^{\frac{1}{2}} \Gamma\left(\nu+\frac{1}{2}\right)\left[K_\nu(a) \mathbf{L}_{\nu-1}(a)+\mathbf{L}_\nu(a) K_{\nu-1}(a)\right]
$$

and [6.561.8]

$$
\int_0^1 x^{\nu+1} K_\nu(a x) \mathrm{d} x=2^\nu a^{-\nu-2} \Gamma(\nu+1)-a^{-1} K_{\nu+1}(a).
$$

It follows by substitution that

$$I_0(t) := \int_0^t u^\nu K_\nu\left(\frac{u}{\rho}\right) du = 2^{\nu-1} t \rho^\nu \pi^{\frac{1}{2}} \Gamma\left(\nu+\frac{1}{2}\right)\left[K_\nu\left(\frac{t}{\rho}\right) \mathbf{L}_{\nu-1}\left(\frac{t}{\rho}\right)+\mathbf{L}_\nu\left(\frac{t}{\rho}\right) K_{\nu-1}\left(\frac{t}{\rho}\right)\right]$$

and

$$I_1(t) := \int_0^t u^{\nu+1} K_\nu\left(\frac{u}{\rho}\right) du = 2^\nu \rho^{\nu+2} \Gamma(\nu+1) - t^{\nu+1}\rho K_{\nu+1}\left(\frac{t}{\rho}\right).$$

We can then write the final integral in terms of these helper integrals.
