# IntegratedGPs

[![Build Status](https://github.com/THargreaves/IntegratedGPs.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/THargreaves/IntegratedGPs.jl/actions/workflows/CI.yml?query=branch%3Amaster)


We provide IntegratedGPs.jl, a library for performing inference with Integrated Gaussian Processes.
An "integrated GP" is a Gaussian Process which is the integral of another GP.
This library provides functions for evaluating the kernel of the integrated GP in terms of the base GP.
In particular, we provide closed-form analytic integrated kernels for the Matérn, Squared Exponential, and Rational Quadratic kernels.

## Referencing

Please cite the following paper if you use this library:

> R. J. McDougall, T. Hargreaves and S. J. Godsill, "Exact Integration of Stationary Gaussian Process Kernels," in IEEE Open Journal of Signal Processing, vol. 7, pp. 257-265, 2026, doi: [10.1109/OJSP.2026.3656062](10.1109/OJSP.2026.3656062)

## Reproducibility

The results from the paper listed above can be reproduced by running the file `scripts/ojsp.jl` with the figures being produced in `scripts/figs/`.