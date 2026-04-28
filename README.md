# IntegratedGPs.jl

[![Build Status](https://github.com/THargreaves/IntegratedGPs.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/THargreaves/IntegratedGPs.jl/actions/workflows/CI.yml?query=branch%3Amaster)


We provide `IntegratedGPs.jl`, a library for performing inference with Integrated Gaussian Processes.
An "integrated GP" is a Gaussian Process which is the integral of another GP.
This library provides functions for evaluating the kernel of the integrated GP in terms of the base GP.
In particular, we provide closed-form analytic integrated kernels for the Matérn, Squared Exponential, and Rational Quadratic kernels.

## Installation

This library is currently unregistered. It can be included in a Julia project through:
```julia
(@v1.12) pkg> add https://github.com/THargreaves/IntegratedGPs.jl.git
```
or by cloning the repo onto your file system (see Reproducibility).

## Usage 

Example Julia scripts can be found in [scripts/](scripts/).
A minimal example is shown below:
```julia
using IntegratedGPs
using LinearAlgebra
using Plots

ν = 1.5
ρ = 0.8
σk = 1.2
gp = constructmatern(ν, ρ, σk^2)
igp = integrate(gp)


N = 100
K = [kernel(igp, float(i), float(j)) for i in 1:N, j in 1:N]
K = 0.5 * (K + K')

proc = cholesky(K).L * randn(N)

plot(proc)
```

## Referencing

Please cite the following paper if you use this library:

> R. J. McDougall, T. Hargreaves and S. J. Godsill, "Exact Integration of Stationary Gaussian Process Kernels," in IEEE Open Journal of Signal Processing, vol. 7, pp. 257-265, 2026, doi: [10.1109/OJSP.2026.3656062](https://doi.org/10.1109/OJSP.2026.3656062)

## Reproducibility

The results from the paper listed above can be reproduced by running the file [scripts/ojsp.jl](scripts/ojsp.jl). This script will download the multi-object tracking dataset: [TeamTrack](https://www.kaggle.com/datasets/atomscott/teamtrack) and run all analysis scripts used in the paper. The figures will be produced in [scripts/figs/](scripts/figs/).

A clean reproducibility test can be performed by running the following instructions in the terminal:
```bash
Documents> git clone https://github.com/THargreaves/IntegratedGPs.jl.git
Documents> cd IntegratedGPs.jl
Documents/IntegratedGPs.jl> julia
```
followed by the following instructions in the Julia REPL:
```julia
(@v1.12) pkg> activate IntegratedGPs
(IntegratedGPs) pkg> instantiate
(IntegratedGPs) pkg> activate scripts
(scripts) pkg> instantiate
julia> include("scripts/ojsp.jl")
```
Note that the package mode is entered by pressing the `]` key, and the backspace returning to the standard REPL.
