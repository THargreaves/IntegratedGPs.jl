module IntegratedMaternGPs

import Bessels: besselk
import SpecialFunctions: gamma
import Struve: struvel
import Base: isapprox
using LinearAlgebra
using LRUCache
using HCubature

export MaternGP, IntegratedMaternGP, kernel
export windowed_cholesky_update!,
    windowed_cholesky_remove_first!, windowed_cholesky_add_last!

export PolynomialExp, CompoundPolynomialExp
export +, show, isequal, isapprox
export integrate, materntocpe, cpetomaternmixture

abstract type GPKernel end
abstract type StationaryGPKernel <: GPKernel end

struct IntegratedGPKernel <: GPKernel
    base_kernel::GPKernel
end

struct MaternGP{T} <: StationaryGPKernel
    ν::T
    ρ::T
    σ2::T
end

isapprox(a::MaternGP, b::MaternGP; rtol=1E-8) = isapprox(a.ν,  b.ν;  rtol) && 
                                                isapprox(a.ρ,  b.ρ;  rtol) && 
                                                isapprox(a.σ2, b.σ2; rtol)

function kernel(gp::GPKernel, s, t)
    error("GP Kernel has not been implemented.")
end
function kernel(igp::IntegratedGPKernel, s, t)
    return hcubature((s, t) -> kernel(igp.base_kernel, s, t), [0.0, 0.0], [s, t])
end
function kernel(gp_mixture::Vector{T}, s, t) where T <: GPKernel
    sum([kernel(gp, s, t) for gp in gp_mixture])
end

function I0(gp::GPKernel, t)
    hquadrature((x) -> kernel(gp, 0, x), 0.0, t)
end

function I1(gp::GPKernel, t)
    hquadrature((x) -> x * kernel(gp, 0, x), 0.0, t)
end


function kernel(gp::MaternGP, s, t)
    ν = gp.ν
    ρ = gp.ρ
    σ2 = gp.σ2

    d = abs(s - t)
    if d == 0
        return σ2
    else
        return σ2 *
               (2^(1 - ν) / gamma(ν)) *
               (sqrt(2ν) * d / ρ)^ν *
               besselk(ν, sqrt(2ν) * d / ρ)
    end
end

struct IntegratedMaternGP{T} <: GPKernel
    ν::T
    ρ::T
    σ2::T
    # Cached constants for I0, I1 computations
    C0::T
    C1_const::T
    C1_bessel::T
    # LRU caches for evaluations of I0 and I1
    I0_cache::LRU{T,T}
    I1_cache::LRU{T,T}
end

function IntegratedMaternGP(ν::T, ρ::T, σ2::T; cache_size=1000) where {T}
    # Compute constants
    C0 = σ2 * sqrt(π) * gamma(ν + 0.5) / gamma(ν)
    C1_const = σ2 * ρ^2
    C1_bessel = σ2 * 2^(1 - ν) * ρ^2 / (gamma(ν) * 2ν)

    I0_cache = LRU{T,T}(; maxsize=cache_size)
    I1_cache = LRU{T,T}(; maxsize=cache_size)
    return IntegratedMaternGP{T}(ν, ρ, σ2, C0, C1_const, C1_bessel, I0_cache, I1_cache)
end

function kernel(gp::IntegratedMaternGP, s, t)
    # Special case where s = t — only middle integral is non-zero
    if s == t
        return 2s * I0(gp, s) - 2I1(gp, s)
    end

    Δ = abs(s - t)
    m = min(s, t)
    M = max(s, t)

    # Compute component integrals (potentially using LRU cache)
    I0_Δ, I0_m, I0_M = I0(gp, Δ), I0(gp, m), I0(gp, M)
    I1_Δ, I1_m, I1_M = I1(gp, Δ), I1(gp, m), I1(gp, M)

    # Combine over the three piecewise integral regions
    Ia = 2m * I0_Δ - I1_Δ
    Ib = (s + t) * (I0_m - I0_Δ) - 2 * (I1_m - I1_Δ)
    Ic = M * (I0_M - I0_m) - (I1_M - I1_m)

    return Ia + Ib + Ic
end

function I0(gp::IntegratedMaternGP, t)
    get!(gp.I0_cache, t) do
        _I0(gp, t)
    end
end

function _I0(gp::IntegratedMaternGP{T}, t) where {T}
    ν = gp.ν
    ρ = gp.ρ

    # Special case
    t == 0 && return T(0)

    x = sqrt(2ν) * t / ρ
    return (
        gp.C0 * t * (besselk(ν, x) * struvel(ν - 1, x) + struvel(ν, x) * besselk(ν - 1, x))
    )
end

function I1(gp::IntegratedMaternGP, t)
    get!(gp.I1_cache, t) do
        _I1(gp, t)
    end
end

function _I1(gp::IntegratedMaternGP{T}, t) where {T}
    ν = gp.ν
    ρ = gp.ρ

    # Special case
    t == 0 && return T(0)

    x = sqrt(2ν) * t / ρ
    return gp.C1_const - gp.C1_bessel * x^(ν + 1) * besselk(ν + 1, x)
end
I1(gp::GPKernel, t1, t2) = I1(gp, t2) - I1(gp, t1)

function windowed_cholesky_update!(F::Cholesky, ks::AbstractVector)
    """
    Update the Cholesky factorization of a covariance kernel matrix from one window shift.

    The current factorization is stored in `F` which is updated in-place without any
    allocations. `ks` stores entries corresponding to the new row of the covariance kernel
    matrix.

    See: https://en.wikipedia.org/wiki/Cholesky_decomposition#Adding_and_removing_rows_and_columns
    """
    windowed_cholesky_remove_first!(F)
    windowed_cholesky_add_last!(F, ks)
    return F
end

function windowed_cholesky_remove_first!(F::Cholesky)
    """
    Remove the first row and column from the Cholesky factorization in-place.

    This will leave arbitrary values in the last row and column, which can be be filled with
    a new row/column when the next time point is added.
    """
    d = size(F.U, 1)
    U = F.U

    @inbounds @views begin
        # Remove the first time point using a rank-one update
        lowrankupdate!(Cholesky(UpperTriangular(U[2:d, 2:d])), U[1, 2:d])

        # Shuffle the result to the top-left
        for i in 1:(d - 1)
            for j in 1:(d - 1)
                U[i, j] = U[i + 1, j + 1]
            end
        end
    end

    return F
end

function windowed_cholesky_add_last!(F::Cholesky, ks::AbstractVector)
    """
    Add a new row and column to the Cholesky factorization in-place.

    This will replace the last row and column of the factorization with the new values.
    """
    d = size(F.U, 1)
    U = F.U

    @inbounds @views begin
        # Add the new time point
        ldiv!(U[1:(d - 1), d], UpperTriangular(U[1:(d - 1), 1:(d - 1)])', ks[1:(d - 1)])
        v2 = sum(abs2, U[1:(d - 1), d])
        U[d, d] = sqrt(ks[d] - v2)
    end

    return F
end


using Polynomials;

# Representation of Σ c_i * x^{n_i} exp(-beta x)
struct PolynomialExp
    polynomial::Polynomial
    beta::Number
end
# Representation of Σ c_i * x^{n_i} exp(-beta_i x)
struct CompoundPolynomialExp{T<:Number,P<:Polynomial}
    polynomials::Dict{T,P}
end

PolynomialExp(arr::Vector{T}, beta::Number) where T <: Number = PolynomialExp(Polynomial(arr), beta)
(pe::PolynomialExp)(x) = pe.polynomial(x) * exp(-pe.beta * x);

oneunit(PolynomialExp) = PolynomialExp(Polynomial([1.0]), 0.0)
Base.:*(c::Number, pe::PolynomialExp) = PolynomialExp(c * pe.polynomial, pe.beta);
Base.:/(pe::PolynomialExp, c::Number) = PolynomialExp(pe.polynomial / c, pe.beta);
degree(pe::PolynomialExp) = Polynomials.degree(pe.polynomial)


(cpe::CompoundPolynomialExp)(x) = sum(PolynomialExp(poly, beta)(x) for (beta, poly) in cpe.polynomials)

zero(CompoundPolynomialExp) = CompoundPolynomialExp(Dict{Number, Polynomial}())
CompoundPolynomialExp(itr::Vector{Pair{T, P}}) where {T <: Number, P <: Polynomial} = CompoundPolynomialExp(Dict(itr))
CompoundPolynomialExp(itr::Vector{Pair{T, P}}) where {T <: Number, P <: Vector} = CompoundPolynomialExp(Dict([(k, Polynomial(v)) for (k, v) in itr]))
CompoundPolynomialExp(pe::PolynomialExp) = CompoundPolynomialExp([pe.beta => pe.polynomial])

CompoundPolynomialExp(c::Number) = CompoundPolynomialExp([0 => Polynomial([c])])

Base.isequal(a::CompoundPolynomialExp, b::CompoundPolynomialExp) = issetequal(keys(a.polynomials), keys(b.polynomials)) && all([isapprox(v, b.polynomials[k], rtol=1E-8) for (k, v) in a.polynomials])

Base.:+(a::PolynomialExp, b::PolynomialExp) = CompoundPolynomialExp(a) + CompoundPolynomialExp(b)
function Base.:+(a::CompoundPolynomialExp, b::CompoundPolynomialExp)
    only_a = setdiff(keys(a.polynomials), keys(b.polynomials))
    only_b = setdiff(keys(b.polynomials), keys(a.polynomials))
    shared_betas = intersect(keys(a.polynomials), keys(b.polynomials))
    new_polynomials = Dict{Number, Polynomial}()

    for beta in only_a
        new_polynomials[beta] = a.polynomials[beta]
    end
    for beta in only_b
        new_polynomials[beta] = b.polynomials[beta]
    end
    for beta in shared_betas
        new_polynomials[beta] = a.polynomials[beta] + b.polynomials[beta]
    end

    return CompoundPolynomialExp(new_polynomials)
end
Base.:+(cpe::CompoundPolynomialExp, pe::PolynomialExp) = cpe + CompoundPolynomialExp(pe)
Base.:*(cpe::CompoundPolynomialExp, p::Polynomial) = CompoundPolynomialExp([(beta => poly * p) for (beta, poly) in cpe.polynomials])

Base.show(io::IO, pe::PolynomialExp) =  print(io, "($(string(pe.polynomial)))exp(-($(pe.beta))x)")
function Base.show(io::IO, cpe::CompoundPolynomialExp)
    res = join([PolynomialExp(poly, beta) for (beta, poly) in cpe.polynomials], " + ")
    print(io, res)
end
Base.convert(::Type{CompoundPolynomialExp}, x::Float64) = CompoundPolynomialExp(Dict([(0, Polynomial([x]))]))


# Calculate n! / k! where n >= k
factorial_ratio(n, k) = factorial(n) / factorial(k)

function integrate(pe::PolynomialExp)
    res = zero(CompoundPolynomialExp)
    beta = pe.beta
    poly = pe.polynomial

    if beta == 0
        res += PolynomialExp(Polynomials.integrate(poly), beta)
    else
        for n in 0:degree(pe)
            new_poly = Polynomial([factorial_ratio(n, n_min_i) / beta^(n - n_min_i + 1) for n_min_i in 0:n])
            res += PolynomialExp(poly[n] * new_poly, beta)
            res += poly[n] * oneunit(PolynomialExp) / beta^(n + 1)
        end
    end
    res
end

integrate(cpe::CompoundPolynomialExp) = sum([integrate(PolynomialExp(poly, beta)) for (beta, poly) in cpe.polynomials])

I0(cpe::CompoundPolynomialExp, t) = integrate(cpe)(t)
I1(cpe::CompoundPolynomialExp, t) = integrate(cpe * Polynomial([0, 1]))(t)

function materntocpe(gp::MaternGP)
    !isapprox(gp.ν % 1.0, 0.5; rtol=1E-8) && error("Provided Matern kernel does not have a finite CPE kernel.")

    p::Int = floor(gp.ν)

    beta = sqrt(2 * gp.ν) / gp.ρ
    const_factor = gp.σ2 / factorial_ratio(2 * p, p)
    base_coefs = [binomial(p, p_min_i) * factorial_ratio(p + (p - p_min_i), p) * (2 * sqrt(2 * gp.ν) / gp.ρ)^p_min_i for p_min_i in 0:p]
    CompoundPolynomialExp([beta => const_factor * Polynomial(base_coefs)])
end

function cpetomaternmixture(cpe::CompoundPolynomialExp)
    res = Vector{MaternGP}()
    for (beta, poly) in cpe.polynomials
        new_poly = poly 
        while sum(new_poly[0:end].^2) > 1E-8
            p = Polynomials.degree(new_poly)
            ν = p + 0.5
            ρ = sqrt(2ν) / beta

            matern_base = MaternGP(ν, ρ, 1.0)
            base_cpe = materntocpe(matern_base)
            base_poly = base_cpe.polynomials[beta]
            σ2 = new_poly[p] / base_poly[p]
            base_poly *= σ2

            new_poly -= base_poly

            push!(res, MaternGP(ν, ρ, σ2))
        end
    end
    res
end


struct SSM{T} where T
    A::Matrix{T}
    Q::Matrix{T}
    H::Matrix{T}
end

function fit_cov(ssm::SSM)
    size(ssm.H)[1] != 1 && error("SSM needs to have one output for covariance matching to work.")
    eigen_vals = eigen(ssm.A)[1]

    one_hot(n) = [i == n ? 1 : 0 for i in 1:n]

    prev_eigen = -Inf
    mult = 0
    basis = Vector{PolynomialExp}()
    for eig in eigen_vals
        push!(basis, PolynomialExp(onehot(mult + 1), -eig))
        if is_approx(eig, prev_eigen, rtol=1E-4)
            mult = 0
        else
            mult += 1
        end
        prev_eigen = eig
    end
    N = 10

    A = zeros((N, length(basis)))
    v = zeros((N,1))

    ssm_cov = Q

    for t in 1:N
        v[t] = ssm.H * ssm_cov * ssm.H'
        for (ind, pe) in enumerate(basis) 
            A[t,ind] = pe(t)
        end

        ssm_cov = ssm.A * ssm_cov * ssm.A' + ssm.Q
    end

    coefs = inv(A' * A) * A' * v 

    res = sum(coefs .* basis)
end


end # module IntegratedMaternGPs
