module IntegratedMaternGPs

import Bessels: besselk
import SpecialFunctions: gamma
import Struve: struvel
using LinearAlgebra
using LRUCache

export MaternGP, IntegratedMaternGP, kernel
export windowed_cholesky_update!,
    windowed_cholesky_remove_first!, windowed_cholesky_add_last!

export PolynomialExp, CompoundPolynomialExp
export +, show, isequal
export integrate

struct MaternGP{T}
    ν::T
    ρ::T
    σ2::T
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

struct IntegratedMaternGP{T}
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
I1(gp::IntegratedMaternGP, t1, t2) = I1(gp, t2) - I1(gp, t1)

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

Base.isequal(a::CompoundPolynomialExp, b::CompoundPolynomialExp) = issetequal(keys(a.polynomials), keys(b.polynomials)) && all([v == b.polynomials[k] for (k, v) in a.polynomials])

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

function Base.show(io::IO, cpe::CompoundPolynomialExp)
    res = "CPE: "
    cnt = 0 
    for (k, v) in cpe.polynomials
        cnt += 1
        res *= "($(string(v)))exp(-($(k))x)" * (cnt == length(keys(cpe.polynomials)) ? "" : " + ")
    end
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



end # module IntegratedMaternGPs
