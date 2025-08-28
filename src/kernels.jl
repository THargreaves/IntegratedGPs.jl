import Bessels: besselk
import SpecialFunctions: gamma
import Struve: struvel
using LRUCache

export AbstractMaternGP,
    AbstractIntegratedMaternGP,
    MaternGP,
    IntegratedMaternGP,
    CPEMaternGP,
    IntegratedCPEMaternGP,
    kernel,
    integrate,
    I0,
    I1,
    Integrated
export windowed_cholesky_update!,
    windowed_cholesky_remove_first!, windowed_cholesky_add_last!

abstract type AbstractGPKernel end
abstract type AbstractRadialGPKernel <: AbstractGPKernel end
abstract type AbstractIntegratedGPKernel <: AbstractGPKernel end
abstract type AbstractIntegratedRadialGPKernel <: AbstractIntegratedGPKernel end

struct Integrated{T<:AbstractGPKernel} <: AbstractIntegratedGPKernel
    base_kernel::T
end

function kernel(gp::AbstractGPKernel, s, t)
    return error("GP Kernel has not been implemented.")
end
function kernel(gp_mixture::Vector{T}, s, t) where {T<:AbstractGPKernel}
    return sum(gp -> kernel(gp, s, t), gp_mixture)
end

function kernel(igp::Integrated{T}, s, t) where {T<:AbstractGPKernel}
    return hcubature(x -> kernel(igp.base_kernel, x[1], x[2]), [0.0, 0.0], [s, t])
end

function I0(gp::AbstractIntegratedRadialGPKernel, t)
    get!(gp.I0_cache, t) do
        _I0(gp, t)
    end
end
_I0(gp::AbstractIntegratedRadialGPKernel, t) = error("The integrated radial GP _I0 
                                                        function has not been implemented.")

function I1(gp::AbstractIntegratedRadialGPKernel, t)
    get!(gp.I1_cache, t) do
        _I1(gp, t)
    end
end
_I1(gp::AbstractIntegratedRadialGPKernel, t) = error("The integrated radial GP _I1 
                                                        function has not been implemented.")
I1(gp::AbstractIntegratedRadialGPKernel, t1, t2) = I1(gp, t2) - I1(gp, t1)

function kernel(gp::AbstractIntegratedRadialGPKernel, s, t)
    Δ = abs(s - t)
    contribution(x) = x * I0(gp, x) - I1(gp, x)

    return contribution(s) - contribution(Δ) + contribution(t)
end

abstract type AbstractMaternGP <: AbstractRadialGPKernel end;
abstract type AbstractIntegratedMaternGP <: AbstractIntegratedRadialGPKernel end;

struct MaternGP{T} <: AbstractMaternGP
    ν::T
    ρ::T
    σ2::T
end

struct CPEMaternGP{T<:Complex,PT<:Polynomial{T}} <: AbstractMaternGP
    ν::T
    ρ::T
    σ2::T
    cpe::CompoundPolynomialExp{T,PT}

    function CPEMaternGP(
        ν::T, ρ::T, σ2::T, cpe::CompoundPolynomialExp{T,PT}
    ) where {T,PT<:Polynomial{T}}
        if !isinteger(ν - 0.5)
            error("CPE Matern GP needs ν to be of the form p + 0.5; given $(ν)")
        else
            new{T,PT}(ν, ρ, σ2, cpe)
        end
    end
end

function CPEMaternGP(ν::T, ρ::T, σ2::T) where {T}
    cpe = materntocpe(ν, ρ, σ2)
    return CPEMaternGP(complex(ν), complex(ρ), complex(σ2), cpe)
end

function AbstractMaternGP(ν::T, ρ::T, σ2::T) where {T<:AbstractFloat}
    return isinteger(ν - 0.5) ? CPEMaternGP(ν, ρ, σ2) : MaternGP(ν, ρ, σ2)
end

function isapprox(a::MaternGP, b::MaternGP; rtol=1E-8)
    return isapprox(a.ν, b.ν; rtol) &&
           isapprox(a.ρ, b.ρ; rtol) &&
           isapprox(a.σ2, b.σ2; rtol)
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

kernel(gp::CPEMaternGP, s, t) = gp.cpe(abs(s - t))

struct IntegratedMaternGP{T} <: AbstractIntegratedMaternGP
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

function IntegratedMaternGP(gp::MaternGP{T}; cache_size=1000) where {T}
    return IntegratedMaternGP(gp.ν, gp.ρ, gp.σ2; cache_size)
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

function _I1(gp::IntegratedMaternGP{T}, t) where {T}
    ν = gp.ν
    ρ = gp.ρ

    # Special case
    t == 0 && return T(0)

    x = sqrt(2ν) * t / ρ
    return gp.C1_const - gp.C1_bessel * x^(ν + 1) * besselk(ν + 1, x)
end

struct IntegratedCPEMaternGP{T,PT<:Polynomial{T}} <: AbstractIntegratedMaternGP
    ν::T
    ρ::T
    σ2::T

    # Store the CPE closed-forms for I0 and I1 
    I0_cpe::CompoundPolynomialExp{T,PT}
    I1_cpe::CompoundPolynomialExp{T,PT}

    # LRU caches for evaluations of I0 and I1
    I0_cache::LRU{T,T}
    I1_cache::LRU{T,T}
end

function IntegratedCPEMaternGP(gp::CPEMaternGP{T}; cache_size=1000) where {T}
    I0_cpe = I0_form(gp.cpe)
    I1_cpe = I1_form(gp.cpe)

    I0_cache = LRU{T,T}(; maxsize=cache_size)
    I1_cache = LRU{T,T}(; maxsize=cache_size)
    return IntegratedCPEMaternGP(gp.ν, gp.ρ, gp.σ2, I0_cpe, I1_cpe, I0_cache, I1_cache)
end

_I0(gp::IntegratedCPEMaternGP{T}, t) where {T} = gp.I0_cpe(t)
_I1(gp::IntegratedCPEMaternGP{T}, t) where {T} = gp.I1_cpe(t)

AbstractIntegratedMaternGP(gp::MaternGP) = IntegratedMaternGP(gp)
AbstractIntegratedMaternGP(gp::CPEMaternGP) = IntegratedCPEMaternGP(gp)

function integrate(gp_mixture::Vector{T}) where {T<:AbstractGPKernel}
    return [integrate(gp) for gp in gp_mixture]
end
integrate(gp::T) where {T<:AbstractGPKernel} = Integrated{T}(gp)
integrate(gp::AbstractMaternGP) = AbstractIntegratedMaternGP(gp)

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
