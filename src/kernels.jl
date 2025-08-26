import Bessels: besselk
import SpecialFunctions: gamma
import Struve: struvel
using LRUCache

export MaternGP, IntegratedMaternGP, GeneralMaternGP, IntegratedGeneralMaternGP, CPEMaternGP, IntegratedCPEMaternGP, kernel, integrate, I0, I1
export windowed_cholesky_update!,
    windowed_cholesky_remove_first!, windowed_cholesky_add_last!

abstract type GPKernel end
abstract type StationaryGPKernel <: GPKernel end
abstract type IntegratedGPKernel <: GPKernel end
abstract type IntegratedStationaryGPKernel <: IntegratedGPKernel end

struct IntegratedGeneralGPKernel <: IntegratedGPKernel
    base_kernel::GPKernel
end

function kernel(gp::GPKernel, s, t)
    error("GP Kernel has not been implemented.")
end
function kernel(gp_mixture::Vector{T}, s, t) where T <: GPKernel
    sum([kernel(gp, s, t) for gp in gp_mixture])
end

function kernel(igp::IntegratedGeneralGPKernel, s, t)
    return hcubature((s, t) -> kernel(igp.base_kernel, s, t), [0.0, 0.0], [s, t])
end


function I0(gp::IntegratedStationaryGPKernel, t)
    get!(gp.I0_cache, t) do
        _I0(gp, t)
    end
end
_I0(gp::IntegratedStationaryGPKernel, t) = error("The integrated stationary GP _I0 function has not been implemented.") 

function I1(gp::IntegratedStationaryGPKernel, t)
    get!(gp.I1_cache, t) do
        _I1(gp, t)
    end
end
_I1(gp::IntegratedStationaryGPKernel, t) = error("The integrated stationary GP _I1 function has not been implemented.")
I1(gp::IntegratedStationaryGPKernel, t1, t2) = I1(gp, t2) - I1(gp, t1)


function kernel(gp::IntegratedStationaryGPKernel, s, t)
    Δ = abs(s - t)
    contribution(x) = x * I0(gp, x) - I1(gp, x)
    
    return contribution(s) - contribution(Δ) + contribution(t)
end


abstract type MaternGP <: StationaryGPKernel end;
abstract type IntegratedMaternGP <: IntegratedStationaryGPKernel end;

struct GeneralMaternGP{T1, T2, T3} <: MaternGP
    ν::T1
    ρ::T2
    σ2::T3
end

struct CPEMaternGP{T1, T2, T3} <: MaternGP
    ν::T1
    ρ::T2
    σ2::T3
    cpe::CompoundPolynomialExp

    CPEMaternGP(ν::T1, ρ::T2, σ2::T3, cpe::CompoundPolynomialExp) where {T1, T2, T3} = !isinteger(ν - 0.5) ? error("CPE Matern GP needs ν to be of the form p + 0.5; given $(ν)") : new{T1, T2, T3}(ν, ρ, σ2, cpe) 
end

function CPEMaternGP(ν, ρ, σ2)
    cpe = materntocpe(ν, ρ, σ2)
    CPEMaternGP(ν, ρ, σ2, cpe)
end

MaternGP(ν, ρ, σ2) = isinteger(ν - 0.5) ? CPEMaternGP(ν, ρ, σ2) : GeneralMaternGP(ν, ρ, σ2)

isapprox(a::GeneralMaternGP, b::GeneralMaternGP; rtol=1E-8) = isapprox(a.ν,  b.ν;  rtol) && 
                                                isapprox(a.ρ,  b.ρ;  rtol) && 
                                                isapprox(a.σ2, b.σ2; rtol)

function kernel(gp::GeneralMaternGP, s, t)
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

struct IntegratedGeneralMaternGP{T} <: IntegratedMaternGP
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

IntegratedGeneralMaternGP(gp::GeneralMaternGP{T}; cache_size=1000) where {T} = IntegratedGeneralMaternGP(gp.ν, gp.ρ, gp.σ2; cache_size)
function IntegratedGeneralMaternGP(ν::T, ρ::T, σ2::T; cache_size=1000) where {T}
    # Compute constants
    C0 = σ2 * sqrt(π) * gamma(ν + 0.5) / gamma(ν)
    C1_const = σ2 * ρ^2
    C1_bessel = σ2 * 2^(1 - ν) * ρ^2 / (gamma(ν) * 2ν)

    I0_cache = LRU{T,T}(; maxsize=cache_size)
    I1_cache = LRU{T,T}(; maxsize=cache_size)
    return IntegratedGeneralMaternGP{T}(ν, ρ, σ2, C0, C1_const, C1_bessel, I0_cache, I1_cache)
end

function _I0(gp::IntegratedGeneralMaternGP{T}, t) where {T}
    ν = gp.ν
    ρ = gp.ρ

    # Special case
    t == 0 && return T(0)

    x = sqrt(2ν) * t / ρ
    return (
        gp.C0 * t * (besselk(ν, x) * struvel(ν - 1, x) + struvel(ν, x) * besselk(ν - 1, x))
    )
end

function _I1(gp::IntegratedGeneralMaternGP{T}, t) where {T}
    ν = gp.ν
    ρ = gp.ρ

    # Special case
    t == 0 && return T(0)

    x = sqrt(2ν) * t / ρ
    return gp.C1_const - gp.C1_bessel * x^(ν + 1) * besselk(ν + 1, x)
end

struct IntegratedCPEMaternGP{T} <: IntegratedMaternGP
    ν::T
    ρ::T
    σ2::T

    # Store the CPE closed-forms for I0 and I1 
    I0_cpe::CompoundPolynomialExp
    I1_cpe::CompoundPolynomialExp

    # LRU caches for evaluations of I0 and I1
    I0_cache::LRU{T,T}
    I1_cache::LRU{T,T}
end

function IntegratedCPEMaternGP(gp::CPEMaternGP{T}; cache_size=1000) where {T}
    I0_cpe = I0_form(gp.cpe)
    I1_cpe = I1_form(gp.cpe)

    I0_cache = LRU{T,T}(; maxsize=cache_size)
    I1_cache = LRU{T,T}(; maxsize=cache_size)
    return IntegratedCPEMaternGP{T}(gp.ν, gp.ρ, gp.σ2, I0_cpe, I1_cpe, I0_cache, I1_cache)
end

_I0(gp::IntegratedCPEMaternGP{T}, t) where {T} = gp.I0_cpe(t)
_I1(gp::IntegratedCPEMaternGP{T}, t) where {T} = gp.I1_cpe(t)

IntegratedMaternGP(gp::GeneralMaternGP) = IntegratedGeneralMaternGP(gp)
IntegratedMaternGP(gp::CPEMaternGP) = IntegratedCPEMaternGP(gp)


integrate(gp_mixture::Vector{GPKernel}) = [integrate(gp) for gp in gp_mixture]
integrate(gp::GPKernel) = IntegratedGeneralGPKernel(gp)
integrate(gp::MaternGP) = IntegratedMaternGP(gp)



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
