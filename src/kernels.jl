import Bessels: besselk
import SpecialFunctions: gamma
import Struve: struvel
using LRUCache

export MaternGP, IntegratedMaternGP, kernel
export windowed_cholesky_update!,
    windowed_cholesky_remove_first!, windowed_cholesky_add_last!

abstract type GPKernel end
abstract type StationaryGPKernel <: GPKernel end

struct IntegratedGPKernel <: GPKernel
    base_kernel::GPKernel
end

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


struct MaternGP{T} <: StationaryGPKernel
    ν::T
    ρ::T
    σ2::T
end

isapprox(a::MaternGP, b::MaternGP; rtol=1E-8) = isapprox(a.ν,  b.ν;  rtol) && 
                                                isapprox(a.ρ,  b.ρ;  rtol) && 
                                                isapprox(a.σ2, b.σ2; rtol)

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
