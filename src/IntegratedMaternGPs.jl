module IntegratedMaternGPs

import Bessels: besselk
import SpecialFunctions: gamma
import Struve: struvel
using LinearAlgebra
using StaticArrays

export MaternGP, IntegratedMaternGP, kernel
export windowed_cholesky_update!

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
    # Cached multiplicative constant — all terms not involving u
    C::T
end

function IntegratedMaternGP(ν::T, ρ::T, σ2::T) where {T}
    C = σ2 * (2^(1 - ν) / gamma(ν)) * (sqrt(2ν) / ρ)^ν
    return IntegratedMaternGP{T}(ν, ρ, σ2, C)
end

# TODO: there are probably simplifications and cancellations possible here
# TODO: bit worried about numerical stability here
# TODO: the sqrt(2ν)^{-ν} factor in the argument can likely be cancelled with C
function kernel(gp::IntegratedMaternGP, s, t)
    C = gp.C

    # Special case where s = t — only middle integral is non-zero
    if s == t
        return C * (2s * I0(gp, s) - 2I1(gp, s))
    end

    Δ = abs(s - t)
    m = min(s, t)
    M = max(s, t)

    Ia = 2m * I0(gp, Δ) - I1(gp, Δ)
    Ib = (s + t) * I0(gp, Δ, m) - 2I1(gp, Δ, m)
    Ic = M * I0(gp, m, M) - I1(gp, m, M)

    return C * (Ia + Ib + Ic)
end

# Helper functions from derivations
function I0(gp::IntegratedMaternGP{T}, t) where {T}
    ν = gp.ν
    ρ = gp.ρ

    # Special case
    t == 0 && return T(0)

    x = sqrt(2ν) * t / ρ
    return (2^(ν - 1) * t * ρ^ν * sqrt(2ν)^(-ν) * sqrt(π) * gamma(ν + T(0.5))) *
           (besselk(ν, x) * struvel(ν - 1, x) + struvel(ν, x) * besselk(ν - 1, x))
end
I0(gp::IntegratedMaternGP, t1, t2) = I0(gp, t2) - I0(gp, t1)

function I1(gp::IntegratedMaternGP{T}, t) where {T}
    ν = gp.ν
    ρ = gp.ρ

    # Special case
    t == 0 && return T(0)

    x = sqrt(2ν) * t / ρ
    return (
        2^ν * ρ^(ν + 2) * sqrt(2ν)^(-ν - 2) * gamma(ν + 1) -
        t^(ν + 1) * ρ / sqrt(2ν) * besselk(ν + 1, x)
    )
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
    d = length(ks)
    @inbounds @views begin
        U = F.U
        # Remove the first time point using a rank-one update
        lowrankupdate!(Cholesky(UpperTriangular(U[2:d, 2:d])), U[1, 2:d])

        # Shuffle the result to the top-left
        for i in 1:(d - 1)
            for j in 1:(d - 1)
                U[i, j] = U[i + 1, j + 1]
            end
        end

        # Add the new time point
        ldiv!(U[1:(d - 1), d], UpperTriangular(U[1:(d - 1), 1:(d - 1)])', ks[1:(d - 1)])
        v2 = sum(abs2, U[1:(d - 1), d])
        U[d, d] = sqrt(ks[d] - v2)
    end

    return F
end

end # IntegratedMaternGPs.jl
