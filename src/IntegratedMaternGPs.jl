module IntegratedMaternGPs

import Bessels: besselk
import SpecialFunctions: gamma
import Struve: struvel

export MaternGP, IntegratedMaternGP, kernel

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
        return σ2 * (2^(1 - ν) / gamma(ν)) * (sqrt(2ν) * d / ρ)^ν * besselk(ν, d / ρ)
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

    return (2^(ν - 1) * t * ρ^ν * sqrt(π) * gamma(ν + T(0.5))) * (
        besselk(ν, t / ρ) * struvel(ν - 1, t / ρ) +
        struvel(ν, t / ρ) * besselk(ν - 1, t / ρ)
    )
end
I0(gp::IntegratedMaternGP, t1, t2) = I0(gp, t2) - I0(gp, t1)

function I1(gp::IntegratedMaternGP{T}, t) where {T}
    ν = gp.ν
    ρ = gp.ρ

    # Special case
    t == 0 && return T(0)

    return (2^ν * ρ^(ν + 2) * gamma(ν + 1) - t^(ν + 1) * ρ * besselk(ν + 1, t / ρ))
end
I1(gp::IntegratedMaternGP, t1, t2) = I1(gp, t2) - I1(gp, t1)

end # IntegratedMaternGPs.jl
