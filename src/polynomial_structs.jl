using Polynomials

import Base: isapprox

# Representation of Σ c_i * x^{n_i} exp(-beta x)
struct PolynomialExp{T <: AbstractFloat, PT <: Polynomial{Complex{T}}}
    polynomial::PT
    beta::Complex{T}
end

# Representation of Σ c_i * x^{n_i} exp(-beta_i x)
struct CompoundPolynomialExp{T <: AbstractFloat, PT<:Polynomial{Complex{T}}}
    polynomials::Dict{Complex{T},PT}
end