using Polynomials

import Base: isapprox

# Representation of Σ c_i * x^{n_i} exp(-beta x)
struct PolynomialExp{T, PT <: Polynomial{T}}
    polynomial::PT
    beta::T
end
# Representation of Σ c_i * x^{n_i} exp(-beta_i x)
struct CompoundPolynomialExp{T,PT<:Polynomial{T}}
    polynomials::Dict{T,PT}
end