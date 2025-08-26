using Polynomials

import Base: isapprox

# Representation of Σ c_i * x^{n_i} exp(-beta x)
struct PolynomialExp
    polynomial::Polynomial
    beta::Number
end
# Representation of Σ c_i * x^{n_i} exp(-beta_i x)
struct CompoundPolynomialExp{T<:Number,P<:Polynomial}
    polynomials::Dict{T,P}
end