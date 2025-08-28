using Polynomials

import Base: isapprox

# Representation of Σ c_i * x^{n_i} exp(-beta x)
struct PolynomialExp{T<:Complex,PT<:Polynomial{T}}
    polynomial::PT
    beta::T
end

# Representation of Σ c_i * x^{n_i} exp(-beta_i x)
struct CompoundPolynomialExp{T<:Complex,PT<:Polynomial{}}
    # It may be easier to just use a Vector of Tuples instead of a Dict.
    # Just make sure to store the keys in some well-defined order.
    polynomials::Dict{T,PT}

    # Unrolling the Dict allows for faster iteration over key-value pairs 
    key_lookup::Vector{T}
    value_lookup::Vector{PT}
end
