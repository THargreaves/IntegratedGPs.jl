using Polynomials

import Base: isapprox

# Representation of Σ c_i * x^{n_i} exp(-beta x)
struct PolynomialExp{T <: AbstractFloat, PT <: Polynomial{Complex{T}}}
    polynomial::PT
    beta::Complex{T}
end

# Representation of Σ c_i * x^{n_i} exp(-beta_i x)
struct CompoundPolynomialExp{T <: AbstractFloat, PT<:Polynomial{Complex{T}}}
    # It may be easier to just use a Vector of Tuples instead of a Dict.
    # Just make sure to store the keys in some well-defined order.
    polynomials::Dict{Complex{T},PT}

    # Unrolling the Dict allows for faster iteration over key-value pairs 
    key_lookup::Vector{Complex{T}}
    value_lookup::Vector{PT}
end