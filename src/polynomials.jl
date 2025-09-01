using Polynomials, HCubature, LinearAlgebra, MatrixEquations
import Base: isapprox

export PolynomialExp, CompoundPolynomialExp, SSM
export +, show, isequal, isapprox, zero
export integrate, materntocpe, cpetomaternmixture, ssm2GPKernel

# Representation of Σ c_i * x^{n_i} exp(-beta x)
struct PolynomialExp{T<:Number,PT<:ImmutablePolynomial{T}}
    polynomial::PT
    beta::T
end

# Representation of Σ c_i * x^{n_i} exp(-beta_i x)
struct CompoundPolynomialExp{T<:Number,PT<:ImmutablePolynomial{T}}
    # It may be easier to just use a Vector of Tuples instead of a Dict.
    # Just make sure to store the keys in some well-defined order.
    polynomials::Dict{T,PT}

    # Unrolling the Dict allows for faster iteration over key-value pairs 
    key_lookup::Vector{T}
    value_lookup::Vector{PT}
end

function PolynomialExp(arr::Vector{T1}, beta::T2) where {T1,T2}
    return PolynomialExp(ImmutablePolynomial{float(T1)}(arr), float(beta))
end
function PolynomialExp(arr::Vector{T1}, beta::T2) where {T1<:Real,T2<:Complex}
    return PolynomialExp(Vector{complex(T1)}(arr), beta)
end
function PolynomialExp(arr::Vector{T1}, beta::T2) where {T1<:Complex,T2<:Real}
    return PolynomialExp(arr, complex(beta))
end
PolynomialExp(c::T) where {T} = PolynomialExp([c], zero(T))

function (pe::PolynomialExp{T,PT})(x::T2) where {T,PT<:ImmutablePolynomial{T},T2}
    return evalpoly(x, pe.polynomial.coeffs) * exp(-pe.beta * x)
end
#function (pe::PolynomialExp{T,PT})(
#    x::T2
#) where {T<:Complex,PT<:ImmutablePolynomial{T},T2<:Real}
#    return pe(complex(x))
#end;

Base.oneunit(::Type{PolynomialExp{T}}) where {T} = PolynomialExp(oneunit(T))
Base.zero(::Type{PolynomialExp{T}}) where {T} = PolynomialExp(zero(T))
Base.:*(c::Number, pe::PolynomialExp) = PolynomialExp(c * pe.polynomial, pe.beta);
function Base.:*(
    c::T2, cpe::CompoundPolynomialExp{T,PT}
) where {T,PT<:ImmutablePolynomial{T},T2}
    new_pairs::Vector{Pair{T,PT}} = [
        (beta => c * poly) for (beta, poly) in zip(cpe.key_lookup, cpe.value_lookup)
    ]
    return CompoundPolynomialExp(new_pairs)
end;
Base.:/(pe::PolynomialExp, c::Number) = PolynomialExp(pe.polynomial / c, pe.beta);
degree(pe::PolynomialExp) = Polynomials.degree(pe.polynomial)

function (cpe::CompoundPolynomialExp{T,PT})(x::T2) where {T,PT<:ImmutablePolynomial{T},T2}
    s::T = 0
    for (beta, poly) in zip(cpe.key_lookup, cpe.value_lookup)
        s += evalpoly(x, poly.coeffs) * exp(-beta * x)
    end
    return s
end

function increase_poly_deg(poly::PT, n::T2) where {T,PT<:ImmutablePolynomial{T},T2<:Integer}
    return ImmutablePolynomial(
        NTuple{n + 1,T}(i <= Polynomials.degree(poly) ? poly[i] : 0 for i in 0:(n))
    )
end

function CompoundPolynomialExp(dict::Dict{T,PT}) where {T,PT<:AbstractPolynomial{T}}
    degs = [Polynomials.degree(v) for (k, v) in dict]
    max_deg = max(0, maximum(degs))

    N = length(dict)
    new_dict = Dict{T,ImmutablePolynomial{T,:x,max_deg + 1}}()
    key_lookup = Vector{T}(undef, N)
    value_lookup = Vector{ImmutablePolynomial{T,:x,max_deg + 1}}(undef, N)

    for (i, (k, v)) in enumerate(dict)
        new_poly = increase_poly_deg(v, max_deg)
        key_lookup[i] = k
        value_lookup[i] = new_poly

        new_dict[k] = new_poly
    end

    return CompoundPolynomialExp(new_dict, key_lookup, value_lookup)
end
function CompoundPolynomialExp(
    dict::Dict{T,PT}
) where {T<:Complex,T2<:Real,PT<:AbstractPolynomial{T2}}
    new_pairs::Vector{Pair{T,ImmutablePolynomial{complex(T2)}}} = [
        k => ImmutablePolynomial{complex(T2)}(v) for (k, v) in dict
    ]
    return CompoundPolynomialExp(new_pairs)
end
function CompoundPolynomialExp(itr::Vector{Pair{T,PT}}) where {T,PT<:AbstractPolynomial{T}}
    return CompoundPolynomialExp(Dict(itr))
end
function CompoundPolynomialExp(
    itr::Vector{Pair{T,PT}}
) where {T,T2,PT<:AbstractPolynomial{T2}}
    return CompoundPolynomialExp(
        Dict([float(k) => ImmutablePolynomial{float(T2)}(v) for (k, v) in itr])
    )
end
function CompoundPolynomialExp(itr::Vector{Pair{T,PT}}) where {T,T2,PT<:Vector{T2}}
    return CompoundPolynomialExp(
        Dict([(float(k), ImmutablePolynomial(float.(v))) for (k, v) in itr])
    )
end
function CompoundPolynomialExp(
    itr::Vector{Pair{T,PT}}
) where {T<:Real,T2<:Complex,PT<:AbstractPolynomial{T2}}
    return CompoundPolynomialExp(
        Dict([(complex(k), ImmutablePolynomial{T}(v)) for (k, v) in itr])
    )
end
function CompoundPolynomialExp(
    itr::Vector{Pair{T,PT}}
) where {T<:Complex,T2<:Real,PT<:AbstractPolynomial{T2}}
    return CompoundPolynomialExp(
        Dict([(k, ImmutablePolynomial{complex(T)}(v)) for (k, v) in itr])
    )
end

function CompoundPolynomialExp(p::Pair{T,PT}) where {T,T2,PT<:Vector{T2}}
    return CompoundPolynomialExp([p])
end
function CompoundPolynomialExp(p::Pair{T,PT}) where {T,T2,PT<:AbstractPolynomial{T2}}
    return CompoundPolynomialExp([p])
end
function CompoundPolynomialExp(pe::PolynomialExp{T,PT}) where {T,PT<:ImmutablePolynomial{T}}
    return CompoundPolynomialExp(pe.beta => pe.polynomial)
end

function CompoundPolynomialExp(c::T) where {T<:Number}
    return CompoundPolynomialExp(zero(float(T)) => ImmutablePolynomial{float(T)}([c]))
end

function Base.oneunit(
    ::Type{CompoundPolynomialExp{T,PT}}
) where {T,PT<:ImmutablePolynomial{T}}
    return CompoundPolynomialExp(oneunit(T))
end
function Base.zero(::Type{CompoundPolynomialExp{T,PT}}) where {T,PT<:ImmutablePolynomial{T}}
    return CompoundPolynomialExp{T,PT}(Dict())
end

function Base.isequal(a::CompoundPolynomialExp, b::CompoundPolynomialExp)
    return issetequal(keys(a.polynomials), keys(b.polynomials)) && all([
        Polynomials.isapprox(v, b.polynomials[k]; rtol=1E-8) for (k, v) in a.polynomials
    ])
end

function Base.:+(a::PolynomialExp, b::PolynomialExp)
    return CompoundPolynomialExp(a) + CompoundPolynomialExp(b)
end
function Base.:+(
    a::CompoundPolynomialExp{T,PT1}, b::CompoundPolynomialExp{T,PT2}
) where {T,PT1<:ImmutablePolynomial{T},PT2<:ImmutablePolynomial{T}}
    max_deg = max(
        maximum([Polynomials.degree(poly) for poly in a.value_lookup]),
        maximum([Polynomials.degree(poly) for poly in b.value_lookup]),
    )
    only_a = setdiff(keys(a.polynomials), keys(b.polynomials))
    only_b = setdiff(keys(b.polynomials), keys(a.polynomials))
    shared_betas = intersect(keys(a.polynomials), keys(b.polynomials))

    a_pairs::Vector{Pair{T,ImmutablePolynomial{T,:x,max_deg + 1}}} = [
        beta => increase_poly_deg(a.polynomials[beta], max_deg) for beta in only_a
    ]
    b_pairs::Vector{Pair{T,ImmutablePolynomial{T,:x,max_deg + 1}}} = [
        beta => increase_poly_deg(b.polynomials[beta], max_deg) for beta in only_b
    ]
    shared_pairs::Vector{Pair{T,ImmutablePolynomial{T,:x,max_deg + 1}}} = [
        beta =>
            increase_poly_deg(a.polynomials[beta], max_deg) +
            increase_poly_deg(b.polynomials[beta], max_deg) for beta in shared_betas
    ]
    return CompoundPolynomialExp(vcat(a_pairs, b_pairs, shared_pairs))
end
function Base.:+(
    cpe::CompoundPolynomialExp{T,PT}, pe::PolynomialExp{T,PT}
) where {T,PT<:ImmutablePolynomial{T}}
    return cpe + CompoundPolynomialExp(pe)
end
function Base.:*(
    cpe::CompoundPolynomialExp{T,PT}, p::PT
) where {T,PT<:ImmutablePolynomial{T}}
    return CompoundPolynomialExp([
        (beta => poly * p) for (beta, poly) in zip(cpe.key_lookup, cpe.value_lookup)
    ])
end
function Base.:*(
    cpe::CompoundPolynomialExp{T,PT}, p::PT2
) where {T,PT<:ImmutablePolynomial{T},T2,PT2<:AbstractPolynomial{T2}}
    return CompoundPolynomialExp([
        (beta => poly * p) for (beta, poly) in zip(cpe.key_lookup, cpe.value_lookup)
    ])
end
function Base.:*(
    cpe::CompoundPolynomialExp{T,PT}, p::PT2
) where {T,PT<:ImmutablePolynomial{T},T2<:Integer,PT2<:AbstractPolynomial{T2}}
    return CompoundPolynomialExp([
        (beta => poly * ImmutablePolynomial{float(T)}(p)) for
        (beta, poly) in zip(cpe.key_lookup, cpe.value_lookup)
    ])
end
function Base.:*(
    cpe::CompoundPolynomialExp{T,PT}, p::PT2
) where {T,PT<:ImmutablePolynomial{T},T2<:Integer,PT2<:AbstractPolynomial{Complex{T2}}}
    return cpe * ImmutablePolynomial{float(T2)}(p)
end

function Base.show(io::IO, pe::PolynomialExp)
    return print(io, "($(string(pe.polynomial)))exp(-($(pe.beta))x)")
end
function Base.show(io::IO, cpe::CompoundPolynomialExp)
    res = join(
        [
            PolynomialExp(poly, beta) for
            (beta, poly) in zip(cpe.key_lookup, cpe.value_lookup)
        ],
        " + ",
    )
    return print(io, res)
end
function Base.convert(::Type{CompoundPolynomialExp}, x::T) where {T}
    return CompoundPolynomialExp(Dict([(zero(T), ImmutablePolynomial{T}([x]))]))
end
Base.convert(::Type{CompoundPolynomialExp}, pe::PolynomialExp) = CompoundPolynomialExp(pe)

# Convenient function for calculating n! / k! in some of the integral terms to follow
factorial_ratio(n, k) = factorial(n) / factorial(k)

# Evaluate the integral of x^n exp(-beta x) for beta != 0
function integrated_monomial(n, beta)
    return CompoundPolynomialExp([
        0 => [factorial(n) * beta^(-n - 1)],
        beta =>
            [-factorial_ratio(n, n_min_i) / (beta)^(n - n_min_i + 1) for n_min_i in 0:n],
    ])
end

function integrate(pe::PolynomialExp)
    beta = pe.beta
    poly = pe.polynomial

    # If beta = 0, the PolynomialExp is just a polynomial, so standard polynomial 
    # integration is sufficient
    iszero(beta) && return CompoundPolynomialExp(beta => Polynomials.integrate(poly))

    # Integrate the PolynomialExp term by term, given that x^n exp(-beta x) can 
    # be integrated exactly
    return sum(n -> poly[n] * integrated_monomial(n, beta), 0:degree(pe))
end

function integrate(cpe::CompoundPolynomialExp)
    return sum([
        integrate(PolynomialExp(poly, beta)) for
        (beta, poly) in zip(cpe.key_lookup, cpe.value_lookup)
    ])
end # Integrate the CompoundPolynomialExp term by term

I0_form(cpe::CompoundPolynomialExp) = integrate(cpe)
function I1_form(cpe::CompoundPolynomialExp{T,PT}) where {T,PT<:ImmutablePolynomial{T}}
    return integrate(cpe * ImmutablePolynomial{float(T)}([0, 1]))
end
