using Polynomials

export PolynomialExp, CompoundPolynomialExp
export integrate

# Representation of Σ c_i * x^{n_i} exp(-beta x)
struct PolynomialExp
    polynomial::Polynomial
    beta::Number
end
# Representation of Σ c_i * x^{n_i} exp(-beta_i x)
struct CompoundPolynomialExp{T<:Number,P<:Polynomial}
    polynomials::Dict{T,P}
end

function PolynomialExp(arr::Vector{T}, beta::Number) where {T<:Number}
    return PolynomialExp(Polynomial(arr), beta)
end
(pe::PolynomialExp)(x) = pe.polynomial(x) * exp(-pe.beta * x);

oneunit(PolynomialExp) = PolynomialExp(Polynomial([1.0]), 0.0)
Base.:*(c::Number, pe::PolynomialExp) = PolynomialExp(c * pe.polynomial, pe.beta);
Base.:/(pe::PolynomialExp, c::Number) = PolynomialExp(pe.polynomial / c, pe.beta);
degree(pe::PolynomialExp) = Polynomials.degree(pe.polynomial)

function (cpe::CompoundPolynomialExp)(x)
    return sum(PolynomialExp(poly, beta)(x) for (beta, poly) in cpe.polynomials)
end

zero(CompoundPolynomialExp) = CompoundPolynomialExp(Dict{Number,Polynomial}())
function CompoundPolynomialExp(itr::Vector{Pair{T,P}}) where {T<:Number,P<:Polynomial}
    return CompoundPolynomialExp(Dict(itr))
end
function CompoundPolynomialExp(itr::Vector{Pair{T,P}}) where {T<:Number,P<:Vector}
    return CompoundPolynomialExp(Dict([(k, Polynomial(v)) for (k, v) in itr]))
end
CompoundPolynomialExp(pe::PolynomialExp) = CompoundPolynomialExp([pe.beta => pe.polynomial])

CompoundPolynomialExp(c::Number) = CompoundPolynomialExp([0 => Polynomial([c])])

function Base.isequal(a::CompoundPolynomialExp, b::CompoundPolynomialExp)
    return issetequal(keys(a.polynomials), keys(b.polynomials)) &&
           all([v == b.polynomials[k] for (k, v) in a.polynomials])
end

function Base.:+(a::PolynomialExp, b::PolynomialExp)
    return CompoundPolynomialExp(a) + CompoundPolynomialExp(b)
end
function Base.:+(a::CompoundPolynomialExp, b::CompoundPolynomialExp)
    only_a = setdiff(keys(a.polynomials), keys(b.polynomials))
    only_b = setdiff(keys(b.polynomials), keys(a.polynomials))
    shared_betas = intersect(keys(a.polynomials), keys(b.polynomials))
    new_polynomials = Dict{Number,Polynomial}()

    for beta in only_a
        new_polynomials[beta] = a.polynomials[beta]
    end
    for beta in only_b
        new_polynomials[beta] = b.polynomials[beta]
    end
    for beta in shared_betas
        new_polynomials[beta] = a.polynomials[beta] + b.polynomials[beta]
    end

    return CompoundPolynomialExp(new_polynomials)
end
Base.:+(cpe::CompoundPolynomialExp, pe::PolynomialExp) = cpe + CompoundPolynomialExp(pe)

function Base.show(io::IO, pe::PolynomialExp)
    return print(io, "($(string(pe.polynomial)))exp(-($(pe.beta))x)")
end
function Base.show(io::IO, cpe::CompoundPolynomialExp)
    res = join([PolynomialExp(poly, beta) for (beta, poly) in cpe.polynomials], " + ")
    return print(io, res)
end
function Base.convert(::Type{CompoundPolynomialExp}, x::Float64)
    return CompoundPolynomialExp(Dict([(0, Polynomial([x]))]))
end

# Calculate n! / k! where n >= k
factorial_ratio(n, k) = factorial(n) / factorial(k)

function integrate(pe::PolynomialExp)
    res = zero(CompoundPolynomialExp)
    beta = pe.beta
    poly = pe.polynomial

    if beta == 0
        res += PolynomialExp(Polynomials.integrate(poly), beta)
    else
        for n in 0:degree(pe)
            new_poly = Polynomial([
                factorial_ratio(n, n_min_i) / beta^(n - n_min_i + 1) for n_min_i in 0:n
            ])
            res += PolynomialExp(poly[n] * new_poly, beta)
            res += poly[n] * oneunit(PolynomialExp) / beta^(n + 1)
        end
    end
    return res
end

function integrate(cpe::CompoundPolynomialExp)
    return sum([integrate(PolynomialExp(poly, beta)) for (beta, poly) in cpe.polynomials])
end
