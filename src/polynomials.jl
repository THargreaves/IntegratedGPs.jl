using Polynomials, HCubature, LinearAlgebra, MatrixEquations

export PolynomialExp, CompoundPolynomialExp, SSM
export +, show, isequal, isapprox, zero
export integrate, materntocpe, cpetomaternmixture, ssm2GPKernel

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
function CompoundPolynomialExp(
    itr::Vector{Pair{T,PT}}
) where {T<:Complex,T2<:Real,PT<:AbstractPolynomial{T2}}
    return CompoundPolynomialExp(
        Dict([k => ImmutablePolynomial{complex(T2)}(v) for (k, v) in itr])
    )
end
function CompoundPolynomialExp(
    itr::Vector{Pair{T,PT}}
) where {T<:Real,T2<:Complex,PT<:AbstractPolynomial{T2}}
    return CompoundPolynomialExp(Dict([complex(k) => v for (k, v) in itr]))
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

materntocpe(gp::MaternGP) = materntocpe(gp.ν, gp.ρ, gp.σ2)
materntocpe(gp::CPEMaternGP) = gp.cpe
# In the specific case when ν = p + 0.5 (p ∈ Z), the Matern kernel can be evaluated exactly 
# as a CompoundPolynomialExp
function materntocpe(ν::T, ρ::T, σ2::T) where {T}
    !isinteger(ν - 0.5) && error("Provided Matern kernel does not have a CPE kernel.")

    p = Int(ν - 0.5)

    beta = sqrt(2ν) / ρ
    const_factor = σ2 / factorial_ratio(2p, p)
    base_coefs = [
        binomial(p, p_min_i) *
        factorial_ratio(p + (p - p_min_i), p) *
        (2 * sqrt(2ν) / ρ)^p_min_i for p_min_i in 0:p
    ]
    return CompoundPolynomialExp([beta => const_factor * ImmutablePolynomial(base_coefs)])
end

# Determine a Matern Mixture with the same closed-form as a given CPE using a simple form 
# of Gaussian elimination in the space of PolynomialExps
function cpetomaternmixture(
    cpe::CompoundPolynomialExp{T,PT}
) where {T,PT<:ImmutablePolynomial{T}}
    poly_degs = [
        Polynomials.degree(poly) for (beta, poly) in zip(cpe.key_lookup, cpe.value_lookup)
    ]
    num_terms = sum(poly_degs .+ 1)
    matern_mixture = Vector{CPEMaternGP}(undef, num_terms)
    next_mixture_ind = 1
    for (ind, (beta, poly)) in enumerate(cpe.polynomials)
        temp_poly = poly
        for p in poly_degs[ind]:-1:0
            ν::T = p + 0.5
            ρ::T = sqrt(2ν) / beta

            base_cpe = materntocpe(ν, ρ, T(1))
            base_poly = only(values(base_cpe.polynomials))
            σ2::T = temp_poly[p] / base_poly[p]

            temp_poly -= σ2 * base_poly

            matern_mixture[next_mixture_ind] = CPEMaternGP(
                ν, ρ, σ2, CompoundPolynomialExp(beta => σ2 * base_poly)
            )
            next_mixture_ind += 1
        end
    end
    (next_mixture_ind != num_terms + 1) &&
        error("Matern mixture vector has not been filled up.")

    # Reduce the mixture to only non-trivial components
    return filter((gp::CPEMaternGP) -> !iszero(gp.σ2), matern_mixture)
end

struct SSM{T}
    A::Matrix{T}
    Q::Matrix{T}
    H::Matrix{T}

    function SSM(A::Matrix{T}, Q::Matrix{T}, H::Matrix{T}) where {T}
        return (
                   !(
                       length(size(A)) == 2 && length(size(Q)) == 2 && length(size(H)) == 2
                   ) && error(
                       "All provided matrices must be 2-dimensional; provided A: $(size(A)) and Q: $(size(Q)) and H: $(size(H))",
                   )
               ) ||
               (
                   !(size(Q)[1] == size(Q)[2]) &&
                   error("Q must be a square matrix; provided size is $(size(Q)).")
               ) ||
               (
                   !(size(A)[2] == size(Q)[1]) && error(
                       "A and Q must have compatible sizes; provided $(size(A)) and $(size(Q))",
                   )
               ) ||
               (
                   !(size(H)[2] == size(A)[1]) && error(
                       "H and A must have compatible sizes; provided $(size(H)) and $(size(A))",
                   )
               ) ||
               new{T}(A, Q, H)
    end
end

# Evaluate the SSM Cov for N time steps. Since the Cov is known to have the form of a CPE, 
# the exact coefficients can be evaluated by solving a system of linear equations.
function fit_cov(ssm::SSM{T}) where {T<:AbstractFloat}
    size(ssm.H)[1] != 1 &&
        error("SSM needs to have one output for covariance matching to work.")
    eigen_vals = eigen(ssm.A).values

    onehot(n::Int) = [i == n ? 1 : 0 for i in 1:n]

    N = minimum(size(ssm.A))
    basis = Vector{CompoundPolynomialExp}(undef, N)

    prev_eigen = -Inf
    mult = Inf
    for (ind, eig) in enumerate(eigen_vals)
        # If the current eigenvalue is the same as the previous one, increase 
        # the multiplicity, otherwise reset to 0
        mult = Base.isapprox(eig, prev_eigen; rtol=1E-4) ? (mult + 1) : 0

        basis[ind] = PolynomialExp(onehot(mult + 1), -log(eig < 0 ? Complex(eig) : eig))

        prev_eigen = eig
    end

    M = zeros(complex(T), (N, N))
    v = zeros((N, 1))

    # Solve the Discrete Algebraic Lyapunov Equation to get the stationary process variance
    process_σ2 = only(ssm.H * lyapd(ssm.A, ssm.Q) * ssm.H')
    ssm_cov = process_σ2 * ssm.A

    for t in 1:N
        v[t] = only(ssm.H * ssm_cov * ssm.H')
        for (ind, pe) in enumerate(basis)
            M[t, ind] = pe(float(t))
        end

        ssm_cov = ssm.A * ssm_cov
    end

    coefs = inv(M) * v

    return sum(coefs .* basis)
end

# Since the SSM Cov is a CPE, and a CPE is a Matern Mixture, 
# the SSM Mixture is a Matern Mixture
ssm2GPKernel(ssm::SSM) = cpetomaternmixture(fit_cov(ssm))
