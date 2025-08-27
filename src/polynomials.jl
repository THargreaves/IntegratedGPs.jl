using Polynomials, HCubature, LinearAlgebra, MatrixEquations

export PolynomialExp, CompoundPolynomialExp, SSM
export +, show, isequal, isapprox, zero
export integrate, materntocpe, cpetomaternmixture, ssm2GPKernel


PolynomialExp(arr::Vector{Complex{T}}, beta::Complex{T}) where T <: AbstractFloat = PolynomialExp(Polynomial(arr), beta)
PolynomialExp(arr::Vector{T}, beta::Complex{T}) where T <: AbstractFloat = PolynomialExp(complex.(arr), beta)
PolynomialExp(arr::Vector{T}, beta::T) where T <: Real = PolynomialExp(complex.(float.(arr)), complex(float(beta)))
PolynomialExp(arr::Vector{Complex{T}}, beta::Complex{T}) where T <: Integer = PolynomialExp(float.(arr), float(beta))
PolynomialExp(arr::Vector{T1}, beta::Complex{T2}) where {T1 <: Integer, T2 <: AbstractFloat} = PolynomialExp(float.(arr), beta)
PolynomialExp(arr::Vector{T1}, beta::Complex{T2}) where {T1 <: AbstractFloat, T2 <: Integer} = PolynomialExp(arr, float(beta))
PolynomialExp(arr::Vector{T}, beta::Complex{T}) where T <: Integer = PolynomialExp(float.(arr), float(beta))
PolynomialExp(arr::Vector{T1}, beta::T2) where {T1, T2 <: Real} = PolynomialExp(arr, complex(beta))
PolynomialExp(c::T) where {T <: Number} = PolynomialExp([c], zero(T))

(pe::PolynomialExp)(x) = pe.polynomial(x) * exp(-pe.beta * x);

Base.oneunit(::Type{PolynomialExp{T}}) where {T} = PolynomialExp(oneunit(Complex{T}))
Base.zero(::Type{PolynomialExp{T}}) where {T} = PolynomialExp(zero(Complex{T}))
Base.:*(c::Number, pe::PolynomialExp) = PolynomialExp(c * pe.polynomial, pe.beta);
function Base.:*(c::Number, cpe::CompoundPolynomialExp)
    return CompoundPolynomialExp([beta => c * poly for (beta, poly) in cpe.polynomials])
end;
Base.:/(pe::PolynomialExp, c::Number) = PolynomialExp(pe.polynomial / c, pe.beta);
degree(pe::PolynomialExp) = Polynomials.degree(pe.polynomial)

(cpe::CompoundPolynomialExp)(x) = sum([PolynomialExp(poly, beta)(x) for (beta, poly) in cpe.polynomials])

CompoundPolynomialExp(itr::Vector{Pair{Complex{T}, PT}}) where {T <: AbstractFloat, PT <: Polynomial{Complex{T}}} = CompoundPolynomialExp(Dict(itr))
CompoundPolynomialExp(itr::Vector{Pair{T, PT}}) where {T <: AbstractFloat, T2, PT <: Polynomial{T2}} = CompoundPolynomialExp(Dict([complex(float(k)) => Polynomial{complex(float(T2))}(v) for (k, v) in itr]))
CompoundPolynomialExp(itr::Vector{Pair{Complex{T}, PT}}) where {T <: AbstractFloat, PT <: Vector{Complex{T}}} = CompoundPolynomialExp(Dict([(k, Polynomial(v)) for (k, v) in itr]))
CompoundPolynomialExp(itr::Vector{Pair{T, PT}}) where {T, T2, PT <: Vector{T2}} = CompoundPolynomialExp(Dict([(complex(float(k)), Polynomial(complex.(float.(v)))) for (k, v) in itr]))

CompoundPolynomialExp(p::Pair{T, PT}) where {T, T2, PT <: Vector{T2}} = CompoundPolynomialExp([p])
CompoundPolynomialExp(p::Pair{T, PT}) where {T, T2, PT <: Polynomial{T2}} = CompoundPolynomialExp([p])
CompoundPolynomialExp(pe::PolynomialExp) = CompoundPolynomialExp(pe.beta => pe.polynomial)

CompoundPolynomialExp(c::T) where T <: Number = CompoundPolynomialExp(zero(complex(float(T))) => Polynomial{complex(float(T))}([c]))

Base.oneunit(::Type{CompoundPolynomialExp{T, PT}}) where {T <: AbstractFloat, PT <:Polynomial{Complex{T}}} = CompoundPolynomialExp(oneunit(T))
Base.zero(::Type{CompoundPolynomialExp{T, PT}}) where {T <: AbstractFloat, PT <:Polynomial{Complex{T}}} = CompoundPolynomialExp{T, PT}(Dict())

function Base.isequal(a::CompoundPolynomialExp, b::CompoundPolynomialExp)
    return issetequal(keys(a.polynomials), keys(b.polynomials)) && all([
        Polynomials.isapprox(v, b.polynomials[k]; rtol=1E-8) for (k, v) in a.polynomials
    ])
end

function Base.:+(a::PolynomialExp, b::PolynomialExp)
    return CompoundPolynomialExp(a) + CompoundPolynomialExp(b)
end
function Base.:+(a::CompoundPolynomialExp, b::CompoundPolynomialExp)
    only_a = setdiff(keys(a.polynomials), keys(b.polynomials))
    only_b = setdiff(keys(b.polynomials), keys(a.polynomials))
    shared_betas = intersect(keys(a.polynomials), keys(b.polynomials))

    return CompoundPolynomialExp(
        vcat(
            [beta => a.polynomials[beta] for beta in only_a],
            [beta => b.polynomials[beta] for beta in only_b],
            [beta => a.polynomials[beta] + b.polynomials[beta] for beta in shared_betas],
        ),
    )
end
Base.:+(cpe::CompoundPolynomialExp, pe::PolynomialExp) = cpe + CompoundPolynomialExp(pe)
function Base.:*(cpe::CompoundPolynomialExp, p::Polynomial)
    return CompoundPolynomialExp([(beta => poly * p) for (beta, poly) in cpe.polynomials])
end

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
Base.convert(::Type{CompoundPolynomialExp}, pe::PolynomialExp) = CompoundPolynomialExp(pe)

# Convenient function for calculating n! / k! in some of the integral terms to follow
factorial_ratio(n, k) = factorial(n) / factorial(k)

# Evaluate the integral of x^n exp(-beta x) for beta != 0
function integrated_monomial(n, beta)
    return CompoundPolynomialExp([
        0 => [beta^(-n - 1)],
        beta => [factorial_ratio(n, n_min_i) / beta^(n - n_min_i + 1) for n_min_i in 0:n],
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
    sum(n -> poly[n] * integrated_monomial(n, beta), 0:degree(pe))
end

function integrate(cpe::CompoundPolynomialExp)
    return sum([integrate(PolynomialExp(poly, beta)) for (beta, poly) in cpe.polynomials])
end # Integrate the CompoundPolynomialExp term by term

I0_form(cpe::CompoundPolynomialExp) = integrate(cpe)
I1_form(cpe::CompoundPolynomialExp) = integrate(cpe * Polynomial([0, 1]))


materntocpe(gp::MaternGP) = materntocpe(gp.ν, gp.ρ, gp.σ2)
materntocpe(gp::CPEMaternGP) = gp.cpe
# In the specific case when ν = p + 0.5 (p ∈ Z), the Matern kernel can be evaluated exactly 
# as a CompoundPolynomialExp
function materntocpe(ν, ρ, σ2)
    !isinteger(ν - 0.5) && error("Provided Matern kernel does not have a CPE kernel.")

    p = Int(ν - 0.5)

    beta = sqrt(2ν) / ρ
    const_factor = σ2 / factorial_ratio(2p, p)
    base_coefs = [
        binomial(p, p_min_i) *
        factorial_ratio(p + (p - p_min_i), p) *
        (2 * sqrt(2ν) / ρ)^p_min_i for p_min_i in 0:p
    ]
    return CompoundPolynomialExp([beta => const_factor * Polynomial(base_coefs)])
end

# Determine a Matern Mixture with the same closed-form as a given CPE using a simple form 
# of Gaussian elimination in the space of PolynomialExps
function cpetomaternmixture(cpe::CompoundPolynomialExp{T, PT}) where {T <: AbstractFloat, PT <: Polynomial{Complex{T}}}
    poly_degs = [Polynomials.degree(poly) for (beta, poly) in cpe.polynomials]
    num_terms = sum(poly_degs .+ 1)
    matern_mixture = Vector{CPEMaternGP}(undef, num_terms)
    next_mixture_ind = 1
    for (ind, (beta, poly)) in enumerate(cpe.polynomials)
        temp_poly = poly
        for p in poly_degs[ind]:-1:0
            ν::Complex{T} = p + 0.5
            ρ = sqrt(2ν) / beta

            base_cpe = materntocpe(ν, ρ, 1.0)
            base_poly = only(values(base_cpe.polynomials))
            σ2 = temp_poly[p] / base_poly[p]

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
function fit_cov(ssm::SSM)
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

    M = zeros(ComplexF64, (N, N))
    v = zeros((N, 1))

    # Solve the Discrete Algebraic Lyapunov Equation to get the stationary process variance
    process_σ2 = only(ssm.H * lyapd(ssm.A, ssm.Q) * ssm.H') 
    ssm_cov = process_σ2 * ssm.A

    for t in 1:N
        v[t] = only(ssm.H * ssm_cov * ssm.H')
        for (ind, pe) in enumerate(basis)
            M[t, ind] = pe(t)
        end

        ssm_cov = ssm.A * ssm_cov
    end

    coefs = inv(M) * v

    return sum(coefs .* basis)
end

# Since the SSM Cov is a CPE, and a CPE is a Matern Mixture, 
# the SSM Mixture is a Matern Mixture
ssm2GPKernel(ssm::SSM) = cpetomaternmixture(fit_cov(ssm))