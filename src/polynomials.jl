using Polynomials, HCubature, LinearAlgebra, MatrixEquations

import Base: isapprox

export PolynomialExp, CompoundPolynomialExp, SSM
export +, show, isequal, isapprox
export integrate, materntocpe, cpetomaternmixture, ssm2GPKernel

# Representation of Σ c_i * x^{n_i} exp(-beta x)
struct PolynomialExp
    polynomial::Polynomial
    beta::Number
end
# Representation of Σ c_i * x^{n_i} exp(-beta_i x)
struct CompoundPolynomialExp{T<:Number,P<:Polynomial}
    polynomials::Dict{T,P}
end

PolynomialExp(arr::Vector{T}, beta::Number) where T <: Number = PolynomialExp(Polynomial(arr), beta)
(pe::PolynomialExp)(x) = pe.polynomial(x) * exp(-pe.beta * x);

Base.oneunit(::Type{PolynomialExp}) = PolynomialExp([1.0], 0.0)
Base.zero(::Type{PolynomialExp}) = PolynomialExp([0], 0)
Base.:*(c::Number, pe::PolynomialExp) = PolynomialExp(c * pe.polynomial, pe.beta);
Base.:*(c::Number, cpe::CompoundPolynomialExp) = CompoundPolynomialExp([beta => c * poly for (beta, poly) in cpe.polynomials]);
Base.:/(pe::PolynomialExp, c::Number) = PolynomialExp(pe.polynomial / c, pe.beta);
degree(pe::PolynomialExp) = Polynomials.degree(pe.polynomial)

(cpe::CompoundPolynomialExp)(x) = sum(PolynomialExp(poly, beta)(x) for (beta, poly) in cpe.polynomials)

CompoundPolynomialExp(itr::Vector{Pair{T, P}}) where {T <: Number, P <: Polynomial} = CompoundPolynomialExp(Dict(itr))
CompoundPolynomialExp(itr::Vector{Pair{T, P}}) where {T <: Number, P <: Vector} = CompoundPolynomialExp(Dict([(k, Polynomial(v)) for (k, v) in itr]))
CompoundPolynomialExp(pe::PolynomialExp) = CompoundPolynomialExp([pe.beta => pe.polynomial])

CompoundPolynomialExp(c::Number) = CompoundPolynomialExp([0 => Polynomial([c])])
Base.zero(::Type{CompoundPolynomialExp}) = CompoundPolynomialExp([0 => [0]])

Base.isequal(a::CompoundPolynomialExp, b::CompoundPolynomialExp) = issetequal(keys(a.polynomials), keys(b.polynomials)) && all([Polynomials.isapprox(v, b.polynomials[k], rtol=1E-8) for (k, v) in a.polynomials])

Base.:+(a::PolynomialExp, b::PolynomialExp) = CompoundPolynomialExp(a) + CompoundPolynomialExp(b)
function Base.:+(a::CompoundPolynomialExp, b::CompoundPolynomialExp)
    only_a = setdiff(keys(a.polynomials), keys(b.polynomials))
    only_b = setdiff(keys(b.polynomials), keys(a.polynomials))
    shared_betas = intersect(keys(a.polynomials), keys(b.polynomials))

    CompoundPolynomialExp(vcat([beta => a.polynomials[beta] for beta in only_a], 
                                [beta => b.polynomials[beta] for beta in only_b], 
                                [beta => a.polynomials[beta] + b.polynomials[beta] for beta in shared_betas]))
end
Base.:+(cpe::CompoundPolynomialExp, pe::PolynomialExp) = cpe + CompoundPolynomialExp(pe)
Base.:*(cpe::CompoundPolynomialExp, p::Polynomial) = CompoundPolynomialExp([(beta => poly * p) for (beta, poly) in cpe.polynomials])

Base.show(io::IO, pe::PolynomialExp) =  print(io, "($(string(pe.polynomial)))exp(-($(pe.beta))x)")
function Base.show(io::IO, cpe::CompoundPolynomialExp)
    res = join([PolynomialExp(poly, beta) for (beta, poly) in cpe.polynomials], " + ")
    print(io, res)
end
Base.convert(::Type{CompoundPolynomialExp}, x::Float64) = CompoundPolynomialExp(Dict([(0, Polynomial([x]))]))
Base.convert(::Type{CompoundPolynomialExp}, pe::PolynomialExp) = CompoundPolynomialExp(pe)

# Convenient function for calculating n! / k! in some of the integral terms to follow
factorial_ratio(n, k) = factorial(n) / factorial(k)

# Evaluate the integral of x^n exp(-beta x) for beta != 0
integrated_monomial(n, beta) = CompoundPolynomialExp([
                                                        0 => [beta^(-n - 1)], 
                                                        beta => [factorial_ratio(n, n_min_i) / beta^(n - n_min_i + 1) for n_min_i in 0:n]
                                                    ])

function integrate(pe::PolynomialExp)
    beta = pe.beta
    poly = pe.polynomial

    # If beta = 0, the PolynomialExp is just a polynomial, so standard polynomial integration is sufficient
    iszero(beta) && return CompoundPolynomialExp([beta => Polynomials.integrate(poly)])

    # Integrate the PolynomialExp term by term, given that x^n exp(-beta x) can be integrated exactly
    sum(poly .* [integrated_monomial(n, beta) for n in 0:degree(pe)])
end


integrate(cpe::CompoundPolynomialExp) = sum([integrate(PolynomialExp(poly, beta)) for (beta, poly) in cpe.polynomials]) # Integrate the CompoundPolynomialExp term by term

I0(cpe::CompoundPolynomialExp, t) = integrate(cpe)(t)
I1(cpe::CompoundPolynomialExp, t) = integrate(cpe * Polynomial([0, 1]))(t)


materntocpe(ν, ρ, σ2) = materntocpe(MaternGP(ν, ρ, σ2))
# In the specific case when ν = p + 0.5 (p ∈ Z), the Matern kernel can be evaluated exactly as a CompoundPolynomialExp
function materntocpe(gp::MaternGP)
    !isinteger(gp.ν - 0.5) && error("Provided Matern kernel does not have a finite CPE kernel.")

    p = Int(gp.ν - 0.5)

    beta = sqrt(2 * gp.ν) / gp.ρ
    const_factor = gp.σ2 / factorial_ratio(2 * p, p)
    base_coefs = [binomial(p, p_min_i) * factorial_ratio(p + (p - p_min_i), p) * (2 * sqrt(2 * gp.ν) / gp.ρ)^p_min_i for p_min_i in 0:p]
    CompoundPolynomialExp([beta => const_factor * Polynomial(base_coefs)])
end

# Determine a Matern Mixture with the same closed-form as a given CPE using a simple form of Gaussian elimination in the space of PolynomialExps
function cpetomaternmixture(cpe::CompoundPolynomialExp)
    poly_degs = [Polynomials.degree(poly) for (beta, poly) in cpe.polynomials]
    num_terms = sum(poly_degs .+ 1)
    matern_mixture = Vector{MaternGP}(undef, num_terms)
    next_mixture_ind = 1
    for (ind, (beta, poly)) in enumerate(cpe.polynomials)
        temp_poly = poly 
        for p in poly_degs[ind]:-1:0
            ν = p + 0.5
            ρ = sqrt(2ν) / beta

            base_cpe = materntocpe(ν, ρ, 1.0)
            base_poly = only(values(base_cpe.polynomials))
            σ2 = temp_poly[p] / base_poly[p]

            temp_poly -= σ2 * base_poly

            matern_mixture[next_mixture_ind] = MaternGP(ν, ρ, σ2)
            next_mixture_ind += 1
        end
    end
    (next_mixture_ind != num_terms + 1) && error("Matern mixture vector has not been filled up.")

    # Reduce the mixture to only non-trivial components
    filter((gp::MaternGP) -> !iszero(gp.σ2), matern_mixture)
end


struct SSM{T}
    A::Matrix{T}
    Q::Matrix{T}
    H::Matrix{T}

    SSM(A::Matrix{T}, Q::Matrix{T}, H::Matrix{T}) where T = (!(length(size(A)) == 2 && length(size(Q)) == 2 && length(size(H)) == 2) && error("All provided matrices must be 2-dimensional; provided A: $(size(A)) and Q: $(size(Q)) and H: $(size(H))")) ||
                                                            (!(size(Q)[1] == size(Q)[2]) && error("Q must be a square matrix; provided size is $(size(Q)).")) ||
                                                            (!(size(A)[2] == size(Q)[1]) && error("A and Q must have compatible sizes; provided $(size(A)) and $(size(Q))")) ||
                                                            (!(size(H)[2] == size(A)[1]) && error("H and A must have compatible sizes; provided $(size(H)) and $(size(A))")) ||
                                                            new{T}(A, Q, H)
end


# Evaluate the SSM Cov for N time steps. Since the Cov is known to have the form of a CPE, the exact coefficients can be evaluated by solving a system of linear equations.
function fit_cov(ssm::SSM)
    size(ssm.H)[1] != 1 && error("SSM needs to have one output for covariance matching to work.")
    eigen_vals = eigen(ssm.A).values

    onehot(n::Int) = [i == n ? 1 : 0 for i in 1:n]

    N = minimum(size(ssm.A))
    basis = Vector{CompoundPolynomialExp}(undef, N)

    prev_eigen = -Inf
    mult = 0
    for (ind, eig) in enumerate(eigen_vals)
        basis[ind] = PolynomialExp(onehot(mult + 1), -log(eig))

        mult = Base.isapprox(eig, prev_eigen, rtol=1E-4) ? (mult + 1) : 0
        prev_eigen = eig
    end

    M = zeros((N, N))
    v = zeros((N, 1))

    process_σ2 = only(ssm.H * lyapd(ssm.A, ssm.Q) * ssm.H')
    ssm_cov = process_σ2 * ssm.A

    for t in 1:N
        v[t] = only(ssm.H * ssm_cov * ssm.H')
        for (ind, pe) in enumerate(basis) 
            M[t,ind] = pe(t)
        end

        ssm_cov = ssm.A * ssm_cov
    end

    coefs = inv(M) * v 

    sum(coefs .* basis)
end

# Since the SSM Cov is a CPE, and a CPE is a Matern Mixture, the SSM Mixture is a Matern Mixture
ssm2GPKernel(ssm::SSM) = cpetomaternmixture(fit_cov(ssm))