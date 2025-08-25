using Polynomials, HCubature, LinearAlgebra

import Base: isapprox

export PolynomialExp, CompoundPolynomialExp
export +, show, isequal, isapprox
export integrate, materntocpe, cpetomaternmixture

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

oneunit(PolynomialExp) = PolynomialExp(Polynomial([1.0]), 0.0)
Base.:*(c::Number, pe::PolynomialExp) = PolynomialExp(c * pe.polynomial, pe.beta);
Base.:/(pe::PolynomialExp, c::Number) = PolynomialExp(pe.polynomial / c, pe.beta);
degree(pe::PolynomialExp) = Polynomials.degree(pe.polynomial)

(cpe::CompoundPolynomialExp)(x) = sum(PolynomialExp(poly, beta)(x) for (beta, poly) in cpe.polynomials)

zero(CompoundPolynomialExp) = CompoundPolynomialExp(Dict{Number, Polynomial}())
CompoundPolynomialExp(itr::Vector{Pair{T, P}}) where {T <: Number, P <: Polynomial} = CompoundPolynomialExp(Dict(itr))
CompoundPolynomialExp(itr::Vector{Pair{T, P}}) where {T <: Number, P <: Vector} = CompoundPolynomialExp(Dict([(k, Polynomial(v)) for (k, v) in itr]))
CompoundPolynomialExp(pe::PolynomialExp) = CompoundPolynomialExp([pe.beta => pe.polynomial])

CompoundPolynomialExp(c::Number) = CompoundPolynomialExp([0 => Polynomial([c])])

Base.isequal(a::CompoundPolynomialExp, b::CompoundPolynomialExp) = issetequal(keys(a.polynomials), keys(b.polynomials)) && all([Polynomials.isapprox(v, b.polynomials[k], rtol=1E-8) for (k, v) in a.polynomials])

Base.:+(a::PolynomialExp, b::PolynomialExp) = CompoundPolynomialExp(a) + CompoundPolynomialExp(b)
function Base.:+(a::CompoundPolynomialExp, b::CompoundPolynomialExp)
    only_a = setdiff(keys(a.polynomials), keys(b.polynomials))
    only_b = setdiff(keys(b.polynomials), keys(a.polynomials))
    shared_betas = intersect(keys(a.polynomials), keys(b.polynomials))
    new_polynomials = Dict{Number, Polynomial}()

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
Base.:*(cpe::CompoundPolynomialExp, p::Polynomial) = CompoundPolynomialExp([(beta => poly * p) for (beta, poly) in cpe.polynomials])

Base.show(io::IO, pe::PolynomialExp) =  print(io, "($(string(pe.polynomial)))exp(-($(pe.beta))x)")
function Base.show(io::IO, cpe::CompoundPolynomialExp)
    res = join([PolynomialExp(poly, beta) for (beta, poly) in cpe.polynomials], " + ")
    print(io, res)
end
Base.convert(::Type{CompoundPolynomialExp}, x::Float64) = CompoundPolynomialExp(Dict([(0, Polynomial([x]))]))

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


integrate(cpe::CompoundPolynomialExp) = sum([integrate(PolynomialExp(poly, beta)) for (beta, poly) in cpe.polynomials])

I0(cpe::CompoundPolynomialExp, t) = integrate(cpe)(t)
I1(cpe::CompoundPolynomialExp, t) = integrate(cpe * Polynomial([0, 1]))(t)

function materntocpe(gp::MaternGP)
    !Base.isapprox(gp.ν % 1.0, 0.5; rtol=1E-8) && error("Provided Matern kernel does not have a finite CPE kernel.")

    p::Int = floor(gp.ν)

    beta = sqrt(2 * gp.ν) / gp.ρ
    const_factor = gp.σ2 / factorial_ratio(2 * p, p)
    base_coefs = [binomial(p, p_min_i) * factorial_ratio(p + (p - p_min_i), p) * (2 * sqrt(2 * gp.ν) / gp.ρ)^p_min_i for p_min_i in 0:p]
    CompoundPolynomialExp([beta => const_factor * Polynomial(base_coefs)])
end

function cpetomaternmixture(cpe::CompoundPolynomialExp)
    res = Vector{MaternGP}()
    for (beta, poly) in cpe.polynomials
        new_poly = poly 
        while sum(new_poly[0:end].^2) > 1E-8
            p = Polynomials.degree(new_poly)
            ν = p + 0.5
            ρ = sqrt(2ν) / beta

            matern_base = MaternGP(ν, ρ, 1.0)
            base_cpe = materntocpe(matern_base)
            base_poly = base_cpe.polynomials[beta]
            σ2 = new_poly[p] / base_poly[p]
            base_poly *= σ2

            new_poly -= base_poly

            push!(res, MaternGP(ν, ρ, σ2))
        end
    end
    res
end


struct SSM{T}
    A::Matrix{T}
    Q::Matrix{T}
    H::Matrix{T}
end

function fit_cov(ssm::SSM)
    size(ssm.H)[1] != 1 && error("SSM needs to have one output for covariance matching to work.")
    eigen_vals = eigen(ssm.A)[1]

    one_hot(n) = [i == n ? 1 : 0 for i in 1:n]

    prev_eigen = -Inf
    mult = 0
    basis = Vector{PolynomialExp}()
    for eig in eigen_vals
        push!(basis, PolynomialExp(onehot(mult + 1), -eig))
        if is_approx(eig, prev_eigen, rtol=1E-4)
            mult = 0
        else
            mult += 1
        end
        prev_eigen = eig
    end
    N = 10

    A = zeros((N, length(basis)))
    v = zeros((N,1))

    ssm_cov = Q

    for t in 1:N
        v[t] = ssm.H * ssm_cov * ssm.H'
        for (ind, pe) in enumerate(basis) 
            A[t,ind] = pe(t)
        end

        ssm_cov = ssm.A * ssm_cov * ssm.A' + ssm.Q
    end

    coefs = inv(A' * A) * A' * v 

    res = sum(coefs .* basis)
end

function ssm2GPKernel(ssm::SSM)
    cpetomaternmixture(fit_cov(ssm))
end