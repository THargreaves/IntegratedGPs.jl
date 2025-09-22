export cpe_mixture_to_ssm

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
    return Mixture(filter((gp::CPEMaternGP) -> !iszero(gp.σ2), matern_mixture))
end

# Evaluate the SSM Cov for N time steps. Since the Cov is known to have the form of a CPE, 
# the exact coefficients can be evaluated by solving a system of linear equations.
function fit_cov(ssm::SSM{T}) where {T}
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

function cpe_mixture_to_ssm(gp::Mixture{CPEMaternGP}, Ts::T=1.0) where {T}
    N = sum(
        sum([
            [Polynomials.degree(poly) + 1 for poly in gp_component.cpe.value_lookup] for
            gp_component in gp.mixture
        ]),
    )
    poles = zeros(N)
    i = 1
    for gp_component in gp.mixture
        for (beta, poly) in gp_component.cpe.polynomials
            for _ in 0:Polynomials.degree(poly)
                poles[i] = exp(-beta)
                i += 1
            end
        end
    end

    ar_poly_coeffs = fromroots(poles).coeffs
    A = zeros((N, N))
    for i in 1:(N - 1)
        A[i + 1, i] = 1
    end
    A[1, :] = -reverse(ar_poly_coeffs[1:N])
    Q = zeros(T, (N, N))
    Q[1, 1] = 1.0

    v = [kernel(gp, 0.0, i) for i in 1:N]
    Σ = lyapd(A, Q)
    M = [A^i * Σ for i in 1:N]

    E(vec) = sum([(only(vec' * M[i] * vec) - v[i])^2 for i in 1:N])
    G(vec) = 4 * sum([(only(vec' * M[i] * vec) - v[i]) * M[i] * vec for i in 1:N])
    function H(vec)
        return 4 * sum([
            ((only(vec' * M[i] * vec) - v[i]) * M[i] + 2 * M[i] * vec * vec' * M[i]') for
            i in 1:N
        ])
    end

    vec0 = ones((N, 1)) * 1E1

    vec = vec0
    cnt = 0
    while E(vec) > 1E-12 && cnt < 100_000
        vec = vec - inv(H(vec)) * G(vec)
        cnt += 1
    end
    println("SSM construction error: $(E(vec)) in $cnt steps")

    H = zeros((1, N))
    Q[1, 1] *= vec[1]^2
    H[1, :] = vec' / vec[1]
    return SSM(A, Q, H)
end

# Since the SSM Cov is a CPE, and a CPE is a Matern Mixture, 
# the SSM Mixture is a Matern Mixture
ssm2GPKernel(ssm::SSM{T}) where {T} = cpetomaternmixture(fit_cov(ssm))
