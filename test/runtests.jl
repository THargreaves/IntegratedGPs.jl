using Test
using TestItems
using TestItemRunner

@run_package_tests

@testitem "Integrated Matern Kernel" begin
    using IntegratedMaternGPs
    using HCubature

    ν = 1.5
    ρ = 2.0
    σ2 = 1.0
    gp = MaternGP(ν, ρ, σ2)
    int_gp = IntegratedMaternGP(ν, ρ, σ2)

    # Test s ≠ t case
    s, t = 0.8, 1.1
    kernel_numerical = HCubature.hcubature(x -> kernel(gp, x[1], x[2]), [0.0, 0.0], [s, t])[1]
    kernel_analytical = kernel(int_gp, s, t)
    @test kernel_numerical ≈ kernel_analytical rtol = 1e-8

    # Test s = t case
    s = t = 0.8
    kernel_numerical = HCubature.hcubature(x -> kernel(gp, x[1], x[2]), [0.0, 0.0], [s, t])[1]
    kernel_analytical = kernel(int_gp, s, t)
    @test kernel_numerical ≈ kernel_analytical rtol = 1e-8
end

@testitem "LRU Cache" begin
    using IntegratedMaternGPs
    using LRUCache

    ν = 1.5
    ρ = 2.0
    σ2 = 1.0
    gp = IntegratedMaternGP(ν, ρ, σ2)

    s, t = 0.8, 1.1
    kernel(gp, s, t)
    kernel(gp, s, t)

    @test cache_info(gp.I0_cache).hits == 3
    @test cache_info(gp.I1_cache).hits == 3
end

@testitem "Cholesky Update" begin
    using IntegratedMaternGPs
    using LinearAlgebra
    using StaticArrays

    ν = 1.5
    ρ = 2.0
    σ2 = 1.0
    gp = MaternGP(ν, ρ, σ2)

    d = 10
    ts = collect(LinRange(0.0, 1.0, d + 1))

    # Create initial covariance kernel and factorise
    K = zeros(d, d)
    for i in 1:d, j in 1:d
        K[i, j] = kernel(gp, ts[i], ts[j])
    end
    F = cholesky(K)

    # Manually compute updated Cholesky
    K_new = zeros(d, d)
    for i in 1:d, j in 1:d
        K_new[i, j] = kernel(gp, ts[i + 1], ts[j + 1])
    end
    F_new = cholesky(K_new)

    # Update using in-place function
    F_updated = copy(F)
    ks = SVector{d,Float64}(K_new[d, :])
    windowed_cholesky_update!(F_updated, ks)

    # Compare results
    @test F_updated.U ≈ F_new.U rtol = 1e-8

    # Verify that no allocations are made
    @test (@allocated windowed_cholesky_update!(F, ks)) == 0
end

@testitem "Shifting Transition Mean Update" begin
    using IntegratedMaternGPs
    using LinearAlgebra
    using StableRNGs

    n = 10
    rng = StableRNG(1234)

    f = rand(rng, n)
    F_sparse = ShiftingTransition(f)

    F_dense = zeros(n, n)
    F_dense[1, :] = f
    F_dense[2:end, 1:(end - 1)] .= Matrix(I, n - 1, n - 1)

    # Out-of-place multiplication
    x = rand(rng, n)
    y_sparse = F_sparse * x
    y_dense = F_dense * x
    @test y_sparse ≈ y_dense

    # In-place multiplication
    x_sparse = copy(x)
    mul!(x_sparse, F_sparse, x_sparse)
    @test x_sparse ≈ y_dense
end

@testitem "Shifting Transition Covariance Update" begin
    using IntegratedMaternGPs
    using LinearAlgebra
    using StableRNGs

    n = 10
    rng = StableRNG(1234)

    f = rand(rng, n)
    F_sparse = ShiftingTransition(f)

    F_dense = zeros(n, n)
    F_dense[1, :] = f
    F_dense[2:end, 1:(end - 1)] .= Matrix(I, n - 1, n - 1)

    # Construct symmetric PSD matrix
    S = rand(rng, n, n)
    S = Symmetric(S * S' + I)

    # Out-of-place quadratic form
    C_sparse = quadratic_form(F_sparse, S)
    C_dense = F_dense * S * F_dense'
    @test C_sparse ≈ C_dense

    # In-place quadratic form
    S_sparse = deepcopy(S)
    quadratic_form!(S_sparse, F_sparse, S_sparse)
    @test S_sparse ≈ C_dense
end

@testitem "Expanding Transition Mean Update" begin
    using IntegratedMaternGPs
    using LinearAlgebra
    using StableRNGs

    n = 10
    rng = StableRNG(1234)

    f = rand(rng, n)
    F_sparse = ExpandingTransition(f)

    F_dense = zeros(n + 1, n)
    F_dense[1, :] = f
    F_dense[2:end, :] .= Matrix(I, n, n)

    # Out-of-place multiplication
    x = rand(rng, n)
    y_sparse = F_sparse * x
    y_dense = F_dense * x
    @test y_sparse ≈ y_dense

    # In-place multiplication
    x_sparse = Vector{Float64}(undef, n + 1)
    x_sparse[2:end] .= x
    mul!(x_sparse, F_sparse, @view x_sparse[2:end])
    @test x_sparse ≈ y_dense
end

@testitem "Expanding Transition Covariance Update" begin
    using IntegratedMaternGPs
    using LinearAlgebra
    using StableRNGs

    n = 10
    rng = StableRNG(1234)

    f = rand(rng, n)
    F_sparse = ExpandingTransition(f)

    F_dense = zeros(n + 1, n)
    F_dense[1, :] = f
    F_dense[2:end, :] .= Matrix(I, n, n)

    # Construct symmetric PSD matrix
    S = rand(rng, n, n)
    S = Symmetric(S * S' + I)

    # Out-of-place quadratic form
    C_sparse = quadratic_form(F_sparse, S)
    C_dense = F_dense * S * F_dense'
    @test C_sparse ≈ C_dense

    # In-place quadratic form
    S_sparse = Symmetric(zeros(n + 1, n + 1))
    quadratic_form!(S_sparse, F_sparse, S)
    @test S_sparse ≈ C_dense
end


@testitem "Basic CPE operations" begin
    using IntegratedMaternGPs
    using Polynomials
    
    import Base: isapprox

    PE = PolynomialExp
    CPE = CompoundPolynomialExp

    # Test that adding PolynomialExp terms gives the expected CompoundPolynomialExp
    a = PE([1, 2, 3],  1 + 3im) + 
        PE([4, 5, 6], -2 - 5im)
    expected_a = CPE([1 + 3im => [1, 2, 3], -2 - 5im => [4, 5, 6]])
    
    b = PE([-2.5, 0, 0.5, 0.9], 4 + 7im) + PE([ 2.1, 0.9, 4.3], -2 - 5im)
    expected_b = CPE([4 + 7im => [-2.5, 0, 0.5, 0.9], -2 - 5im => [2.1, 0.9, 4.3]])

    @test isequal(a, expected_a) && isequal(b, expected_b)

    # Test that adding CompoundPolynomialExp terms gives the expected CompoundPolynomialExp
    c = a + b
    expected_c = CPE([
                         1 + 3im => [   1,   2,    3], 
                        -2 - 5im => [ 6.1, 5.9, 10.3], 
                         4 + 7im => [-2.5,   0,  0.5, 0.9]
                    ])
    @test isequal(c, expected_c)

    functions_match(f, g) = all([isapprox(f(x), g(x)) for x in 0:1E-1:5])


    # Test that floats are correctly converted to CPEs and evaluating a constant expression gives the expected result
    const_val = 4.0
    const_cpe = CPE(const_val)
    @test functions_match(const_cpe, (x) -> const_val)

    # Test that CPEs which are exactly polynomials are evaluated correctly
    poly_cpe = CPE([0 => [1, 5, -7]])
    equiv_poly(x) = 1 + 5 * x - 7 * x^2
    @test functions_match(poly_cpe, equiv_poly)

    # Test that a general CPE matches the explicit function it corresponds to 
    generic_cpe = CPE([0 => [2, 3, 0, -5], 1.5 => [2.1, 3.2], 4.8 => [0, 0, 5]])
    equiv_foo(x) = (2 + 3 * x - 5 * x^3) + (2.1 + 3.2 * x) * exp(-1.5 * x) + 5 * x^2 * exp(-4.8 * x)
    @test functions_match(generic_cpe, equiv_foo)  
end

@testitem "CPE Integration" begin 
    using IntegratedMaternGPs
    using HCubature

    import Base: isapprox

    CPE = CompoundPolynomialExp

    functions_match(f, g) = all([isapprox(f(x), g(x), rtol=1E-8) for x in 0:1E-1:5])
    integrals_match(f::CPE) = functions_match(integrate(f), (x) -> hquadrature((y) -> f(y), 0.0, x)[1])

    # Test that constants integrate correctly
    const_val = 5.345
    const_cpe = CPE(const_val)
    @test integrals_match(const_cpe)

    # Test that polynomials integrate correctly
    poly_cpe = CPE([0 => [2.3, -5.425, 8.234, 0.987]])
    @test integrals_match(poly_cpe)

    # Test that some more general CPE integral evaluates to the right thing
    generic_cpe = CPE([1.567 => [5.023, -1.23], 4.254 => [0, 0.93, 0, 10.92]])
    @test integrals_match(poly_cpe)
end

@testitem "Matern to CPE" begin
    using IntegratedMaternGPs
    
    import Base: isapprox

    CPE = CompoundPolynomialExp

    functions_match(f, g) = all([isapprox(f(x), g(x), rtol=1E-8) for x in 0:1E-1:5])

    # The case ν = 0.5 is known exactly, test that it corresponds to the expected expression
    gp_p0 = MaternGP(0.5, 1.0, 1.0)
    target_p0 = CPE([1 => [1]])
    @test isequal(materntocpe(gp_p0), target_p0)

    # The case ν = 1.5 is known exactly, test that it corresponds to the expected expression
    gp_p1 = MaternGP(1.5, 1.0, 1.0)
    target_p1 = CPE([sqrt(3) => [1, sqrt(3)]])
    @test isequal(materntocpe(gp_p1), target_p1)

    # The case ν = 2.5 is known exactly, test that it corresponds to the expected expression
    gp_p2 = MaternGP(2.5, 1.0, 1.0)
    target_p2 = CPE([sqrt(5) => [1, sqrt(5), 5 / 3]])
    @test isequal(materntocpe(gp_p2), target_p2)


    # Test that some more general Matern GP has the same covariance as its corresponding CPE
    ν = 5.5
    ρ = 3.2
    σ2 = 4.5
    gp = MaternGP(ν, ρ, σ2)

    cpe = materntocpe(gp)
    @test functions_match(cpe, (t) -> kernel(gp, 0, t))
end

@testitem "CPE to Matern Mixture" begin
    using IntegratedMaternGPs
    
    import Base: isapprox
    
    CPE = CompoundPolynomialExp

    functions_match(f, g) = all([isapprox(f(x), g(x), rtol=1E-8) for x in 0:1E-1:5])

    # Take the known cases of Matern -> CPE and check that the inverse still matches
    cpe_p0 = CPE([1 => [1]])
    target_p0 = [MaternGP(0.5, 1.0, 1.0)]
    candidate_p0 = cpetomaternmixture(cpe_p0)
    @test functions_match((t) -> kernel(candidate_p0, 0.0, t), (t) -> kernel(target_p0, 0.0, t))

    cpe_p1 = CPE([sqrt(3) => [1, sqrt(3)]])
    target_p1 = [MaternGP(1.5, 1.0, 1.0)]
    candidate_p1 = cpetomaternmixture(cpe_p1)
    @test functions_match((t) -> kernel(candidate_p1, 0.0, t), (t) -> kernel(target_p1, 0.0, t))

    cpe_p2 = CPE([sqrt(5) => [1, sqrt(5), 5 / 3]])
    target_p2 = [MaternGP(2.5, 1.0, 1.0)]
    candidate_p2 = cpetomaternmixture(cpe_p2)
    @test functions_match((t) -> kernel(candidate_p2, 0.0, t), (t) -> kernel(target_p2, 0.0, t))

    # Test that some more general CPE has the same form as its Matern Mixture
    cpe_general = CPE([0.1 => [9.3, 1.23], 1.542 => [8.432, 0.32, 7.543], 6.222 => [0.0, 1.11, 0.234]])
    candidate_general = cpetomaternmixture(cpe_general)
    @test functions_match(cpe_general, (t) -> kernel(candidate_general, 0.0, t))
end

@testitem "SSM to Matern Mixture" begin
    using IntegratedMaternGPs

    import Base: isapprox
    using LinearAlgebra, MatrixEquations

    CPE = CompoundPolynomialExp

    functions_match(f, g) = all([isapprox(f(x), g(x), rtol=1E-8) for x in 0:1E-1:5])

    # The AR(1) process has a known closed form covariance. Test that it matches the equivalent Matern Mixture.
    a = 0.9
    σ2 = 1.0
    ar_1 = SSM([a;;], [σ2;;], [1.0;;])
    candidate_kernel = ssm2GPKernel(ar_1)
    target_cov = CPE([-log(a) => [σ2 / (1 - a^2)]])

    @test functions_match((t) -> kernel(candidate_kernel, 0.0, t), target_cov)

    general_A = [0.52 -0.32; -0.80 0.32]
    general_Q = [3.58 1.78; 1.78 4.56]
    general_H = [5.43 -0.67;]
    any([abs(z) > 1 for z in eigen(general_A).values]) && error("A matrix should not have poles outside the unit circle; found $(eigen(general_A).values) with magnitude $([abs(z) for z in eigen(general_A).values])")
    det(general_Q) <= 0 && error("Q must be positive definite.")

    general_ssm = SSM(general_A, general_Q, general_H)
    general_kernel = ssm2GPKernel(general_ssm)
    stationary_σ2 = only(general_H * lyapd(general_A, general_Q) * general_H')

    @test functions_match((t) -> kernel(general_kernel, 0.0, t), (t) -> stationary_σ2 * general_H * exp(general_A * t) * general_H')
end