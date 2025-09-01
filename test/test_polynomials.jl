@testitem "Basic CPE operations" begin
    using IntegratedMaternGPs
    using Polynomials

    import Base: isapprox

    PE = PolynomialExp
    CPE = CompoundPolynomialExp

    # Test that adding PolynomialExp terms gives the expected CompoundPolynomialExp
    a = PE([1, 2, 3], 1 + 3im) + PE([4, 5, 6], -2 - 5im)
    expected_a = CPE([1 + 3im => [1, 2, 3], -2 - 5im => [4, 5, 6]])

    b = PE([-2.5, 0, 0.5, 0.9], 4 + 7im) + PE([2.1, 0.9, 4.3], -2 - 5im)
    expected_b = CPE([4 + 7im => [-2.5, 0, 0.5, 0.9], -2 - 5im => [2.1, 0.9, 4.3]])

    @test isequal(a, expected_a) && isequal(b, expected_b)

    # Test that adding CompoundPolynomialExp terms gives the expected CompoundPolynomialExp
    c = a + b
    expected_c = CPE([
        1 + 3im => [1, 2, 3], -2 - 5im => [6.1, 5.9, 10.3], 4 + 7im => [-2.5, 0, 0.5, 0.9]
    ])
    @test isequal(c, expected_c)

    functions_match(f, g) = all(x -> isapprox(f(x), g(x)), 0:1E-1:5)

    # Test that floats are correctly converted to CPEs and evaluating a constant 
    # expression gives the expected result
    const_val = 4.0
    const_cpe = CPE(const_val)
    @test functions_match(const_cpe, (x) -> const_val)

    # Test that CPEs which are exactly polynomials are evaluated correctly
    poly_cpe = CPE(0 => [1, 5, -7])
    equiv_poly(x) = 1 + 5 * x - 7 * x^2
    @test functions_match(poly_cpe, equiv_poly)

    # Test that a general CPE matches the explicit function it corresponds to 
    generic_cpe = CPE([0 => [2, 3, 0, -5], 1.5 => [2.1, 3.2], 4.8 => [0, 0, 5]])
    equiv_foo(x) =
        (2 + 3 * x - 5 * x^3) + (2.1 + 3.2 * x) * exp(-1.5 * x) + 5 * x^2 * exp(-4.8 * x)
    @test functions_match(generic_cpe, equiv_foo)
end

@testitem "CPE Integration" begin
    using IntegratedMaternGPs
    using HCubature

    import Base: isapprox

    CPE = CompoundPolynomialExp

    functions_match(f, g) = all(x -> isapprox(f(x), g(x); rtol=1E-8), 0:1E-1:5)
    integrals_match(f::CPE) =
        functions_match(integrate(f), x -> hquadrature(y -> f(y), 0.0, x)[1])

    # Test that constants integrate correctly
    const_val = 5.345
    const_cpe = CPE(const_val)
    @test integrals_match(const_cpe)

    # Test that polynomials integrate correctly
    poly_cpe = CPE(0 => [2.3, -5.425, 8.234, 0.987])
    @test integrals_match(poly_cpe)

    # Test that some more general CPE integral evaluates to the right thing
    generic_cpe = CPE([1.567 => [5.023, -1.23], 4.254 => [0, 0.93, 0, 10.92]])
    @test integrals_match(poly_cpe)
end

@testitem "SSM to Matern Mixture" begin
    using IntegratedMaternGPs

    import Base: isapprox
    using LinearAlgebra, MatrixEquations

    CPE = CompoundPolynomialExp

    functions_match(f, g) = all(x -> isapprox(f(x), g(x); rtol=1E-8), 0:1E-1:5)
    functions_match_at_int(f, g) = all(x -> isapprox(f(x), g(x); rtol=1E-8), 0:1:10)

    # The AR(1) process has a known closed form covariance. 
    # Test that it matches the equivalent Matern Mixture.
    a = 0.9
    σ2 = 1.0
    ar_1 = SSM([a;;], [σ2;;], [1.0;;])
    candidate_kernel = ssm2GPKernel(ar_1)
    target_cov = CPE(-log(a) => [σ2 / (1 - a^2)])

    @test functions_match(t -> kernel(candidate_kernel, 0.0, t), target_cov)

    # Check if some other SSM has the same covariance as its corresponding kernel
    general_A = [0.52 -0.32; -0.20 0.60]
    general_Q = [3.58 1.78; 1.78 4.56]
    general_H = [5.43 -0.67;]
    any(z -> abs(z) > 1, eigen(general_A).values) &&
        error("A matrix should not have poles outside the 
              unit circle; found $(eigen(general_A).values) with 
              magnitude $([abs(z) for z in eigen(general_A).values])")
    det(general_Q) <= 0 && error("Q must be positive definite.")

    general_ssm = SSM(general_A, general_Q, general_H)
    general_kernel = ssm2GPKernel(general_ssm)
    radial_σ2 = only(general_H * lyapd(general_A, general_Q) * general_H')

    @test functions_match_at_int(
        t -> real(kernel(general_kernel, 0.0, t)),
        t -> radial_σ2 * real(only(general_H * general_A^t * general_H')),
    )

    # TODO: Implement the case corresponding to complex Matern arguments
end