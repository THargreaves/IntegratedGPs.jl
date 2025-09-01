@testitem "Matern kernel case distinction" begin
    using IntegratedMaternGPs

    import Base: isapprox

    functions_match(f, g) = all(x -> isapprox(f(x), g(x); rtol=1E-8), 0:1E-1:5)

    # Test if the Matern GPs are separated into cases when ν = p + 0.5
    ν1 = 1.6
    ν2 = 1.5
    ρ = 0.8
    σ2 = 5.4
    gp1 = constructmatern(ν1, ρ, σ2)
    gp2 = constructmatern(ν2, ρ, σ2)

    @test typeof(gp1) <: AbstractMaternGP && isa(gp1, MaternGP)
    @test typeof(gp2) <: AbstractMaternGP && isa(gp2, CPEMaternGP)

    # Test if the two Matern GP cases have the same kernel
    ν = 3.5
    ρ = 1.2
    σ2 = 4.6
    gp_cpe = CPEMaternGP(ν, ρ, σ2)
    gp_gen = MaternGP(ν, ρ, σ2)

    @test functions_match((t) -> kernel(gp_cpe, 0.0, t), (t) -> kernel(gp_gen, 0.0, t))
end

@testitem "Matern kernel I0 & I1 verification" begin
    using IntegratedMaternGPs
    using HCubature

    import Base: isapprox

    functions_match(f, g) = all(x -> isapprox(f(x), g(x); rtol=1E-8), 0:1E-1:5)

    # Test that the I0 and I1 functions provide the same answers as numerical methods
    ν = 4.3
    ρ = 2.1
    σ2 = 5.3

    gp = constructmatern(ν, ρ, σ2)
    igp = integrate(gp)

    @test functions_match(
        t -> I0(igp, t), t -> hquadrature(s -> kernel(gp, 0, s), 0.0, t)[1]
    )
    @test functions_match(
        t -> I1(igp, t), t -> hquadrature(s -> s * kernel(gp, 0, s), 0.0, t)[1]
    )
end

@testitem "Integrated Matern Kernel" begin
    using IntegratedMaternGPs
    using HCubature

    ν = 1.5
    ρ = 2.0
    σ2 = 1.0
    gp = MaternGP(ν, ρ, σ2)
    int_gp = integrate(gp)

    # Test s ≠ t case
    s, t = 0.8, 1.1
    kernel_numeric = HCubature.hcubature(x -> kernel(gp, x[1], x[2]), [0.0, 0.0], [s, t])[1]
    kernel_analytical = kernel(int_gp, s, t)
    @test kernel_numeric ≈ kernel_analytical rtol = 1e-8

    # Test s = t case
    s = t = 0.8
    kernel_numeric = HCubature.hcubature(x -> kernel(gp, x[1], x[2]), [0.0, 0.0], [s, t])[1]
    kernel_analytical = kernel(int_gp, s, t)
    @test kernel_numeric ≈ kernel_analytical rtol = 1e-8
end

@testitem "Integrated CPE Matern Kernel" begin
    using IntegratedMaternGPs
    using HCubature

    ν = 2.5
    ρ = 2.0
    σ2 = 1.0
    gp = CPEMaternGP(ν, ρ, σ2)
    int_gp = integrate(gp)

    # Test s ≠ t case
    s, t = 0.8, 1.1
    kernel_numeric = HCubature.hcubature(x -> kernel(gp, x[1], x[2]), [0.0, 0.0], [s, t])[1]
    kernel_analytical = kernel(int_gp, s, t)
    @test kernel_numeric ≈ kernel_analytical rtol = 1e-8

    # Test s = t case
    s = t = 0.8
    kernel_numeric = HCubature.hcubature(x -> kernel(gp, x[1], x[2]), [0.0, 0.0], [s, t])[1]
    kernel_analytical = kernel(int_gp, s, t)
    @test kernel_numeric ≈ kernel_analytical rtol = 1e-8
end

@testitem "Integrated Rational Quadratic Kernel" begin
    using IntegratedMaternGPs
    using HCubature

    α = 1.3
    l = 2.0
    σ2 = 1.0
    gp = RationalQuadraticGP(α, l, σ2)
    int_gp = integrate(gp)

    # Test s ≠ t case
    s, t = 0.8, 1.1
    kernel_numeric = HCubature.hcubature(x -> kernel(gp, x[1], x[2]), [0.0, 0.0], [s, t])[1]
    kernel_analytical = kernel(int_gp, s, t)
    @test kernel_numeric ≈ kernel_analytical rtol = 1e-8

    # Test s = t case
    s = t = 0.8
    kernel_numeric = HCubature.hcubature(x -> kernel(gp, x[1], x[2]), [0.0, 0.0], [s, t])[1]
    kernel_analytical = kernel(int_gp, s, t)
    @test kernel_numeric ≈ kernel_analytical rtol = 1e-8

    # Special case where α = 1  
    α = 1.0
    gp = RationalQuadraticGP(α, l, σ2)
    int_gp = integrate(gp)
    s, t = 0.8, 1.1
    kernel_numeric = HCubature.hcubature(x -> kernel(gp, x[1], x[2]), [0.0, 0.0], [s, t])[1]
    kernel_analytical = kernel(int_gp, s, t)
    @test kernel_numeric ≈ kernel_analytical rtol = 1e-8
end

@testitem "LRU Cache" begin
    using IntegratedMaternGPs
    using LRUCache

    ν = 1.5
    ρ = 2.0
    σ2 = 1.0
    gp = integrate(MaternGP(ν, ρ, σ2))

    s, t = 0.8, 1.1
    kernel(gp, s, t)
    kernel(gp, s, t)

    @test cache_info(gp.I0_cache).hits == 3
    @test cache_info(gp.I1_cache).hits == 3
end

@testitem "Matern to CPE" begin
    using IntegratedMaternGPs

    import Base: isapprox

    CPE = CompoundPolynomialExp

    functions_match(f, g) = all(x -> isapprox(f(x), g(x); rtol=1E-8), 0:1E-1:5)

    # The case ν = 0.5 is known exactly, test that it corresponds to the expected expression
    gp_p0 = MaternGP(0.5, 1.0, 1.0)
    target_p0 = CPE(1 => [1])
    @test isequal(materntocpe(gp_p0), target_p0)

    # The case ν = 1.5 is known exactly, test that it corresponds to the expected expression
    gp_p1 = MaternGP(1.5, 1.0, 1.0)
    target_p1 = CPE(sqrt(3) => [1, sqrt(3)])
    @test isequal(materntocpe(gp_p1), target_p1)

    # The case ν = 2.5 is known exactly, test that it corresponds to the expected expression
    gp_p2 = MaternGP(2.5, 1.0, 1.0)
    target_p2 = CPE(sqrt(5) => [1, sqrt(5), 5 / 3])
    @test isequal(materntocpe(gp_p2), target_p2)

    # Test that some more general Matern GP has the same covariance as its corresponding CPE
    ν = 5.5
    ρ = 3.2
    σ2 = 4.5
    gp = MaternGP(ν, ρ, σ2)

    cpe = materntocpe(gp)
    @test functions_match(cpe, t -> kernel(gp, 0, t))
end

@testitem "CPE to Matern Mixture" begin
    using IntegratedMaternGPs

    import Base: isapprox

    CPE = CompoundPolynomialExp

    functions_match(f, g) = all(x -> isapprox(f(x), g(x); rtol=1E-8), 0:1E-1:5)

    # Take the known cases of Matern -> CPE and check that the inverse still matches
    cpe_p0 = CPE(1 => [1])
    target_p0 = Mixture([MaternGP(0.5, 1.0, 1.0)])
    candidate_p0 = cpetomaternmixture(cpe_p0)
    @test functions_match(t -> kernel(candidate_p0, 0.0, t), t -> kernel(target_p0, 0.0, t))

    cpe_p1 = CPE(sqrt(3) => [1, sqrt(3)])
    target_p1 = Mixture([MaternGP(1.5, 1.0, 1.0)])
    candidate_p1 = cpetomaternmixture(cpe_p1)
    @test functions_match(t -> kernel(candidate_p1, 0.0, t), t -> kernel(target_p1, 0.0, t))

    cpe_p2 = CPE(sqrt(5) => [1, sqrt(5), 5 / 3])
    target_p2 = Mixture([MaternGP(2.5, 1.0, 1.0)])
    candidate_p2 = cpetomaternmixture(cpe_p2)
    @test functions_match(t -> kernel(candidate_p2, 0.0, t), t -> kernel(target_p2, 0.0, t))

    # Test that some more general CPE has the same form as its Matern Mixture
    cpe_general = CPE([
        0.1 => [9.3, 1.23], 1.542 => [8.432, 0.32, 7.543], 6.222 => [0.0, 1.11, 0.234]
    ])
    candidate_general = cpetomaternmixture(cpe_general)
    @test functions_match(cpe_general, t -> kernel(candidate_general, 0.0, t))
end