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


@testitem "Basic CPE operations" begin
    using IntegratedMaternGPs
    using Polynomials

    PE = PolynomialExp
    CPE = CompoundPolynomialExp

    a = PE([1, 2, 3],  1 + 3im) + 
        PE([4, 5, 6], -2 - 5im)
    expected_a = CPE([1 + 3im => [1, 2, 3], -2 - 5im => [4, 5, 6]])
    
    b = PE([-2.5, 0, 0.5, 0.9], 4 + 7im) + PE([ 2.1, 0.9, 4.3], -2 - 5im)
    expected_b = CPE([4 + 7im => [-2.5, 0, 0.5, 0.9], -2 - 5im => [2.1, 0.9, 4.3]])

    # Test that adding PolynomialExp terms gives the expected CompoundPolynomialExp
    @test isequal(a, expected_a) && isequal(b, expected_b)

    c = a + b
    expected_c = CPE([
                         1 + 3im => [   1,   2,    3], 
                        -2 - 5im => [ 6.1, 5.9, 10.3], 
                         4 + 7im => [-2.5,   0,  0.5, 0.9]
                    ])
    # Test that adding CompoundPolynomialExp terms gives the expected CompoundPolynomialExp
    @test isequal(c, expected_c)

    functions_match(f, g) = all(isapprox(f(x), g(x)) for x in 0:1E-2:5)


    const_val = 4.0
    const_cpe = CPE(const_val)
    # Test that floats are correctly converted to CPEs and evaluating a constant expression gives the expected result
    @test functions_match(const_cpe, (x) -> const_val)

    poly_cpe = CPE([0 => [1, 5, -7]])
    equiv_poly(x) = 1 + 5 * x - 7 * x^2
    @test functions_match(poly_cpe, equiv_poly)

    generic_cpe = CPE([0 => [2, 3, 0, -5], 1.5 => [2.1, 3.2], 4.8 => [0, 0, 5]])
    equiv_foo(x) = (2 + 3 * x - 5 * x^3) + (2.1 + 3.2 * x) * exp(-1.5 * x) + 5 * x^2 * exp(-4.8 * x)
    @test functions_match(generic_cpe, equiv_foo)  
end

@testitem "CPE Integration" begin 
    using IntegratedMaternGPs
    using HCubature

    CPE = CompoundPolynomialExp

    functions_match(f, g) = all(isapprox(f(x), g(x), rtol=1E-8) for x in 0:1E-2:5)
    integrals_match(f::CPE) = functions_match(integrate(f), (x) -> hquadrature((y) -> f(y), 0.0, x)[1])

    const_val = 5.345
    const_cpe = CPE([0 => [const_val]])
    @test integrals_match(const_cpe)

    poly_cpe = CPE([0 => [2.3, -5.425, 8.234, 0.987]])
    @test integrals_match(poly_cpe)

    generic_cpe = CPE([1.567 => [5.023, -1.23], 4.254 => [0, 0.93, 0, 10.92]])
    @test integrals_match(poly_cpe)
end