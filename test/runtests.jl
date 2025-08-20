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
