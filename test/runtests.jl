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
