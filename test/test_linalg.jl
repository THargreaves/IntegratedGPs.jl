@testitem "Cholesky Update" begin
    using IntegratedGPs
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
    using IntegratedGPs
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
    using IntegratedGPs
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
    using IntegratedGPs
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
    using IntegratedGPs
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

@testitem "Shifting Offset Transition Mean Update" begin
    using IntegratedGPs
    using LinearAlgebra
    using StableRNGs

    n = 10
    rng = StableRNG(1234)

    f = rand(rng, n) / (n + 1)  # Ensure sum(f) < 1
    f̄ = sum(f)
    F_sparse = ShiftingOffsetTransition(f)

    F_dense = zeros(n + 1, n + 1)
    F_dense[1, 1:n] = f
    F_dense[1, n + 1] = 1 - f̄
    F_dense[2:end, 1:(end - 1)] .= Matrix(I, n, n)

    # Out-of-place multiplication
    x = rand(rng, n + 1)
    y_sparse = F_sparse * x
    y_dense = F_dense * x
    @test y_sparse ≈ y_dense

    # In-place multiplication
    x_sparse = copy(x)
    mul!(x_sparse, F_sparse, x_sparse)
    @test x_sparse ≈ y_dense
end

@testitem "Shifting Offset Transition Covariance Update" begin
    using IntegratedGPs
    using LinearAlgebra
    using StableRNGs

    n = 10
    rng = StableRNG(1234)

    f = rand(rng, n) / (n + 1)  # Ensure sum(f) < 1
    f̄ = sum(f)
    F_sparse = ShiftingOffsetTransition(f)

    F_dense = zeros(n + 1, n + 1)
    F_dense[1, 1:n] = f
    F_dense[1, n + 1] = 1 - f̄
    F_dense[2:end, 1:(end - 1)] .= Matrix(I, n, n)

    # Construct symmetric PSD matrix
    S = rand(rng, n + 1, n + 1)
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

@testitem "Expanding Offset Transition Mean Update" begin
    using IntegratedGPs
    using LinearAlgebra
    using StableRNGs

    n = 10
    rng = StableRNG(1234)

    f = rand(rng, n) / (n + 1)  # Ensure sum(f) < 1
    f̄ = sum(f)
    F_sparse = ExpandingOffsetTransition(f)

    F_dense = zeros(n + 2, n + 1)
    F_dense[1, 1:n] = f
    F_dense[1, n + 1] = 1 - f̄
    F_dense[2:end, :] .= Matrix(I, n + 1, n + 1)

    # Out-of-place multiplication
    x = rand(rng, n + 1)
    y_sparse = F_sparse * x
    y_dense = F_dense * x
    @test y_sparse ≈ y_dense

    # In-place multiplication
    x_sparse = Vector{Float64}(undef, n + 2)
    x_sparse[2:end] .= x
    mul!(x_sparse, F_sparse, @view x_sparse[2:end])
    @test x_sparse ≈ y_dense
end

@testitem "Expanding Offset Transition Covariance Update" begin
    using IntegratedGPs
    using LinearAlgebra
    using StableRNGs

    n = 10
    rng = StableRNG(1234)

    f = rand(rng, n) / (n + 1)  # Ensure sum(f) < 1
    f̄ = sum(f)
    F_sparse = ExpandingOffsetTransition(f)

    F_dense = zeros(n + 2, n + 1)
    F_dense[1, 1:n] = f
    F_dense[1, n + 1] = 1 - f̄
    F_dense[2:end, :] .= Matrix(I, n + 1, n + 1)

    # Construct symmetric PSD matrix
    S = rand(rng, n + 1, n + 1)
    S = Symmetric(S * S' + I)

    # Out-of-place quadratic form
    C_sparse = quadratic_form(F_sparse, S)
    C_dense = F_dense * S * F_dense'
    @test C_sparse ≈ C_dense

    # In-place quadratic form
    S_sparse = Symmetric(zeros(n + 2, n + 2))
    quadratic_form!(S_sparse, F_sparse, S)
    @test S_sparse ≈ C_dense
end

@testitem "Expanding Markovian Transition Test" begin
    using IntegratedGPs
    using LinearAlgebra
    using StableRNGs

    n = 10
    rng = StableRNG(1234)

    f = rand(rng, n - 1) / n  # Ensure sum(f) < 1
    f̄ = sum(f)
    F_sparse = ExpandingMarkovianTransition(f)

    F_dense = zeros(n + 1, n)
    F_dense[1, 1:(n - 1)] = f
    F_dense[1, n] = 1 - f̄
    F_dense[2:end, :] .= Matrix(I, n, n)

    # Verify all elements match
    @test all(F_sparse[i, j] == F_dense[i, j] for i in 1:(n + 1), j in 1:n)

    ###########################
    #### MAT-VEC MUL TESTS ####
    ###########################

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

    ##############################
    #### QUADRATIC FORM TESTS ####
    ##############################

    # Construct symmetric PSD matrix
    S = rand(rng, n, n)
    S = Symmetric(S * S' + I)

    # Out-of-place quadratic form
    C_sparse = quadratic_form(F_sparse, S)
    C_dense = F_dense * S * F_dense'
    @test C_sparse ≈ C_dense

    # In-place quadratic form
    S_sparse_data = Matrix{Float64}(undef, n + 1, n + 1)
    S_sparse_data[2:end, 2:end] .= S
    S_sparse = Symmetric(S_sparse_data)
    S_sparse_sub = Symmetric(@view S_sparse_data[2:end, 2:end])
    quadratic_form!(S_sparse, F_sparse, S_sparse_sub)
    @test S_sparse ≈ C_dense
end

@testitem "Shifting Markovian Transition Test" begin
    using IntegratedGPs
    using LinearAlgebra
    using StableRNGs

    n = 10
    rng = StableRNG(1234)

    f = rand(rng, n - 1) / n  # Ensure sum(f) < 1
    f̄ = sum(f)
    F_sparse = ShiftingMarkovianTransition(f)

    F_dense = zeros(n, n)
    F_dense[1, 1:(n - 1)] = f
    F_dense[1, n] = 1 - f̄
    F_dense[2:n, 1:(n - 1)] .= Matrix(I, n - 1, n - 1)

    # Verify all elements match
    @test all(F_sparse[i, j] == F_dense[i, j] for i in 1:n, j in 1:n)

    ###########################
    #### MAT-VEC MUL TESTS ####
    ###########################

    # Out-of-place multiplication
    x = rand(rng, n)
    y_sparse = F_sparse * x
    y_dense = F_dense * x
    @test y_sparse ≈ y_dense

    # In-place multiplication
    x_sparse = copy(x)
    mul!(x_sparse, F_sparse, x_sparse)
    @test x_sparse ≈ y_dense

    ##############################
    #### QUADRATIC FORM TESTS ####
    ##############################

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
