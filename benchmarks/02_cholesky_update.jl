using IntegratedMaternGPs
using LinearAlgebra
using Plots

function rand_psd(D::Int)
    A = randn(D, D)
    return A * A' + I
end

function naive_cholesky(A, x)
    return A_chol = cholesky!(A)
end

function cholesky_update(A_chol::Cholesky, A_new::Vector, x::Vector)
    return windowed_cholesky_update!(A_chol, A_new)
end

D = 100

A = rand_psd(D + 1);
A_new = A[2:end, end]
A = copy(A[1:(end - 1), 1:(end - 1)])
A_chol = cholesky(A)
x = rand(D);

display(@benchmark naive_cholesky(A_copy, $x) setup = begin
    A_copy = copy($A)
end evals = 1)

display(
    @benchmark cholesky_update(A_chol, $A_new, $x) setup = begin
        A_chol = cholesky(copy($A))
    end evals = 1
)

Ds = [10, 20, 50, 100, 200, 500]
naive_times = Float64[]
update_times = Float64[]

for D in Ds
    println("Benchmarking D = $D")
    A = rand_psd(D + 1)
    A_new = A[2:end, end]
    A = copy(A[1:(end - 1), 1:(end - 1)])
    A_chol = cholesky(A)
    x = rand(D)

    naive_time = @benchmark naive_cholesky(A_copy, $x) setup = begin
        A_copy = copy($A)
    end evals = 1
    push!(naive_times, median(naive_time).time / 1e6)

    update_time = @benchmark cholesky_update(A_chol, $A_new, $x) setup = begin
        A_chol = cholesky(copy($A))
    end evals = 1
    push!(update_times, median(update_time).time / 1e6)
end

# Plot speed-up
speedups = naive_times ./ update_times
plot(
    Ds,
    speedups;
    marker=:o,
    xlabel="Matrix Size D",
    ylabel="Speed-up Factor",
    title="Cholesky Update Speed-up over Naive Cholesky",
    legend=false,
    xscale=:log10,
    yscale=:log2,
    yticks=[1, 2, 4, 8],
    ylims=(1, 8),
)

function windowed_lower_cholesky_update!(F::Cholesky, ks::AbstractVector)
    """
    Update the Cholesky factorization of a covariance kernel matrix from one window shift.

    The current factorization is stored in `F` which is updated in-place without any
    allocations. `ks` stores entries corresponding to the new row of the covariance kernel
    matrix.

    For lower triangular factorization: A = L*L'
    """
    d = size(F.L, 1)
    L = F.L
    
    # Step 1: Remove first column using Givens rotations
    n = d
    v = similar(L, n - 1)
    copyto!(v, @view L[2:end, 1])  # Copy first column (excluding diagonal)

    @inbounds for i in 2:n
        # Compute Givens rotation
        c, s, r = LinearAlgebra.givensAlgorithm(L[i, i], v[i - 1])

        # Store new diagonal element (compressed)
        L[i - 1, i - 1] = r

        # Update remaining elements in row/column
        for j in (i + 1):n
            Lij = L[j, i]  # Lower triangular: j > i
            vj = v[j - 1]
            L[j - 1, i - 1] = c * Lij + s * vj  # Compressed indices
            v[j - 1] = -s' * Lij + c * vj
        end
    end

    # Step 2: Add new column
    @inbounds @views begin
        # Solve L[1:(d-1), 1:(d-1)] * x = ks[1:(d-1)] for new column
        ldiv!(L[d, 1:(d-1)], LowerTriangular(L[1:(d-1), 1:(d-1)]), ks[1:(d-1)])
        
        # Compute new diagonal element: L[d,d] = sqrt(ks[d] - ||x||²)
        v2 = sum(abs2, L[d, 1:(d-1)])
        L[d, d] = sqrt(abs(ks[d] - v2))
    end

    return F
end

# Benchmark this lower Cholesky version
update_times = Float64[]
for D in Ds
    println("Benchmarking D = $D")
    A = rand_psd(D + 1)
    A_new = A[2:end, end]
    A = copy(A[1:(end - 1), 1:(end - 1)])
    A_chol = Cholesky(cholesky(A).L.data, 'L', 0)
    x = rand(D)

    update_time = @benchmark windowed_lower_cholesky_update!($A_chol, $A_new) setup = begin
        A_chol = Cholesky(cholesky(copy($A)).L.data, 'L', 0)
    end evals = 1
    push!(update_times, median(update_time).time / 1e6)
end

# Add to plot
speedups = naive_times ./ update_times
plot!(Ds, speedups; marker=:star, label="Lower Triangular Update")
