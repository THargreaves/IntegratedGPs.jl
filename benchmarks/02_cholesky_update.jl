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

function transposed_update(L::AbstractMatrix, ks::AbstractVector)
    U = L'
    IntegratedMaternGPs.my_lowrankupdate!(U)
    d = size(U, 1)

    @inbounds @views begin
        # Add the new time point
        ldiv!(U[1:(d - 1), d], UpperTriangular(U[1:(d - 1), 1:(d - 1)])', ks[1:(d - 1)])
        v2 = sum(abs2, U[1:(d - 1), d])
        U[d, d] = sqrt(ks[d] - v2)
    end

    return L
end

display(
    @benchmark transposed_update(L, $A_new) setup = begin
        A_chol = cholesky(copy($A))
        L = A_chol.L.data
    end evals = 1
)

# Ds = [10, 20, 50, 100, 200, 500, 1000]
Ds = 10:10:300
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
    # xscale=:log10,
    # yscale=:log2,
)
savefig("cholesky_update_speedup.pdf")

function windowed_lower_cholesky_update!(L::AbstractMatrix, ks::AbstractVector)
    """
    Update the Cholesky factorization of a covariance kernel matrix from one window shift.

    The current factorization is stored in `F` which is updated in-place without any
    allocations. `ks` stores entries corresponding to the new row of the covariance kernel
    matrix.

    For lower triangular factorization: A = L*L'
    """
    n = size(L, 1)

    # Step 1: Remove first column using Givens rotations
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

    d = n
    # Step 2: Add new column
    @inbounds @views begin
        # Solve L[1:(d-1), 1:(d-1)] * x = ks[1:(d-1)] for new column
        ldiv!(L[d, 1:(d - 1)], LowerTriangular(L[1:(d - 1), 1:(d - 1)]), ks[1:(d - 1)])

        # Compute new diagonal element: L[d,d] = sqrt(ks[d] - ||x||²)
        v2 = sum(abs2, L[d, 1:(d - 1)])
        L[d, d] = sqrt(abs(ks[d] - v2))
    end

    return L
end

display(
    @benchmark windowed_lower_cholesky_update!(L, $A_new) setup = begin
        A_chol = cholesky(copy($A))
        L = A_chol.L.data
    end evals = 1
)

# Benchmark this lower Cholesky version
# update_times = Float64[]
# for D in Ds
#     println("Benchmarking D = $D")
#     A = rand_psd(D + 1)
#     A_new = A[2:end, end]
#     A = copy(A[1:(end - 1), 1:(end - 1)])
#     A_chol = Cholesky(cholesky(A).L.data, 'L', 0)
#     x = rand(D)

#     update_time = @benchmark windowed_lower_cholesky_update!($A_chol, $A_new) setup = begin
#         A_chol = Cholesky(cholesky(copy($A)).L.data, 'L', 0)
#     end evals = 1
#     push!(update_times, median(update_time).time / 1e6)
# end

# # Add to plot
# speedups = naive_times ./ update_times
# plot!(Ds, speedups; marker=:star, label="Lower Triangular Update")

# Verify correctness
# A_test = rand_psd(8)
# A_new = A_test[2:end, end]
# A_test_chol = cholesky(copy(A_test[1:(end - 1), 1:(end - 1)]))
# L = A_test_chol.L.data
# A_new_chol = cholesky(copy(A_test[2:end, 2:end]))

# windowed_lower_cholesky_update!(L, A_new)

# tril(L) ≈ A_new_chol.L  # Should be true

# # Verify correctness of upper version
# A_test = rand_psd(8)
# A_new = A_test[2:end, end]
# A_test_chol = cholesky(copy(A_test[1:(end - 1), 1:(end - 1)]))
# A_new_chol = cholesky(copy(A_test[2:end, 2:end]))
# windowed_cholesky_update!(A_test_chol, A_new)
# A_test_chol.U ≈ A_new_chol.U  # Should be true
