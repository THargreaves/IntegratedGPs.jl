"""
An implementation of the first Markovian approximation with no offset of the trajectory.

To avoid degeneracy in the GP posterior, we assume that the first observation time is
strictly greater than zero. This can be avoided by dropping these zero rows/columns in the
covariance matrix since knowledge of x1 does not help us predict further states in the
centered case; neither can we perform inference on a fixed state.
"""

using IntegratedMaternGPs

using Distributions
using KalmanFilters
using LinearAlgebra
using Plots
using Plots.PlotMeasures
using Random

# TODO: LRU cache is ollocating — wtf
# TODO: allocation in solves are a bit slow — rewrite in-place
# TODO: use static arrays if d is small enough (see 01a_SA_bench.jl)
# TODO: need to be careful as state and k/t vector have reverse ordering

####################
#### PARAMETERS ####
####################

# Matern GP
ν = 1.5
ρ = 0.8
σk = 1.2
gp = IntegratedMaternGP(ν, ρ, σk^2)

# Simulation
τ0 = 1.0  # HACK: using to avoid degeneracy
τ = 1.0
K = 100
σϵ = 5.0
SEED = 1234
rng = MersenneTwister(SEED)

# Filtering
d = 20

############################
#### FORWARD SIMULATION ####
############################

function simulate(
    rng::AbstractRNG,
    gp::IntegratedMaternGP,
    τ0::Float64,
    τ::Float64,
    K::Int,
    σϵ::Float64,
    d::Int,
)
    xs_true = Vector{Float64}(undef, K)
    ys = Vector{Float64}(undef, K)

    # Storage containers
    F = Cholesky(UpperTriangular(zeros(d, d)))
    ks = Vector{Float64}(undef, d)
    ts = Vector{Float64}(undef, d)

    # Initial state
    xs_true[1] = rand(rng, Normal(0.0, kernel(gp, τ0, τ0)))
    ys[1] = rand(rng, Normal(xs_true[1], σϵ))

    ts[1] = τ0
    F.U[1, 1] = sqrt(kernel(gp, τ0, τ0))
    F_sub = Cholesky(UpperTriangular(@view F.U[1:1, 1:1]))

    # No window shift for first d steps
    for k in 2:d
        ts[k] = τ0 + (k - 1) * τ
        ks[1:k] .= kernel.(Ref(gp), (@view ts[1:k]), ts[k])

        w = F_sub.U' \ (@view ks[1:(k - 1)])
        f = F_sub.U \ w
        q2 = ks[k] - sum(abs2, w)

        xs_true[k] = rand(rng, Normal(dot(f, @view xs_true[1:(k - 1)]), sqrt(q2)))
        ys[k] = rand(rng, Normal(xs_true[k], σϵ))

        F_sub = Cholesky(UpperTriangular(@view F.U[1:k, 1:k]))
        windowed_cholesky_add_last!(F_sub, @view ks[1:k])
    end

    # Shift window for remaining steps
    for k in (d + 1):K
        t_new = τ0 + (k - 1) * τ
        ks .= kernel.(Ref(gp), ts, t_new)
        k_new = kernel(gp, t_new, t_new)

        w = F.U' \ ks
        f = F.U \ w
        q2 = k_new - sum(abs2, w)

        xs_true[k] = rand(rng, Normal(dot(f, xs_true[(k - d):(k - 1)]), sqrt(q2)))
        ys[k] = rand(rng, Normal(xs_true[k], σϵ))

        # TODO: is there a faster way to do this?
        ks[1:(d - 1)] .= ks[2:d]
        ks[d] = k_new
        windowed_cholesky_update!(F, ks)
        ts[1:(d - 1)] .= ts[2:d]
        ts[d] = t_new
    end

    return xs_true, ys
end

xs_true, ys = simulate(rng, gp, τ0, τ, K, σϵ, d)

# Plot the trajectory and observations
all_ts = τ0 .+ (0:(K - 1)) * τ
p = plot(;
    size=(800, 600),
    title="\nMatern iGP (ν = $ν, ρ = $ρ, σk = $σk)",
    xlabel="Time",
    ylabel="State value",
    margin=20px,
    top_margin=30px,
)
plot!(p, all_ts, xs_true; label="True state", color=:blue, linewidth=2)
scatter!(p, all_ts, ys; label="Observations", color=:red, markersize=4, alpha=0.5)
display(p)

###################
#### FILTERING ####
###################

# We use z to represent the d-sized window of x values

# Start with k = d + 1, start from an arbitrary posterior for z
# μ = rand(rng, d + 1)
# Σ = rand(rng, d + 1, d + 1);
# Σ = Σ * Σ' + I;

# # Compute f and q2
# ts = τ0 .+ (0:(d - 1)) * τ
# K_mat = Matrix{Float64}(undef, d, d)
# for i in 1:d, j in 1:i
#     K_mat[i, j] = kernel(gp, ts[i], ts[j])
#     K_mat[j, i] = K_mat[i, j]  # Symmetric matrix
# end
# F = cholesky(K_mat)

# t_new = τ0 + d * τ
# ks = kernel.(Ref(gp), ts, t_new)
# k_new = kernel(gp, t_new, t_new)

# w = F.U' \ ks
# f = F.U \ w
# q2 = k_new - sum(abs2, w)

# # Perform predict step using sparse representation of F, Q
# # TODO: should use views to reduce allocations of slices
# μ_pred = Vector{Float64}(undef, d + 1)
# μ_pred[1] = dot(f, μ[1:d])
# μ_pred[2:d] = μ[1:(d - 1)]
# μ_pred[d + 1] = μ[d + 1]

# Σ_pred = Matrix{Float64}(undef, d + 1, d + 1)

# Σ_pred[1, 1] = f' * Σ[1:d, 1:d] * f + q2
# Σ_pred[2:d, 1] = Σ[1:(d - 1), 1:d] * f
# Σ_pred[1, 2:d] = Σ_pred[2:d, 1]'
# Σ_pred[2:d, 2:d] = Σ[1:(d - 1), 1:(d - 1)]

# # F * u
# c = Vector{Float64}(undef, d)
# u = @view Σ[1:d, d + 1]
# c[1] = dot(f, u)
# c[2:d] = u[1:(d - 1)]
# Σ_pred[1:d, d + 1] = c
# Σ_pred[d + 1, 1:d] = c'
# Σ_pred[d + 1, d + 1] = Σ[d + 1, d + 1]

# # Verify this matches
# F = zeros(d + 1, d + 1)
# F[1, 1:d] = f
# F[2:d, 1:(d - 1)] = Matrix{Float64}(I, d - 1, d - 1)
# F[d + 1, d + 1] = 1.0
# Q = zeros(d + 1, d + 1)
# Q[1, 1] = q2
# μ_pred_manual = F * μ
# Σ_pred_manual = F * Σ * F' + Q
# println("μ_pred == μ_pred_manual: ", all(μ_pred .≈ μ_pred_manual))
# println("Σ_pred == Σ_pred_manual: ", all(Σ_pred .≈ Σ_pred_manual))

# y_pred = μ_pred[1] + μ_pred[d + 1]
# y_err = ys[d + 1] - y_pred

# S = Σ_pred[1, 1] + 2 * Σ_pred[1, d + 1] + Σ_pred[d + 1, d + 1] + σϵ^2

# # Kalman gain
# G = Vector{Float64}(undef, d + 1)
# G .= (Σ_pred[:, 1] .+ Σ_pred[:, d + 1]) ./ S

# μ_filt = μ_pred + G * y_err
# Σ_filt = Σ_pred - G .* (Σ_pred[:, 1] .+ Σ_pred[:, d + 1])'

# # Verify this matches
# H = zeros(1, d + 1)
# H[1, 1] = 1.0
# H[1, d + 1] = 1.0

# y_err = [ys[d + 1]] - H * μ_pred
# S = H * Σ_pred * H' + [σϵ^2;;]
# G_manual = Σ_pred * H' / S
# μ_filt_manual = μ_pred + G_manual * y_err
# Σ_filt_manual = Σ_pred - G_manual * H * Σ_pred

# println("μ_filt == μ_filt_manual: ", all(μ_filt .≈ μ_filt_manual))
# println("Σ_filt == Σ_filt_manual: ", all(Σ_filt .≈ Σ_filt_manual))

## Combine into generic filtering pass
# Window will grow up until time d, then it will start to shift.

# Filtered states (without fixed lag smoothing)
μs = Vector{Float64}(undef, K)
σ2s = Vector{Float64}(undef, K)

# Stored fixed lag smoothed states too
μs_fixed = Vector{Float64}(undef, K)
σ2s_fixed = Vector{Float64}(undef, K)

# Initialize the state
μ = Vector{Float64}(undef, d)
Σ = Matrix{Float64}(undef, d, d)
μ[1] = 0.0
Σ[1, 1] = kernel(gp, τ0, τ0)

# Perform kalman update
y_pred = μ[1]
y_err = ys[1] - y_pred
S = Σ[1, 1] + σϵ^2
G = Σ[1, 1] / S
μ[1] += G * y_err
Σ[1, 1] -= G * Σ[1, 1]'

F = Cholesky(UpperTriangular(zeros(d, d)))
ks = Vector{Float64}(undef, d)
ts = Vector{Float64}(undef, d)

ts[1] = τ0
F.U[1, 1] = sqrt(kernel(gp, τ0, τ0))
F_sub = Cholesky(UpperTriangular(@view F.U[1:1, 1:1]))

# No window shift for first d steps
for k in 2:d
    ts[k] = τ0 + (k - 1) * τ
    ks[1:k] .= kernel.(Ref(gp), (@view ts[1:k]), ts[k])

    w = F_sub.U' \ (@view ks[1:(k - 1)])
    f = F_sub.U \ w
    q2 = ks[k] - sum(abs2, w)

    f = f[(k - 1):-1:1]  # Reverse ordering to account for x having different ordering to k/t
    A = ExpandingTransition(f)

    # Predict forwards
    μ = A * (@view μ[1:(k - 1)])
    Σ = quadratic_form(A, Symmetric(@view Σ[1:(k - 1), 1:(k - 1)]))
    Σ[1, 1] += q2

    # Perform Kalman update
    y_pred = μ[1]
    y_err = ys[k] - y_pred
    S = Σ[1, 1] + σϵ^2
    G = Σ[1:k, 1] / S
    μ[1:k] += G * y_err
    Σ[1:k, 1:k] -= G * Σ[1:k, 1]'

    # Store the filtered state
    μs[k] = μ[1]
    σ2s[k] = Σ[1, 1]

    F_sub = Cholesky(UpperTriangular(@view F.U[1:k, 1:k]))
    windowed_cholesky_add_last!(F_sub, @view ks[1:k])
end

# Shift window for remaining steps
for k in (d + 1):K
    # Stored fixed lag smoothed states
    μs_fixed[k - d] = μ[1]
    σ2s_fixed[k - d] = Σ[1, 1]

    t_new = τ0 + (k - 1) * τ
    ks .= kernel.(Ref(gp), ts, t_new)
    k_new = kernel(gp, t_new, t_new)

    w = F.U' \ ks
    f = F.U \ w
    q2 = k_new - sum(abs2, w)

    f = f[d:-1:1]  # Reverse ordering to account for x having different ordering to k/t
    A = ShiftingTransition(f)

    # Predict forwards
    μ = A * μ
    Σ = quadratic_form(A, Symmetric(Σ))
    Σ[1, 1] += q2

    # Perform Kalman update
    y_pred = μ[1]
    y_err = ys[k] - y_pred
    S = Σ[1, 1] + σϵ^2
    G = Σ[:, 1] / S
    μ += G * y_err
    Σ -= G * Σ[:, 1]'

    # Store the filtered state
    μs[k] = μ[1]
    σ2s[k] = Σ[1, 1]

    # Update the window
    ks[1:(d - 1)] .= ks[2:d]
    ks[d] = k_new
    windowed_cholesky_update!(F, ks)
    ts[1:(d - 1)] .= ts[2:d]
    ts[d] = t_new
end

plot!(
    p,
    all_ts,
    μs;
    label="Filtered mean (±1σ)",
    color=:green,
    linewidth=2,
    ribbon=sqrt.(σ2s),
    fillalpha=0.2,
)
display(p)

# Plot fixed lag smoothed states
# plot!(
#     p,
#     all_ts[1:(K - d)],
#     μs_fixed[1:(K - d)];
#     label="Fixed lag smoothed mean (±1σ)",
#     color=:orange,
#     linewidth=2,
#     ribbon=sqrt.(σ2s_fixed[1:(K - d)]),
#     fillalpha=0.2,
# )
# display(p)
