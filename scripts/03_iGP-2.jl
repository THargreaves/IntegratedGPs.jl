"""An implementation of the first Markovian approximation with prior for x0."""

using IntegratedMaternGPs

using Distributions
using KalmanFilters
using LinearAlgebra
using Plots
using Plots.PlotMeasures
using Random

####################
#### PARAMETERS ####
####################

# Prior
x1_prior = Normal(5.0, 0.5)

# Matern GP
ν = 1.5
ρ = 0.8
σk = 1.2
gp = IntegratedMaternGP(ν, ρ, σk^2)

# Simulation
τ = 1.0
K = 200
thin = 5
σϵ = 3.0
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
    τ::Float64,
    K::Int,
    x1_prior::Normal,
    σϵ::Float64,
    d::Int,
)
    xs_true = Vector{Float64}(undef, K)
    ys = Vector{Float64}(undef, K)

    # Storage containers
    F = Cholesky(UpperTriangular(zeros(d, d)))
    ks = Vector{Float64}(undef, d)
    ts = Vector{Float64}(undef, d)

    # Sample without x0 and then add it later as an offset
    x0 = rand(rng, x1_prior)
    xs_true[1] = 0.0

    # Initial (non-prior) state
    xs_true[2] = rand(rng, Normal(0.0, kernel(gp, τ, τ)))

    ts[1] = τ
    F.U[1, 1] = sqrt(kernel(gp, τ, τ))
    F_sub = Cholesky(UpperTriangular(@view F.U[1:1, 1:1]))

    # No window shift for first d steps (plus prior)
    for k in 2:d
        ts[k] = k * τ
        ks[1:k] .= kernel.(Ref(gp), (@view ts[1:k]), ts[k])

        w = F_sub.U' \ (@view ks[1:(k - 1)])
        f = F_sub.U \ w
        q2 = ks[k] - sum(abs2, w)

        xs_true[k + 1] = rand(rng, Normal(dot(f, @view xs_true[2:k]), sqrt(q2)))

        F_sub = Cholesky(UpperTriangular(@view F.U[1:k, 1:k]))
        windowed_cholesky_add_last!(F_sub, @view ks[1:k])
    end

    # Shift window for remaining steps
    for k in (d + 1):(K - 1)
        t_new = k * τ
        ks .= kernel.(Ref(gp), ts, t_new)
        k_new = kernel(gp, t_new, t_new)

        w = F.U' \ ks
        f = F.U \ w
        q2 = k_new - sum(abs2, w)

        xs_true[k + 1] = rand(rng, Normal(dot(f, xs_true[(k - d + 1):k]), sqrt(q2)))

        # TODO: is there a faster way to do this?
        ks[1:(d - 1)] .= ks[2:d]
        ks[d] = k_new
        windowed_cholesky_update!(F, ks)
        ts[1:(d - 1)] .= ts[2:d]
        ts[d] = t_new
    end

    # Add prior
    xs_true .+= x0

    # Sample observations
    for k in 1:K
        ys[k] = rand(rng, Normal(xs_true[k], σϵ))
    end

    return xs_true, ys
end

# Simulate at higher resolution
xs_true, ys = simulate(rng, gp, τ / thin, K * thin, x1_prior, σϵ, d * thin);
# Thin measurements
ys = ys[1:thin:end]

# Plot the trajectory and observations
all_ts = (0:(K * thin - 1)) * (τ / thin)
thinned_ts = (0:(K - 1)) * τ
p = plot(;
    size=(800, 600),
    title="\nMatern iGP (ν = $ν, ρ = $ρ, σk = $σk)",
    xlabel="Time",
    ylabel="State value",
    margin=20px,
    top_margin=30px,
)
plot!(p, all_ts, xs_true; label="True state", color=:blue, linewidth=2)
scatter!(p, thinned_ts, ys; label="Observations", color=:red, markersize=4, alpha=0.5)
display(p)

###################
#### FILTERING ####
###################

# Filtered states (without fixed lag smoothing)
μs = Vector{Float64}(undef, K)
σ2s = Vector{Float64}(undef, K)

# Initialize the state
μ = Vector{Float64}(undef, d)
Σ = Matrix{Float64}(undef, d, d)
μ[d] = x1_prior.μ
Σ[d, d] = x1_prior.σ^2

# Perform kalman update
y_pred = μ[d]
y_err = ys[1] - y_pred
S = Σ[d, d] + σϵ^2
G = Σ[d, d] / S
μ[d] += G * y_err
Σ[d, d] -= G * Σ[d, d]'

μs[1] = μ[d]
σ2s[1] = Σ[d, d]

# Initial (non-prior) state
μ[d - 1] = μ[d]
Σ[d - 1, d - 1] = kernel(gp, τ, τ)
# TODO: could set these to zero initially
Σ[d - 1, d] = 0.0
Σ[d, d - 1] = 0.0
y_pred = μ[d - 1]
y_err = ys[2] - y_pred
S = Σ[d - 1, d - 1] + σϵ^2
G = Σ[(d - 1):d, d - 1] / S
μ[(d - 1):d] += G * y_err
Σ[(d - 1):d, (d - 1):d] -= G * Σ[(d - 1):d, d - 1]'

μs[2] = μ[d - 1]
σ2s[2] = Σ[d - 1, d - 1]

# Window has d - 1 previous states plus the prior state
F = Cholesky(UpperTriangular(zeros(d - 1, d - 1)))
ks = Vector{Float64}(undef, d - 1)
ts = Vector{Float64}(undef, d - 1)

ts[1] = τ
F.U[1, 1] = sqrt(kernel(gp, τ, τ))
F_sub = Cholesky(UpperTriangular(@view F.U[1:1, 1:1]))

# No window shift for first d steps
for k in 3:d
    m = k - 2  # Size of the window (excluding prior)

    t_new = (k - 1) * τ
    ks[1:m] .= kernel.(Ref(gp), (@view ts[1:m]), t_new)
    k_new = kernel(gp, t_new, t_new)

    w = F_sub.U' \ (@view ks[1:m])
    f = F_sub.U \ w
    f_rem = 1 - sum(f)
    q2 = k_new - sum(abs2, w)

    f = f[m:-1:1]  # Reverse ordering to account for x having different ordering to k/t

    A = ExpandingMarkovianTransition(f, f_rem)

    # Predict forwards
    @views mul!(μ[(d - m - 1):d], A, μ[(d - m):d])
    @views quadratic_form!(
        Symmetric(Σ[(d - m - 1):d, (d - m - 1):d]), A, Symmetric(Σ[(d - m):d, (d - m):d])
    )
    Σ[d - k + 1, d - k + 1] += q2

    # Perform Kalman update
    i = d - k + 1
    y_pred = μ[i]
    y_err = ys[k + 1] - y_pred
    S = Σ[i, i] + σϵ^2
    G = Σ[i:d, i] / S
    μ[i:d] += G * y_err
    Σ[i:d, i:d] -= G * Σ[i:d, i]'

    # Store the filtered state
    μs[k] = μ[i]
    σ2s[k] = Σ[i, i]

    ks[k - 1] = k_new
    F_sub = Cholesky(UpperTriangular(@view F.U[1:(m + 1), 1:(m + 1)]))
    windowed_cholesky_add_last!(F_sub, @view ks[1:(m + 1)])
    ts[k - 1] = t_new
end

# Shift window for remaining steps
for k in (d + 1):K
    t_new = k * τ
    ks .= kernel.(Ref(gp), ts, t_new)
    k_new = kernel(gp, t_new, t_new)

    w = F.U' \ ks
    f = F.U \ w
    f_prior = 1 - sum(f)
    q2 = k_new - sum(abs2, w)

    f = f[(d - 1):-1:1]  # Reverse ordering to account for x having different ordering to k/t

    A = ShiftingMarkovianTransition(f, f_prior)

    # Predict forwards
    mul!(μ, A, μ)
    quadratic_form!(Symmetric(Σ), A, Symmetric(Σ))
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
    ks[1:(d - 2)] .= ks[2:(d - 1)]
    ks[d - 1] = k_new
    windowed_cholesky_update!(F, ks)
    ts[1:(d - 2)] .= ts[2:(d - 1)]
    ts[d - 1] = t_new
end

plot!(
    p,
    thinned_ts,
    μs;
    label="Filtered mean (±1σ)",
    color=:green,
    linewidth=2,
    ribbon=sqrt.(σ2s),
    fillalpha=0.2,
)
display(p)
