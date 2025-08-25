"""An implementation of the first Markovian approximation with prior for x0."""

using IntegratedMaternGPs

using Distributions
using KalmanFilters
using LinearAlgebra
using Plots
using Plots.PlotMeasures
using Random

# TODO: see items from 01_iGP-1_centered.jl

####################
#### PARAMETERS ####
####################

# Prior
x0_prior = Normal(5.0, 0.5)

# Matern GP
ν = 1.5
ρ = 0.8
σk = 1.2
gp = IntegratedGeneralMaternGP(ν, ρ, σk^2)

# Simulation
τ = 1.0
K = 200
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
    gp::IntegratedGeneralMaternGP,
    τ::Float64,
    K::Int,
    x0_prior::Normal,
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
    x0 = rand(rng, x0_prior)
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

xs_true, ys = simulate(rng, gp, τ, K, x0_prior, σϵ, d)

# Plot the trajectory and observations
all_ts = (0:(K - 1)) * τ
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

# Filtered states (without fixed lag smoothing)
μs = Vector{Float64}(undef, K)
σ2s = Vector{Float64}(undef, K)

# Initialize the state
μ = Vector{Float64}(undef, d + 1)
Σ = Matrix{Float64}(undef, d + 1, d + 1)
μ[d + 1] = x0_prior.μ
Σ[d + 1, d + 1] = x0_prior.σ^2

# Perform kalman update
y_pred = μ[d + 1]
y_err = ys[1] - y_pred
S = Σ[d + 1, d + 1] + σϵ^2
G = Σ[d + 1, d + 1] / S
μ[d + 1] += G * y_err
Σ[d + 1, d + 1] -= G * Σ[d + 1, d + 1]'

μs[1] = μ[d + 1]
σ2s[1] = Σ[d + 1, d + 1]

# Initial (non-prior) state
μ[1] = 0.0
Σ[1, 1] = kernel(gp, τ, τ)
y_pred = μ[1] + μ[d + 1]
y_err = ys[2] - y_pred
S = Σ[1, 1] + 2 * Σ[1, d + 1] + Σ[d + 1, d + 1] + σϵ^2
G = [Σ[1, 1]; Σ[d + 1, 1]] / S
μ_update = G * y_err
μ[1] += μ_update[1]
μ[d + 1] += μ_update[2]

HΣ = [Σ[1, 1]; Σ[d + 1, 1]] + [Σ[1, d + 1]; Σ[d + 1, d + 1]]
Σ_update = G * HΣ'

Σ[1, 1] -= Σ_update[1, 1]
Σ[1, d + 1] -= Σ_update[1, 2]
Σ[d + 1, 1] -= Σ_update[2, 1]
Σ[d + 1, d + 1] -= Σ_update[2, 2]

μs[2] = μ[1]
σ2s[2] = Σ[1, 1]

F = Cholesky(UpperTriangular(zeros(d, d)))
ks = Vector{Float64}(undef, d)
ts = Vector{Float64}(undef, d)

ts[1] = τ
F.U[1, 1] = sqrt(kernel(gp, τ, τ))
F_sub = Cholesky(UpperTriangular(@view F.U[1:1, 1:1]))

# No window shift for first d steps
for k in 2:d
    ts[k] = k * τ
    ks[1:k] .= kernel.(Ref(gp), (@view ts[1:k]), ts[k])

    w = F_sub.U' \ (@view ks[1:(k - 1)])
    f = F_sub.U \ w
    f_prior = 1 - sum(f)  # TODO: still not entirely confident on the derivatiion of this
    q2 = ks[k] - sum(abs2, w)

    f = f[(k - 1):-1:1]  # Reverse ordering to account for x having different ordering to k/t

    # Predict forwards
    μ[2:k] = μ[1:(k - 1)]
    μ[1] = dot(f, μ[2:k]) + f_prior * μ[d + 1]
    # God, these formulas need to be cleaned up/unit tested
    Sa = f' * Σ[1:(k - 1), 1:(k - 1)] * f + f_prior^2 * Σ[d + 1, d + 1] + q2
    Sb = Σ[1:(k - 1), 1:(k - 1)] * f + f_prior * Σ[1:(k - 1), d + 1]
    Sc = dot(f, Σ[1:(k - 1), d + 1]) + f_prior * Σ[d + 1, d + 1]
    Σ[2:k, 2:k] = Σ[1:(k - 1), 1:(k - 1)]
    Σ[2:k, d + 1] = Σ[1:(k - 1), d + 1]
    Σ[d + 1, 2:k] = Σ[2:k, d + 1]'
    Σ[1, 1] = Sa
    Σ[2:k, 1] = Sb
    Σ[1, 2:k] = Σ[2:k, 1]'
    Σ[1, d + 1] = Sc
    Σ[d + 1, 1] = Sc

    # Perform Kalman update
    y_pred = μ[1] + μ[d + 1]
    y_err = ys[k + 1] - y_pred
    S = Σ[1, 1] + 2 * Σ[1, d + 1] + Σ[d + 1, d + 1] + σϵ^2
    # TODO: this really should be rewritten to have x0 next to the first xk
    # Currently making fake matrix, skipping the undef elements to avoid making a mistake
    G = [Σ[1:k, 1]; Σ[d + 1, 1]] / S
    μ_update = G * y_err
    μ[1:k] += μ_update[1:k]
    μ[d + 1] += μ_update[k + 1]

    HΣ = [Σ[1:k, 1]; Σ[d + 1, 1]] + [Σ[1:k, d + 1]; Σ[d + 1, d + 1]]
    Σ_update = G .* HΣ'

    Σ[1:k, 1:k] -= Σ_update[1:k, 1:k]
    Σ[1:k, d + 1] -= Σ_update[1:k, k + 1]
    Σ[d + 1, 1:k] -= Σ_update[k + 1, 1:k]
    Σ[d + 1, d + 1] -= Σ_update[k + 1, k + 1]

    # Store the filtered state
    μs[k + 1] = μ[1]
    σ2s[k + 1] = Σ[1, 1]

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
    f_prior = 1 - sum(f)

    f = f[d:-1:1]  # Reverse ordering to account for x having different ordering to k/t

    q2 = k_new - sum(abs2, w)

    # Predict forwards
    m = dot(f, μ[1:d])
    μ[2:d] = μ[1:(d - 1)]
    μ[1] = m + f_prior * μ[d + 1]

    # TODO: this needs verifying against dense expression
    Sa = f' * Σ[1:d, 1:d] * f + f_prior^2 * Σ[d + 1, d + 1] + q2
    Sb = (
        Σ[1:(d - 1), 1:(d - 1)] * f[1:(d - 1)] +
        Σ[1:(d - 1), d] * f[d] +
        f_prior * Σ[1:(d - 1), d + 1]
    )
    Sc = dot(f, Σ[1:d, d + 1]) + f_prior * Σ[d + 1, d + 1]
    Σ[2:d, 2:d] = Σ[1:(d - 1), 1:(d - 1)]
    Σ[2:d, d + 1] = Σ[1:(d - 1), d + 1]
    Σ[d + 1, 2:d] = Σ[2:d, d + 1]'
    Σ[1, 1] = Sa
    Σ[2:d, 1] = Sb
    Σ[1, 2:d] = Σ[2:d, 1]'
    Σ[1, d + 1] = Sc
    Σ[d + 1, 1] = Sc

    # Perform Kalman update
    y_pred = μ[1] + μ[d + 1]
    y_err = ys[k + 1] - y_pred
    S = Σ[1, 1] + 2 * Σ[1, d + 1] + Σ[d + 1, d + 1] + σϵ^2
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

filtered_states = μs
filtered_states[2:end] .+= μ[d + 1]

plot!(
    p,
    all_ts,
    filtered_states;
    label="Filtered mean (±1σ)",
    color=:green,
    linewidth=2,
    ribbon=sqrt.(σ2s),
    fillalpha=0.2,
)
display(p)
