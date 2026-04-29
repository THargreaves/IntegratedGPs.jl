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
x0_prior = Normal(0.0, 0.0001)

# Matern GP
ν = 0.5
ρ = 1.0
σk = 0.4
gp = integrate(MaternGP(ν, ρ, σk^2))

# Simulation
τ = 0.1
K = 100
σϵ = 3.0
SEED = 1235
rng = MersenneTwister(SEED)

# Filtering
d = 30

############################
#### FORWARD SIMULATION ####
############################

function simulate(
    rng::AbstractRNG,
    gp::AbstractIntegratedMaternGP,
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
p = plot(; xlabel="Time", ylabel="Process value", size=(600, 300))
plot!(p, all_ts, xs_true; label="Integrated Matérn GP (ν = 0.5)", lw=2)
# scatter!(p, all_ts, ys; label="Observations", color=:red, markersize=4, alpha=0.5)
display(p)

gp = integrate(CPEMaternGP(5.5, ρ, σk^2))

# Filtering
d = 20

rng = MersenneTwister(SEED)

############################
#### FORWARD SIMULATION ####
############################

function simulate(
    rng::AbstractRNG, gp, τ::Float64, K::Int, x0_prior::Normal, σϵ::Float64, d::Int
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
plot!(p, all_ts, xs_true; label="Squared Exponential Kernel", lw=2)
# scatter!(p, all_ts, ys; label="Observations", color=:red, markersize=4, alpha=0.5)
display(p)

savefig(p, "example_trajectories.pdf")
