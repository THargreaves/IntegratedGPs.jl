module SCRIPT_04

using CSV
using DataFrames
using Plots
using Random

rng = MersenneTwister(1234)

path = "teamtrack/teamtrack/soccer_top/train/annotations"
prefix = "D_20220220_1"

data_set = 0
segment_starts = 0:30:270
segment_ends = segment_starts .+ 30

# seg_start_str = lpad(string(segment_start), 4, '0')
# seg_end_str = lpad(string(segment_end), 4, '0')

# file_path = joinpath(path, "$(prefix)_$(seg_start_str)_$(seg_end_str).csv")
# df = CSV.read(file_path, DataFrame; header=[1, 2, 3, 4])

# # Collect next 30 seconds and append to df
# seg_start_str = lpad(string(segment_start + 30), 4, '0')
# seg_end_str = lpad(string(segment_end + 30), 4, '0')
# file_path = joinpath(path, "$(prefix)_$(seg_start_str)_$(seg_end_str).csv")
# df2 = CSV.read(file_path, DataFrame; header=[1, 2, 3, 4])
# df = vcat(df, df2)

dfs = DataFrame[]
for (segment_start, segment_end) in zip(segment_starts, segment_ends)
    seg_start_str = lpad(string(segment_start), 4, '0')
    seg_end_str = lpad(string(segment_end), 4, '0')
    file_path = joinpath(path, "$(prefix)_$(seg_start_str)_$(seg_end_str).csv")
    push!(dfs, CSV.read(file_path, DataFrame; header=[1, 2, 3, 4]))
end
df = vcat(dfs...)

rename!(df, names(df)[1] => "Frame")
# Remove "Column##" suffix from column names
for col in names(df)[2:end]
    new_name = replace(col, r"_Column\d+" => "")
    rename!(df, col => new_name)
end

# Plot first two columns (x, y)
frames = 1:size(df, 1)
plot(frames, df[:, 3]; title="Teamtrack Trajectory", xlabel="X", ylabel="Y")

# Compute box centres
for team in 0:1
    for player in 0:10
        # bb_left, bb_top, bb_width, bb_height
        # Of form 0_0_bb_left etc.
        x_col = "$(team)_$(player)_bb_left"
        y_col = "$(team)_$(player)_bb_top"
        w_col = "$(team)_$(player)_bb_width"
        h_col = "$(team)_$(player)_bb_height"
        cx_col = "$(team)_$(player)_bb_cx"
        cy_col = "$(team)_$(player)_bb_cy"
        df[!, cx_col] = df[!, x_col] .+ df[!, w_col] ./ 2
        df[!, cy_col] = df[!, y_col] .+ df[!, h_col] ./ 2
    end
end

# Plot all players
colours = distinguishable_colors(11)
p = plot(;
    title="Teamtrack Trajectories",
    xlabel="X",
    ylabel="Y",
    size=(800, 600),
    legend=:outerright,
    aspect_ratio=1,
)
for team in 0:1
    for player in 0:10
        cx_col = "$(team)_$(player)_bb_cx"
        cy_col = "$(team)_$(player)_bb_cy"
        plot!(
            p,
            df[:, cx_col],
            df[:, cy_col];
            label="Team $(team + 1) Player $(player + 1)",
            color=colours[player + 1],
            lw=2,
            linestyle=team == 0 ? :dot : :dash,
        )
    end
end
display(p)

# Team 2 player 6

# Create ground truth
xs_true = df[:, "1_5_bb_cx"]

# Interpolate missing values
function interpolate_missing(ys)
    n = length(ys)
    for i in 1:n
        if isnan(ys[i])
            # Find next non-missing value
            j = i + 1
            while j <= n && isnan(ys[j])
                j += 1
            end
            if j <= n
                ys[i] = ys[j]
            else
                ys[i] = ys[i - 1]  # Use previous value if at end
            end
        end
    end
    return ys
end

# Thin to one observation every 0.5 seconds (15 frames)
xs_true = xs_true[1:15:end]
K = length(xs_true)
ts = collect(range(0; step=0.5, length=K))

# Normalise to zero mean, unit variance
μ_x = mean(xs_true)
σ_x = std(xs_true)
xs_true = (xs_true .- μ_x) ./ σ_x

σϵ = 0.2  # Observation noise std
ys = xs_true .+ σϵ * randn(rng, K)  # Noisy observations

# Plot truth and observations
p2 = plot(
    ts,
    xs_true;
    title="Team 2 Player 6 X Position",
    xlabel="Frame",
    ylabel="X Position",
    color=:blue,
    lw=2,
    size=(800, 400),
)
scatter!(p2, ts, ys; label="Observations", color=:red, ms=4, alpha=0.5)
display(p2)

using IntegratedGPs
using Distributions
using LinearAlgebra
using ProgressMeter

function compute_filtering_ll(ν, ρ, σ2, σϵ, τ, ys, d=30)
    gp = IntegratedMaternGP(ν, ρ, σ2)
    K = length(ys)

    μs = Vector{Float64}(undef, K)
    σ2s = Vector{Float64}(undef, K)

    # Initialize the state
    μ = Vector{Float64}(undef, d)
    Σ = Matrix{Float64}(undef, d, d)
    ll = 0.0

    # Use uniformative prior for initial state
    μ[d] = ys[1]
    Σ[d, d] = σϵ^2

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
    ll += logpdf(Normal(y_pred, sqrt(S)), ys[2])

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
            Symmetric(Σ[(d - m - 1):d, (d - m - 1):d]),
            A,
            Symmetric(Σ[(d - m):d, (d - m):d]),
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
        # println(k)
        # println(round.(ks, digits=2))
        # println("f: ", any(isnan.(f)), " q2: ", isnan(q2))
        μs[k] = μ[i]
        σ2s[k] = Σ[i, i]
        ll += logpdf(Normal(y_pred, sqrt(S)), ys[k + 1])

        ks[k - 1] = k_new
        F_sub = Cholesky(UpperTriangular(@view F.U[1:(m + 1), 1:(m + 1)]))
        windowed_cholesky_add_last!(F_sub, @view ks[1:(m + 1)])
        ts[k - 1] = t_new
    end

    # Times are now all relative to the shifting prior time (i.e. constant)
    # t_new = d * τ

    # For that reason, w and f are fixed
    t_new = d * τ
    ks .= kernel.(Ref(gp), ts, t_new)
    k_new = kernel(gp, t_new, t_new)
    w = F.U' \ ks
    f = F.U \ w
    f_prior = 1 - sum(f)
    q2 = k_new - sum(abs2, w)
    f = f[(d - 1):-1:1]  # Reverse ordering to account for x having different ordering to k/t

    # Shift window for remaining steps
    for k in (d + 1):K
        ks .= kernel.(Ref(gp), ts, t_new)
        k_new = kernel(gp, t_new, t_new)

        # w = F.U' \ ks
        # f = F.U \ w
        # f_prior = 1 - sum(f)
        # q2 = k_new - sum(abs2, w)

        # f = f[(d - 1):-1:1]  # Reverse ordering to account for x having different ordering to k/t

        A = ShiftingMarkovianTransition(f, f_prior)

        # Predict forwards
        mul!(μ, A, μ)
        quadratic_form!(Symmetric(Σ), A, Symmetric(Σ))
        Σ[1, 1] += q2

        # println(round.(ks, digits=2))
        # println(k_new)
        # println()
        # println(q2)
        # println(Σ[1, 1])

        # Perform Kalman update
        y_pred = μ[1]
        y_err = ys[k] - y_pred
        S = Σ[1, 1] + σϵ^2
        G = Σ[:, 1] / S
        μ += G * y_err
        Σ -= G * Σ[:, 1]'

        # Store the filtered state
        # println(S)
        # println(σϵ^2)
        μs[k] = μ[1]
        σ2s[k] = Σ[1, 1]
        ll += logpdf(Normal(y_pred, sqrt(S)), ys[k])

        # Update the window
        # ks[1:(d - 2)] .= ks[2:(d - 1)]
        # ks[d - 1] = k_new
        # windowed_cholesky_update!(F, ks)
    end

    return ll, μs, σ2s
end

# Test with some parameters
ν = 1.5
ρ = 1.0
σ2 = 1.0

compute_filtering_ll(ν, ρ, σ2, σϵ, 0.5, ys, 30)

# Grid search over parameters
νs = range(0.5, 3.5; length=30)
ρs = 10 .^ range(-3, 0; length=30)
σ2s = 10 .^ range(-3, 0; length=30)

grid_size = length(νs) * length(ρs) * length(σ2s)

grid_lls = Array{Float64}(undef, length(νs), length(ρs), length(σ2s))
flatten_parameter_indices = Tuple{Int,Int,Int}[]
for i in 1:length(νs)
    for j in 1:length(ρs)
        for k in 1:length(σ2s)
            push!(flatten_parameter_indices, (i, j, k))
        end
    end
end

# There is a discontinuity in struvel
# ν = 0.8333333333333334
# ρ = 0.2782559402207124
# sqrt(2 * test_gp.ν) * 0.5 * 15  passes
# sqrt(2 * test_gp.ν) * 0.5 * 16  gives Inf
# sqrt(2 * test_gp.ν) * 0.5 * 17  passes
prog = Progress(grid_size)
Threads.@threads for i in 1:grid_size
    ν = νs[flatten_parameter_indices[i][1]]
    ρ = ρs[flatten_parameter_indices[i][2]]
    σ2 = σ2s[flatten_parameter_indices[i][3]]
    grid_lls[flatten_parameter_indices[i]...] = compute_filtering_ll(
        ν, ρ, σ2, σϵ, 0.5, ys, 10
    )[1]
    next!(prog)
end

h1 = heatmap(
    ρs,
    σ2s,
    dropdims(mean(grid_lls; dims=1); dims=1);
    xscale=:identity,
    yscale=:log10,
    xlabel="ρ",
    ylabel="σ²",
    title="Log-Likelihood (ν averaged)",
    colorbar_title="LL",
    size=(600, 500),
)

h2 = heatmap(
    νs,
    ρs,
    dropdims(mean(grid_lls; dims=3); dims=3);
    xscale=:identity,
    yscale=:log10,
    xlabel="ν",
    ylabel="ρ",
    title="Log-Likelihood (σ² averaged)",
    colorbar_title="LL",
    size=(600, 500),
)

h3 = heatmap(
    νs,
    σ2s,
    dropdims(mean(grid_lls; dims=2); dims=2);
    xscale=:identity,
    yscale=:log10,
    xlabel="ν",
    ylabel="σ²",
    title="Log-Likelihood (ρ averaged)",
    colorbar_title="LL",
    size=(600, 500),
)

display(plot(h1, h2, h3; layout=(3, 1), size=(600, 1500)))

# Pick best parameters
ind = argmax(grid_lls)
best_ν = νs[ind[1]]
best_ρ = ρs[ind[2]]
best_σ2 = σ2s[ind[3]]

# Compute filtered states for best parameters
_, μs, σ2s = compute_filtering_ll(best_ν, best_ρ, best_σ2, σϵ, 0.5, ys, 10)

# Compute MSE
mse = mean((μs .- xs_true) .^ 2)
println("MSE:   $mse")

# Compare to noise
println("Noise: $(σϵ^2)")

# Test on new data
xs_true_val = df[:, "1_6_bb_cx"]
xs_true_val = xs_true_val[1:15:end]
K_val = length(xs_true_val)
xs_true_val = (xs_true_val .- μ_x) ./ σ_x
ys_val = xs_true_val .+ σϵ * randn(rng, K_val)
_, μs_val, σ2s_val = compute_filtering_ll(best_ν, best_ρ, best_σ2, σϵ, 0.5, ys_val, 10)
mse_val = mean((μs_val .- xs_true_val) .^ 2)
println("Validation MSE:   $mse_val")
println("Noise: $(σϵ^2)")

end
