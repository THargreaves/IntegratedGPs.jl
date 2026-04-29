using CSV
using DataFrames
using Plots
using Random

rng = MersenneTwister(1234)
d = 10
thinning_factor = 6
σadd = 0.2
σlabel = 0.02  # label measurment
σϵ = sqrt(σadd^2 + σlabel^2)  # Observation noise std

# Good params:
# d = 10
# thinning_factor = 5
# σadd = 0.2
# σlabel = 0.0  # label measurment
# σϵ = sqrt(σadd^2 + σlabel^2)  # Observation noise std
# Best parameters:
# ν = 0.5
# ρ = 1.0
# σ² = 0.021544346900318832
# MSE:   0.008506540839509363
# Noise: 0.04000000000000001
# Validation MSE:   0.008806008163874932
# Noise: 0.04000000000000001

path = "teamtrack/teamtrack/soccer_top/train/annotations"
prefix = "D_20220220_1"

data_set = 0
segment_starts = 0:30:1230
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
        if (team == 0 && player == 0) || (team == 1 && player == 10)
            continue  # Skip goalkeepers
        end
        # if !(team == 0 && player == 5)
        #     continue  # Only plot team 2 player 6
        # end
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
# xs_true = df[:, "1_5_bb_cx"]
# xs_true = df[:, "1_6_bb_cx"]

# Print proportion missing in each column
sort(mean.(eachcol(ismissing.(df))); rev=true)

# Plot times of missing data
p_missing = plot(;
    title="Missing Data Times",
    xlabel="Frame",
    ylabel="Player",
    size=(800, 600),
    legend=false,
)
for team in 0:1
    for player in 0:10
        if (team == 0 && player == 0) || (team == 1 && player == 10)
            continue  # Skip goalkeepers
        end
        # if !(team == 0 && player == 5)
        #     continue  # Only plot team 2 player 6
        # end
        cx_col = "$(team)_$(player)_bb_cx"
        cy_col = "$(team)_$(player)_bb_cy"
        missing_frames = findall(ismissing.(df[:, cx_col]))
        y_vals = fill(player + 1 + team * 11, length(missing_frames))
        scatter!(p_missing, missing_frames, y_vals; ms=2, color=colours[player + 1])
    end
end
display(p_missing)

# Average over all players who are not goalkeepers (only include those with no missing)
all_xs_true_raw = Vector{Float64}[]
for (i, (team, player)) in enumerate([
    (t, p) for t in 0:1 for p in 0:10 if (t != 0 || p != 0) && (t != 1 || p != 10)
])
    cx_col = "$(team)_$(player)_bb_cx"
    xs = df[:, cx_col]
    if any(ismissing.(xs))
        println("Skipping team $(team + 1) player $(player + 1) due to missing data")
        continue
    end
    push!(all_xs_true_raw, xs)
end

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

# Thin to one observation every 2 seconds (60 frames)
# xs_true = xs_true[1:15:end]
all_xs_true = [xs_true[1:thinning_factor:end] for xs_true in all_xs_true_raw]
K = length(all_xs_true[1])
τ = thinning_factor * 1 / 30
ts = collect(range(0; step=τ, length=K))

# Normalise to zero mean, unit variance
# μ_x = mean(xs_true)
# σ_x = std(xs_true)
# xs_true = (xs_true .- μ_x) ./ σ_x
for i in 1:length(all_xs_true)
    μ_x = mean(all_xs_true[i])
    σ_x = std(all_xs_true[i])
    all_xs_true[i] = (all_xs_true[i] .- μ_x) ./ σ_x
end

# ys = xs_true .+ σϵ * randn(rng, K)  # Noisy observations
all_ys = [xs_true .+ σadd * randn(rng, K) for xs_true in all_xs_true]

# Plot truth and observations
p2 = plot(
    ts,
    all_xs_true[1];
    title="Team 2 Player 6 X Position",
    xlabel="Frame",
    ylabel="X Position",
    color=:blue,
    lw=2,
    size=(800, 400),
)
scatter!(p2, ts, all_ys[1]; label="Observations", color=:red, ms=4, alpha=0.5)
display(p2)

using IntegratedGPs
using Distributions
using LinearAlgebra
using ProgressMeter

function compute_filtering_ll(ν, ρ, σ2, α, σϵ, τ, ys, d=30)
    base_gp = CPEMaternGP(ν, ρ, σ2)
    gp = DynamicsInformedCPEMaternGP(base_gp, -α)
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
        # f_rem = 1 - sum(f)
        f_rem = exp(α * t_new) * (1 - sum(f .* exp.(α * ((@view ts[1:m]) .- t_new))))
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

        # Manually compute Cholesky to avoid numerical issues
        # K_mat = zeros(m + 1, m + 1)
        # for i in 1:(m + 1)
        #     for j in 1:(m + 1)
        #         K_mat[i, j] = kernel(gp, i * τ, j * τ)
        #     end
        # end
        # K_mat = Symmetric(K_mat)
        # try
        # F_sub = cholesky(K_mat; check=true)
        # catch e
        #     println("Dimension: $m")
        #     println(K_mat)
        #     rethrow(e)
        # end

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
    # f_prior = 1 - sum(f)
    f_prior = exp(α * t_new) * (1 - sum(f .* exp.(α * (ts .- t_new))))
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
α = 0.5
ν = 1.5
ρ = 0.1
σ2 = 1.0

compute_filtering_ll(ν, ρ, σ2, α, σϵ, τ, all_ys[1], d)

# Grid search over parameters
νs = range(0.5, 10.5; step=1.0)  # half-integers
ρs = 10 .^ range(-1, 0.0; length=10)
σ2s = 10 .^ range(-4, 0; length=10)
αs = 10 .^ range(-1, 0; length=5)

grid_size = length(νs) * length(ρs) * length(σ2s) * length(αs)

grid_lls = Array{Float64}(undef, length(νs), length(ρs), length(σ2s), length(αs))
flatten_parameter_indices = Tuple{Int,Int,Int,Int}[]
for i in 1:length(νs)
    for j in 1:length(ρs)
        for k in 1:length(σ2s)
            for l in 1:length(αs)
                push!(flatten_parameter_indices, (i, j, k, l))
            end
        end
    end
end

# First verify that we can compute kernel values for all parameters
# K_mat = zeros(d, d)
# for i in 1:grid_size
#     ν = νs[flatten_parameter_indices[i][1]]
#     ρ = ρs[flatten_parameter_indices[i][2]]
#     σ2 = σ2s[flatten_parameter_indices[i][3]]
#     α = αs[flatten_parameter_indices[i][4]]
#     # test_gp = integrate(CPEMaternGP(ν, ρ, σ2))
#     test_gp = integrate(SquaredExponentialGP(ρ, σ2))
#     global K_mat
#     K_mat = zeros(d, d)
#     for i in 1:d
#         for j in 1:d
#             K_mat[i, j] = kernel(test_gp, i * τ, j * τ)
#         end
#     end
#     K_mat = Symmetric(K_mat)
#     try
#         cholesky(K_mat)
#     catch e
#         println("Failed for ν=$ν, ρ=$ρ, σ2=$σ2")
#         rethrow(e)
#     end
# end

prog = Progress(grid_size)
Threads.@threads for i in 1:grid_size
    ν = νs[flatten_parameter_indices[i][1]]
    ρ = ρs[flatten_parameter_indices[i][2]]
    σ2 = σ2s[flatten_parameter_indices[i][3]]
    α = αs[flatten_parameter_indices[i][4]]
    try
        ll = 0.0
        for ys in all_ys
            ll += compute_filtering_ll(ν, ρ, σ2, α, σϵ, τ, ys, d)[1]
        end
        # Normalise by series length and number of series
        ll /= K
        grid_lls[flatten_parameter_indices[i]...] = ll
    catch
        grid_lls[flatten_parameter_indices[i]...] = -Inf
    end
    next!(prog)
end

# Pick best parameters
ind = argmax(grid_lls)
best_ν = νs[ind[1]]
best_ρ = ρs[ind[2]]
best_σ2 = σ2s[ind[3]]
best_α = αs[ind[4]]

# Is optimal value on the edge of the grid?
is_interior = all(ind.I .> 1) && all(ind.I .< (size(grid_lls) .- 1))
println("Is interior: $is_interior")

# Print optimal parameters
println("Best parameters:")
println("ν = $best_ν")
println("ρ = $best_ρ")
println("σ² = $best_σ2")
println("α = $best_α")

mses = Float64[]
for (i, xs_true) in enumerate(all_xs_true)
    ys = all_ys[i]
    _, μs, σ2s = compute_filtering_ll(best_ν, best_ρ, best_σ2, best_α, σϵ, τ, ys, d)
    mse = mean((μs .- xs_true) .^ 2)
    push!(mses, mse)
end
mse = mean(mses)
println("MSE:   $mse")

# Compare to noise
println("Noise: $(σϵ^2)")

#### VALIDATION ON NEW DATA ####

path = "teamtrack/teamtrack/soccer_top/val/annotations"
prefix = "D_20220220_1"

data_set = 0
segment_starts = 1260:30:1500
segment_ends = segment_starts .+ 30

dfs = DataFrame[]
for (segment_start, segment_end) in zip(segment_starts, segment_ends)
    seg_start_str = lpad(string(segment_start), 4, '0')
    seg_end_str = lpad(string(segment_end), 4, '0')
    file_path = joinpath(path, "$(prefix)_$(seg_start_str)_$(seg_end_str).csv")
    push!(dfs, CSV.read(file_path, DataFrame; header=[1, 2, 3, 4]))
end
df_val = vcat(dfs...)

rename!(df_val, names(df_val)[1] => "Frame")
# Remove "Column##" suffix from column names
for col in names(df_val)[2:end]
    new_name = replace(col, r"_Column\d+" => "")
    rename!(df_val, col => new_name)
end

# Compute box centres
for team in 0:1
    for player in 0:10
        if (team == 0 && player == 0) || (team == 1 && player == 10)
            continue  # Skip goalkeepers
        end
        x_col = "$(team)_$(player)_bb_left"
        y_col = "$(team)_$(player)_bb_top"
        w_col = "$(team)_$(player)_bb_width"
        h_col = "$(team)_$(player)_bb_height"
        cx_col = "$(team)_$(player)_bb_cx"
        cy_col = "$(team)_$(player)_bb_cy"
        df_val[!, cx_col] = df_val[!, x_col] .+ df_val[!, w_col] ./ 2
        df_val[!, cy_col] = df_val[!, y_col] .+ df_val[!, h_col] ./ 2
    end
end

# Average over all players who are not goalkeepers (only include those with no missing)
all_xs_true_val_raw = Vector{Float64}[]
for (i, (team, player)) in enumerate([
    (t, p) for t in 0:1 for p in 0:10 if (t != 0 || p != 0) && (t != 1 || p != 10)
])
    cx_col = "$(team)_$(player)_bb_cx"
    xs = df_val[:, cx_col]
    if any(ismissing.(xs))
        println("Skipping team $(team + 1) player $(player + 1) due to missing data")
        continue
    end
    push!(all_xs_true_val_raw, xs)
end

# Thin to one observation every 2 seconds (60 frames)
all_xs_true_val = [xs_true[1:thinning_factor:end] for xs_true in all_xs_true_val_raw]
K_val = length(all_xs_true_val[1])
ts_val = collect(range(0; step=τ, length=K_val))

# Normalise to zero mean, unit variance using training data stats
for i in 1:length(all_xs_true_val)
    μ_x = mean(all_xs_true_val[i])
    σ_x = std(all_xs_true_val[i])
    all_xs_true_val[i] = (all_xs_true_val[i] .- μ_x) ./ σ_x
end

all_ys_val = [xs_true .+ σadd * randn(rng, K_val) for xs_true in all_xs_true_val]
mses_val = Float64[]
for (i, xs_true_val) in enumerate(all_xs_true_val)
    ys_val = all_ys_val[i]
    _, μs_val, σ2s_val = compute_filtering_ll(
        best_ν, best_ρ, best_σ2, best_α, σϵ, τ, ys_val, d
    )
    mse_val = mean((μs_val .- xs_true_val) .^ 2)
    push!(mses_val, mse_val)
end
mse_val = mean(mses_val)
println("Validation MSE:   $mse_val")
println("Noise: $(σϵ^2)")
