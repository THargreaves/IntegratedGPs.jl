using CSV
using DataFrames
using Plots
using Random
using Statistics

rng = MersenneTwister(1234)
d = 20
# thinning_factor = 6
# thinning_factor = 15
thinning_factor = 30
σadd = 0.3
σlabel = 0.02  # label measurment
σϵ = sqrt(σadd^2 + σlabel^2)  # Observation noise std

# Comparison with mixture
# d = 50
# thinning_factor = 6
# σadd = 0.2

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

# Final version
# ν = 2.5
# ρ = 2.154434690031884
# σ² = 0.016681005372000592
# MSE:   0.040397841736590105
# Noise: 0.09039999999999998
# Skipping team 2 player 6 due to missing data
# Validation MSE:   0.043181508860661515
# Noise: 0.09039999999999998
# Progress: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:02
# Is interior: true
# Best Singer parameters:
# α = 145.63484775012444
# σm² = 0.30391953823131973
# σM² = 16.102620275609393
# Validation MSE (Singer):   0.05048830649480435
# Is interior: true
# Best CV parameters:
# σv² = 0.003562247890262444
# σV² = 0.25929437974046676
# Validation MSE (CV):   0.05100603945404705

path = joinpath(split("teamtrack/teamtrack/soccer_top/train/annotations", "/"))
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

# Normalise truth
p = 6
xss = all_xs_true_raw[p]
xss = (xss .- mean(xss)) ./ std(xss)

# Plot truth and observations
num_points = 60
p2 = plot(
    (0:(num_points * thinning_factor - 1)) / 30,
    xss[1:(num_points * thinning_factor)];
    xlabel="Time (s)",
    ylabel="Normalised Horizontal Position",
    label="Full Ground Truth",
    color=:blue,
    lw=2,
)
scatter!(
    p2,
    ts[1:num_points],
    all_xs_true[p][1:num_points];
    label="Thinned Ground Truth",
    color=:blue,
    ms=3,
    alpha=1,
)
scatter!(
    p2,
    ts[1:num_points],
    all_ys[p][1:num_points];
    label="Simulated Observations",
    color=:red,
    ms=3,
    alpha=0.5,
)
display(p2)
savefig(p2, "trajectory_sample.svg")

using IntegratedMaternGPs
using Distributions
using LinearAlgebra
using ProgressMeter

function compute_filtering_ll(ν, ρ, σ2, σϵ, τ, ys, d=30)
    # gp = integrate(CPEMaternGP(ν, ρ, σ2))
    gp = integrate(MaternGP(ν, ρ, σ2))
    # gp = integrate(RationalQuadraticGP(ν, ρ, σ2))
    K = length(ys)

    μs = Vector{Float64}(undef, K)
    σ2s = Vector{Float64}(undef, K)
    μs_pred = Vector{Float64}(undef, K)
    σ2s_pred = Vector{Float64}(undef, K)

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

    μs_pred[2] = μ[d - 1]
    σ2s_pred[2] = Σ[d - 1, d - 1]

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

        μs_pred[k] = μ[d - k + 1]
        σ2s_pred[k] = Σ[d - k + 1, d - k + 1]

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

        μs_pred[k] = μ[1]
        σ2s_pred[k] = Σ[1, 1]

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
        # Σ -= G * Σ[:, 1]'
        # Manually update in-place
        @inbounds for j in 1:d
            s = Σ[1, j]
            # Update jth column
            for i in 1:d
                @inbounds Σ[i, j] -= G[i] * s
            end
        end

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

    return ll, μs, σ2s, μs_pred, σ2s_pred
end

# Test with some parameters
ν = 1.5
ρ = 0.1
σ2 = 1.0

compute_filtering_ll(ν, ρ, σ2, σϵ, τ, all_ys[1], d)

# Grid search over parameters
# νs = range(0.5, 10.5; step=1.0)  # half-integers
νs = sort(unique(vcat(range(0.5, 6.0; length=10), range(0.5, 5.5; step=1.0))))
# νs = [Inf]
ρs = 10 .^ range(-1, 0.5; length=10)
σ2s = 10 .^ range(-4, 0; length=10)

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

# First verify that we can compute kernel values for all parameters
K_mat = zeros(d, d)
for i in 1:grid_size
    ν = νs[flatten_parameter_indices[i][1]]
    ρ = ρs[flatten_parameter_indices[i][2]]
    σ2 = σ2s[flatten_parameter_indices[i][3]]
    # test_gp = integrate(CPEMaternGP(ν, ρ, σ2))
    test_gp = integrate(MaternGP(ν, ρ, σ2))
    # test_gp = integrate(SquaredExponentialGP(ρ, σ2))
    global K_mat
    K_mat = zeros(d, d)
    for i in 1:d
        for j in 1:d
            K_mat[i, j] = kernel(test_gp, i * τ, j * τ)
        end
    end
    K_mat = Symmetric(K_mat)
    try
        cholesky(K_mat)
    catch e
        println("Failed for ν=$ν, ρ=$ρ, σ2=$σ2")
        rethrow(e)
    end
end

prog = Progress(grid_size)
Threads.@threads for i in 1:grid_size
    ν = νs[flatten_parameter_indices[i][1]]
    ρ = ρs[flatten_parameter_indices[i][2]]
    σ2 = σ2s[flatten_parameter_indices[i][3]]
    # grid_lls[flatten_parameter_indices[i]...] = compute_filtering_ll(
    #     ν, ρ, σ2, σϵ, τ, ys, 10
    # )[1]
    try
        ll = 0.0
        for ys in all_ys
            ll += compute_filtering_ll(ν, ρ, σ2, σϵ, τ, ys, d)[1]
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

# Is optimal value on the edge of the grid?
is_interior = all(ind.I .> 1) && all(ind.I .< (size(grid_lls)))
println("Is interior: $is_interior")

# Evaluate the best half-integer case for SSM fitting
println([ind for ind in flatten_parameter_indices])
println([isinteger(νs[ind[1]] - 0.5) ? ind : false for ind in flatten_parameter_indices],)
half_int_inds = filter(
    ind -> ind != false,
    [isinteger(νs[ind[1]] - 0.5) ? ind : false for ind in flatten_parameter_indices],
)
best_half_int_ll = -Inf
best_half_int_ind = 0
for ind in half_int_inds
    global best_half_int_ind, best_half_int_ll
    val = grid_lls[ind...]
    println("$ind gives $val")
    if val > best_half_int_ll
        best_half_int_ll = val
        best_half_int_ind = ind
    end
end

best_half_int_ν = νs[best_half_int_ind[1]]
best_half_int_ρ = ρs[best_half_int_ind[2]]
best_half_int_σ2 = σ2s[best_half_int_ind[3]]
println(
    "Best half-integer value: $best_half_int_ll at ($best_half_int_ν, $best_half_int_ρ, $best_half_int_σ2)",
)

model = Mixture{CPEMaternGP}([
    CPEMaternGP(best_half_int_ν, best_half_int_ρ, best_half_int_σ2)
])
ssm = cpe_mixture_to_ssm(model, 1.0)
println("Constructed SSM:\n$ssm")

# # Evaluate at best ν
# h1 = heatmap(
#     ρs,
#     σ2s,
#     grid_lls[ind[1], :, :]';
#     xscale=:identity,
#     yscale=:log10,
#     xlabel="ρ",
#     ylabel="σ²",
#     title="Log-Likelihood (at best ν = $best_ν)",
#     colorbar_title="LL",
#     size=(600, 500),
# )

# h2 = heatmap(
#     νs,
#     ρs,
#     grid_lls[:, :, ind[3]]';
#     xscale=:identity,
#     yscale=:log10,
#     xlabel="ν",
#     ylabel="ρ",
#     title="Log-Likelihood (maximised over σ²)",
#     colorbar_title="LL",
#     size=(600, 500),
# )

# h3 = heatmap(
#     νs,
#     σ2s,
#     grid_lls[:, ind[2], :]';
#     xscale=:identity,
#     yscale=:log10,
#     xlabel="ν",
#     ylabel="σ²",
#     title="Log-Likelihood (maximised over ρ)",
#     colorbar_title="LL",
#     size=(600, 500),
# )

# h_plot = plot(h1, h2, h3; layout=(3, 1), size=(600, 1500), left_margin=50Plots.px)

# # Plot marginals by fixing other two parameters
m1 = plot(
    νs,
    grid_lls[:, ind[2], ind[3]];
    xscale=:identity,
    xlabel="ν",
    ylabel="Maximum LL over ρ, σ²",
    label="",
    lw=2,
    ylims=(-8.835, -8.788),
)
# Add vertical line at best ν
vline!(m1, [best_ν]; linestyle=:dash, color=:black, label="ML Estimate", lw=2)
savefig(m1, "nu_profile.svg")

# m2 = plot(
#     ρs,
#     grid_lls[ind[1], :, ind[3]];
#     xscale=:identity,
#     xlabel="ρ",
#     ylabel="Max LL",
#     title="Max Log-Likelihood over ν, σ²",
#     size=(600, 400),
# )
# m3 = plot(
#     σ2s,
#     grid_lls[ind[1], ind[2], :];
#     xscale=:log10,
#     xlabel="σ²",
#     ylabel="Max LL",
#     title="Max Log-Likelihood over ν, ρ",
#     size=(600, 400),
# )
# m_plot = plot(m1, m2, m3; layout=(3, 1), size=(600, 1200), left_margin=50Plots.px)

# display(plot(h_plot, m_plot; layout=(1, 2), size=(1200, 1500)))

# Print optimal parameters
println("Best parameters:")
println("ν = $best_ν")
println("ρ = $best_ρ")
println("σ² = $best_σ2")

# Compute filtered states for best parameters
# _, μs, σ2s = compute_filtering_ll(best_ν, best_ρ, best_σ2, σϵ, τ, ys, 10)

# # Compute MSE
# mse = mean((μs .- xs_true) .^ 2)
# println("MSE:   $mse")

mses = Float64[]
for (i, xs_true) in enumerate(all_xs_true)
    ys = all_ys[i]
    _, μs, σ2s, _, _ = compute_filtering_ll(best_ν, best_ρ, best_σ2, σϵ, τ, ys, d)
    mse = mean((μs .- xs_true) .^ 2)
    push!(mses, mse)
end
mse = mean(mses)
println("MSE:   $mse")

# Compare to noise
println("Noise: $(σϵ^2)")

# One-step ahead predictions
one_step_mses = Float64[]
for (i, xs_true) in enumerate(all_xs_true)
    ys = all_ys[i]
    _, _, _, μs_pred, σ2s_pred = compute_filtering_ll(best_ν, best_ρ, best_σ2, σϵ, τ, ys, d)
    mse = mean((μs_pred[2:end] .- xs_true[2:end]) .^ 2)
    push!(one_step_mses, mse)
end
one_step_mse = mean(one_step_mses)
println("One-step ahead MSE:   $one_step_mse")

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
    _, μs_val, σ2s_val = compute_filtering_ll(best_ν, best_ρ, best_σ2, σϵ, τ, ys_val, d)
    mse_val = mean((μs_val .- xs_true_val) .^ 2)
    push!(mses_val, mse_val)
end
mse_val = mean(mses_val)
println("Validation MSE:   $mse_val")
println("Noise: $(σϵ^2)")

# One step ahead predictions
one_step_mses_val = Float64[]
for (i, xs_true_val) in enumerate(all_xs_true_val)
    ys_val = all_ys_val[i]
    _, _, _, μs_pred_val, σ2s_pred_val = compute_filtering_ll(
        best_ν, best_ρ, best_σ2, σϵ, τ, ys_val, d
    )
    mse_val = mean((μs_pred_val[2:end] .- xs_true_val[2:end]) .^ 2)
    push!(one_step_mses_val, mse_val)
end
one_step_mse_val = mean(one_step_mses_val)
println("Validation One-step ahead MSE:   $one_step_mse_val")

# Compare to Singer model
struct SingerModel{T}
    α::T
    σm2::T
    τ::T
    σM2::T  # initialisation std
end

using StaticArrays

function compute_A(model::SingerModel)
    α, σm2, τ = model.α, model.σm2, model.τ
    #! format: off
    return @SMatrix [
        1.0  τ    1 / α^2 * (-1 + α * τ + exp(-α * τ))
        0.0  1.0  1 / α * (1 - exp(-α * τ))
        0.0  0.0  exp(-α * τ)
    ]
    #! format: on
end

# \begin{aligned}
# q 11=\frac{1}{2 \alpha^5}\left[1-e^{-2 \alpha T}+2 \alpha T\right. & +\frac{2 \alpha^3 T^3}{3} \\
# & \left.-2 \alpha^2 T^2-4 \alpha T e^{-\alpha T}\right] \\
# q 12=\frac{1}{2 \alpha^4}\left[e^{-2 \alpha T}+1-2 e^{-\alpha T}\right. & \\
# & \left.+2 \alpha T e^{-\alpha T}-2 \alpha T+\alpha^2 T^2\right]
# \end{aligned}
# \begin{aligned}
# & q 13=\frac{1}{2 \alpha^3}\left[1-e^{-2 \alpha T}-2 \alpha T e^{-\alpha T}\right] \\
# & q 22=\frac{1}{2 \alpha^3}\left[4 e^{-\alpha T}-3-e^{-2 \alpha T}+2 \alpha T\right] \\
# & q 23=\frac{1}{2 \alpha^2}\left[e^{-2 \alpha T}+1-2 e^{-\alpha T}\right] \\
# & q 33=\frac{1}{2 \alpha}\left[1-e^{-2 \alpha T}\right] .
# \end{aligned}

function compute_Q(model::SingerModel)
    α, σm2, τ = model.α, model.σm2, model.τ
    q11 = (
        (1 / (2 * α^5)) * (
            1 - exp(-2 * α * τ) + 2 * α * τ + (2 * α^3 * τ^3) / 3 - 2 * α^2 * τ^2 -
            4 * α * τ * exp(-α * τ)
        )
    )
    q12 = (
        (1 / (2 * α^4)) * (
            exp(-2 * α * τ) + 1 - 2 * exp(-α * τ) + 2 * α * τ * exp(-α * τ) - 2 * α * τ +
            α^2 * τ^2
        )
    )
    q13 = (1 / (2 * α^3)) * (1 - exp(-2 * α * τ) - 2 * α * τ * exp(-α * τ))
    q22 = (1 / (2 * α^3)) * (4 * exp(-α * τ) - 3 - exp(-2 * α * τ) + 2 * α * τ)
    q23 = (1 / (2 * α^2)) * (exp(-2 * α * τ) + 1 - 2 * exp(-α * τ))
    q33 = (1 / (2 * α)) * (1 - exp(-2 * α * τ))
    Q = 2 * σm2 * α * @SMatrix [
        q11 q12 q13
        q12 q22 q23
        q13 q23 q33
    ]
    return Q
end

function compute_μ0(model::SingerModel, ys)
    x1 = ys[1]
    x2 = (ys[2] - ys[1]) / model.τ
    x3 = 0.0
    return @SVector [x1, x2, x3]
end

# \begin{aligned}
# & P 11(1 / 1)=\sigma_R^2 \\
# & P 12(1 / 1)=P 21(1 / 1)=\sigma_R^2 / T \\
# & P 13(1 / 1)=P 31(1 / 1)=0 \\
# & P 22(1 / 1)=2 \sigma_R^2 / T^2+\frac{\sigma_M^2}{\alpha^4 T^2}\left[2-\alpha^2 T^2+\frac{2 \alpha^3 T^3}{3}\right. \\
# & \left.\quad-2 e^{-\alpha T}-2 \alpha T e^{-\alpha T}\right] \\
# & P 23(1 / 1)=P 32(1 / 1)=\frac{\sigma_M^2}{\alpha^2 T}\left[e^{-\alpha T}+\alpha T-1\right]
# \end{aligned}

function compute_Σ0(model::SingerModel, σϵ)
    τ, σM2 = model.τ, model.σM2
    P11 = σϵ^2
    P12 = σϵ^2 / τ
    P13 = 0.0
    P22 = (
        (2 * σϵ^2) / τ^2 +
        (σM2 / (model.α^4 * τ^2)) * (
            2 - model.α^2 * τ^2 + (2 * model.α^3 * τ^3) / 3 - 2 * exp(-model.α * τ) -
            2 * model.α * τ * exp(-model.α * τ)
        )
    )
    P23 = (σM2 / (model.α^2 * τ)) * (exp(-model.α * τ) + model.α * τ - 1)
    P33 = σM2
    P = @SMatrix [
        P11 P12 P13
        P12 P22 P23
        P13 P23 P33
    ]
    return P
end

function compute_filtering_ll_singer(model::SingerModel, σϵ, ys)
    A = compute_A(model)
    Q = compute_Q(model)
    K = length(ys)

    μs = Vector{Float64}(undef, K)
    σ2s = Vector{Float64}(undef, K)

    μs_pred = Vector{Float64}(undef, K)
    σ2s_pred = Vector{Float64}(undef, K)

    # Initialize the state
    μ = compute_μ0(model, ys)
    Σ = compute_Σ0(model, σϵ)
    ll = 0.0

    μs[1] = μ[1]
    σ2s[1] = Σ[1, 1]

    # Prediction loop
    for k in 2:K
        # Predict forwards
        μ = A * μ
        Σ = A * Σ * A' + Q

        μs_pred[k] = μ[1]
        σ2s_pred[k] = Σ[1, 1]

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
        ll += logpdf(Normal(y_pred, sqrt(S)), ys[k])
    end

    return ll, μs, σ2s, μs_pred, σ2s_pred
end

# Test with some parameters
α = 0.5
σm2 = 0.1
σM2 = 1.0
singer_model = SingerModel(α, σm2, τ, σM2)
compute_filtering_ll_singer(singer_model, σϵ, all_ys[1])

# Grid search over parameters
αs = 10 .^ range(-2, 4; length=50)
σm2s = 10 .^ range(-3, 1; length=30)
σM2s = 10 .^ range(-1, 3; length=30)
grid_size_singer = length(αs) * length(σm2s) * length(σM2s)
grid_lls_singer = Array{Float64}(undef, length(αs), length(σm2s), length(σM2s))
flatten_parameter_indices_singer = Tuple{Int,Int,Int}[]
for i in 1:length(αs)
    for j in 1:length(σm2s)
        for k in 1:length(σM2s)
            push!(flatten_parameter_indices_singer, (i, j, k))
        end
    end
end
prog_singer = Progress(grid_size_singer)
Threads.@threads for i in 1:grid_size_singer
    α = αs[flatten_parameter_indices_singer[i][1]]
    σm2 = σm2s[flatten_parameter_indices_singer[i][2]]
    σM2 = σM2s[flatten_parameter_indices_singer[i][3]]
    singer_model = SingerModel(α, σm2, τ, σM2)
    ll = 0.0
    for ys in all_ys
        ll += compute_filtering_ll_singer(singer_model, σϵ, ys)[1]
    end
    # Normalise by series length and number of series
    ll /= K
    grid_lls_singer[flatten_parameter_indices_singer[i]...] = ll
    next!(prog_singer)
end

# Pick best parameters
ind_singer = argmax(grid_lls_singer)
best_α = αs[ind_singer[1]]
best_σm2 = σm2s[ind_singer[2]]
best_σM2 = σM2s[ind_singer[3]]
# Is optimal value on the edge of the grid?
is_interior_singer = all(ind_singer.I .> 1) && all(ind_singer.I .< (size(grid_lls_singer)))
println("Is interior: $is_interior_singer")

# Print optimal parameters
println("Best Singer parameters:")
println("α = $best_α")
println("σm² = $best_σm2")
println("σM² = $best_σM2")

# Compute filtered states for best parameters
singer_model = SingerModel(best_α, best_σm2, τ, best_σM2)

# Validation MSEs for singer
mses_val_singer = Float64[]
for (i, xs_true_val) in enumerate(all_xs_true_val)
    ys_val = all_ys_val[i]
    _, μs_val_singer, σ2s_val_singer, _, _ = compute_filtering_ll_singer(
        singer_model, σϵ, ys_val
    )
    mse_val_singer = mean((μs_val_singer .- xs_true_val) .^ 2)
    push!(mses_val_singer, mse_val_singer)
end
mse_val_singer = mean(mses_val_singer)
println("Validation MSE (Singer):   $mse_val_singer")

# One step ahead predictions for singer
one_step_mses_val_singer = Float64[]
for (i, xs_true_val) in enumerate(all_xs_true_val)
    ys_val = all_ys_val[i]
    _, _, _, μs_pred_val_singer, σ2s_pred_val_singer = compute_filtering_ll_singer(
        singer_model, σϵ, ys_val
    )
    mse_val_singer = mean((μs_pred_val_singer[2:end] .- xs_true_val[2:end]) .^ 2)
    push!(one_step_mses_val_singer, mse_val_singer)
end
one_step_mse_val_singer = mean(one_step_mses_val_singer)
println("Validation One-step ahead MSE (Singer):   $one_step_mse_val_singer")

# Finally compare to nearly constant velocity model
struct CVModel{T}
    σv2::T
    τ::T
    σV2::T  # initialisation std
end

function compute_A(model::CVModel)
    τ = model.τ
    return @SMatrix [
        1.0 τ
        0.0 1.0
    ]
end

function compute_Q(model::CVModel)
    σv2, τ = model.σv2, model.τ
    return σv2 * @SMatrix [
        (τ^3)/3 (τ^2)/2
        (τ^2)/2 τ
    ]
end

function compute_μ0(model::CVModel, ys)
    x1 = ys[1]
    x2 = (ys[2] - ys[1]) / model.τ
    return @SVector [x1, x2]
end

function compute_Σ0(model::CVModel, σϵ)
    τ, σV2 = model.τ, model.σV2
    P11 = σϵ^2
    P12 = σϵ^2 / τ
    P22 = (2 * σϵ^2) / τ^2 + (σV2 * τ) / 3
    P = @SMatrix [
        P11 P12
        P12 P22
    ]
    return P
end

function compute_filtering_ll_cv(model::CVModel, σϵ, ys)
    A = compute_A(model)
    Q = compute_Q(model)
    K = length(ys)

    μs = Vector{Float64}(undef, K)
    σ2s = Vector{Float64}(undef, K)
    μs_pred = Vector{Float64}(undef, K)
    σ2s_pred = Vector{Float64}(undef, K)

    # Initialize the state
    μ = compute_μ0(model, ys)
    Σ = compute_Σ0(model, σϵ)
    ll = 0.0

    μs[1] = μ[1]
    σ2s[1] = Σ[1, 1]

    # Prediction loop
    for k in 2:K
        # Predict forwards
        μ = A * μ
        Σ = A * Σ * A' + Q

        μs_pred[k] = μ[1]
        σ2s_pred[k] = Σ[1, 1]

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
        ll += logpdf(Normal(y_pred, sqrt(S)), ys[k])
    end

    return ll, μs, σ2s, μs_pred, σ2s_pred
end

# Test with some parameters
σv2 = 0.1
σV2 = 1.0
cv_model = CVModel(σv2, τ, σV2)
compute_filtering_ll_cv(cv_model, σϵ, all_ys[1])

# Grid search over parameters
σv2s = 10 .^ range(-3, 1; length=30)
σV2s = 10 .^ range(-1, 3; length=30)
grid_size_cv = length(σv2s) * length(σV2s)
grid_lls_cv = Array{Float64}(undef, length(σv2s), length(σV2s))
flatten_parameter_indices_cv = Tuple{Int,Int}[]
for i in 1:length(σv2s)
    for j in 1:length(σV2s)
        push!(flatten_parameter_indices_cv, (i, j))
    end
end
prog_cv = Progress(grid_size_cv)
Threads.@threads for i in 1:grid_size_cv
    σv2 = σv2s[flatten_parameter_indices_cv[i][1]]
    σV2 = σV2s[flatten_parameter_indices_cv[i][2]]
    cv_model = CVModel(σv2, τ, σV2)
    ll = 0.0
    for ys in all_ys
        ll += compute_filtering_ll_cv(cv_model, σϵ, ys)[1]
    end
    # Normalise by series length and number of series 
    ll /= K
    grid_lls_cv[flatten_parameter_indices_cv[i]...] = ll
    next!(prog_cv)
end
# Pick best parameters
ind_cv = argmax(grid_lls_cv)
best_σv2 = σv2s[ind_cv[1]]
best_σV2 = σV2s[ind_cv[2]]
# Is optimal value on the edge of the grid?
is_interior_cv = all(ind_cv.I .> 1) && all(ind_cv.I .< (size(grid_lls_cv) .- 1))
println("Is interior: $is_interior_cv")

# Print optimal parameters
println("Best CV parameters:")
println("σv² = $best_σv2")
println("σV² = $best_σV2")

# Compute filtered states for best parameters
cv_model = CVModel(best_σv2, τ, best_σV2)
# Validation MSEs for cv
mses_val_cv = Float64[]
for (i, xs_true_val) in enumerate(all_xs_true_val)
    ys_val = all_ys_val[i]
    _, μs_val_cv, σ2s_val_cv, _, _ = compute_filtering_ll_cv(cv_model, σϵ, ys_val)
    mse_val_cv = mean((μs_val_cv .- xs_true_val) .^ 2)
    push!(mses_val_cv, mse_val_cv)
end
mse_val_cv = mean(mses_val_cv)
println("Validation MSE (CV):   $mse_val_cv")

# One step ahead predictions for cv
one_step_mses_val_cv = Float64[]
for (i, xs_true_val) in enumerate(all_xs_true_val)
    ys_val = all_ys_val[i]
    _, _, _, μs_pred_val_cv, σ2s_pred_val_cv = compute_filtering_ll_cv(cv_model, σϵ, ys_val)
    mse_val_cv = mean((μs_pred_val_cv[2:end] .- xs_true_val[2:end]) .^ 2)
    push!(one_step_mses_val_cv, mse_val_cv)
end
one_step_mse_val_cv = mean(one_step_mses_val_cv)
println("Validation One-step ahead MSE (CV):   $one_step_mse_val_cv")
