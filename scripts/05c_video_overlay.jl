module SCRIPT_05b

using CSV
using DataFrames
using Plots
using Random
using Statistics

using VideoIO, Plots, Images

rng = MersenneTwister(1234)
thinning_factor = 30

path = joinpath(split("teamtrack/teamtrack/soccer_top/train/annotations", "/"))
prefix = "D_20220220_1"

segment_starts = 0:30:1230
segment_ends = segment_starts .+ 30

dfs = DataFrame[]
for (segment_start, segment_end) in zip(segment_starts, segment_ends)
    seg_start_str = lpad(string(segment_start), 4, '0')
    seg_end_str = lpad(string(segment_end), 4, '0')
    file_path = joinpath(path, "$(prefix)_$(seg_start_str)_$(seg_end_str).csv")
    println(file_path)
    push!(dfs, CSV.read(file_path, DataFrame; header=[1, 2, 3, 4]))
end
df = vcat(dfs...)

rename!(df, names(df)[1] => "Frame")
# Remove "Column##" suffix from column names
for col in names(df)[2:end]
    new_name = replace(col, r"_Column\d+" => "")
    rename!(df, col => new_name)
end

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

ball_x_col = "BALL_BALL_bb_left"
ball_y_col = "BALL_BALL_bb_top"
ball_w_col = "BALL_BALL_bb_width"
ball_h_col = "BALL_BALL_bb_height"

ball_x = df[:, ball_x_col]
ball_y = df[:, ball_y_col]
ball_w = df[:, ball_w_col]
ball_h = df[:, ball_h_col]

ball_cx = ball_x + 0.5 * ball_w
ball_cy = ball_y + 0.5 * ball_h

# Team 2 player 6

# Print proportion missing in each column
sort(mean.(eachcol(ismissing.(df))); rev=true)

# Average over all players who are not goalkeepers (only include those with no missing)
all_xs_true_raw = Vector{Float64}[]
all_ys_true_raw = Vector{Float64}[]

all_bb_lefts_true_raw = Vector{Float64}[]
all_bb_tops_true_raw = Vector{Float64}[]
all_bb_widths_true_raw = Vector{Float64}[]
all_bb_heights_true_raw = Vector{Float64}[]

for (i, (team, player)) in enumerate([
    (t, p) for t in 0:1 for p in 0:10 if (t != 0 || p != 0) && (t != 1 || p != 10)
])
    cx_col = "$(team)_$(player)_bb_cx"
    xs = df[:, cx_col]

    cy_col = "$(team)_$(player)_bb_cy"
    ys = df[:, cy_col]

    bb_l_col = "$(team)_$(player)_bb_left"
    bb_ls = df[:, bb_l_col]
    bb_t_col = "$(team)_$(player)_bb_top"
    bb_ts = df[:, bb_t_col]
    bb_w_col = "$(team)_$(player)_bb_width"
    bb_ws = df[:, bb_w_col]
    bb_h_col = "$(team)_$(player)_bb_height"
    bb_hs = df[:, bb_h_col]

    if any(ismissing.(xs))
        println("Skipping team $(team + 1) player $(player + 1) due to missing data")
        continue
    end

    push!(all_xs_true_raw, xs)
    push!(all_ys_true_raw, ys)

    push!(all_bb_lefts_true_raw, bb_ls)
    push!(all_bb_tops_true_raw, bb_ts)
    push!(all_bb_widths_true_raw, bb_ws)
    push!(all_bb_heights_true_raw, bb_hs)

    println(length(all_xs_true_raw), " team=$team, player=$player")
end

# USING TEAM 0, PLAYER 6

# Normalise truth
blue_player = 6
red_player = 17
xss = all_xs_true_raw[blue_player]
xss = (xss .- mean(xss)) ./ std(xss)

# Plot truth and observations
num_points = 60

# Open the video file
vid = VideoIO.open("teamtrack/teamtrack/soccer_top/train/videos/D_20220220_1_0030_0060.mp4")
f = VideoIO.openvideo(vid)

# Seek to a specific timestamp (e.g., 5.0 seconds) and read
seek(f, 30.0)
frame = read(f)

close(f)

# Display with Plots.jl
# VideoIO returns frames as RGB arrays — convert to a format Plots understands
img = RGB.(frame)  # ensure it's an RGB image
h, w = size(img)

flip(y) = h - y

img_flipped = img[end:-1:1, :]
p3 = plot(
    img_flipped;
    axis=false,
    border=:none,
    margins=0Plots.mm,
    framestyle=:none,
    size=(w, h),
    xflip=true,
)
plot!(
    all_xs_true_raw[blue_player][1:(num_points * thinning_factor)],
    flip.(all_ys_true_raw[blue_player][1:(num_points * thinning_factor)]);
    linecolor=:blue,
    linewidth=8,
    linestyle=:dot,
    label="",
    margins=0Plots.mm,
    framestyle=:none,
)
plot!(
    all_xs_true_raw[red_player][1:(num_points * thinning_factor)],
    flip.(all_ys_true_raw[red_player][1:(num_points * thinning_factor)]);
    linecolor=:red,
    linewidth=8,
    linestyle=:dot,
    label="",
    margins=0Plots.mm,
    framestyle=:none,
)

for p in 1:17
    bx = all_bb_lefts_true_raw[p][(num_points * thinning_factor)]
    by = all_bb_tops_true_raw[p][(num_points * thinning_factor)]
    bw = all_bb_widths_true_raw[p][(num_points * thinning_factor)]
    bh = all_bb_heights_true_raw[p][(num_points * thinning_factor)]
    plot!(
        [bx, bx + bw, bx + bw, bx, bx],
        [flip(by), flip(by), flip(by + bh), flip(by + bh), flip(by)];
        linecolor=(p <= 9 ? :blue : :red),
        linewidth=4,
        label=(p == 1 ? "Team 1" : (p == 10 ? "Team 2" : "")),
    )
end

bx = ball_x[num_points * thinning_factor]
by = ball_y[num_points * thinning_factor]
bw = ball_w[num_points * thinning_factor]
bh = ball_h[num_points * thinning_factor]

plot!(
    [bx, bx + bw, bx + bw, bx, bx],
    [flip(by), flip(by), flip(by + bh), flip(by + bh), flip(by)];
    linecolor=:black,
    linewidth=4,
    label="Ball",
    legendfontsize=36,
)

savefig(p3, "scripts/figs/sample_overlay.pdf")

end
