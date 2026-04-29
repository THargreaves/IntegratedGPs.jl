using Printf

# include("download_datasets.jl")
include("05a_teamtrack_matern_poly.jl")
include("05b_teamtrack_se.jl")

results = [
    ("iMPG", maximum(SCRIPT_05a.grid_lls), SCRIPT_05a.mse_val),
    ("iSE", maximum(SCRIPT_05b.grid_lls), SCRIPT_05b.mse_val),
    ("Singer", maximum(SCRIPT_05a.grid_lls_singer), SCRIPT_05a.mse_val_singer),
    ("CV", maximum(SCRIPT_05a.grid_lls_cv), SCRIPT_05a.mse_val_cv),
]

println()
@printf("%-8s  %15s  %15s\n", "Model", "Training LL", "Validation MSE")
println(repeat('-', 42))
for (name, ll, mse) in results
    @printf("%-8s  %15.2f  %15.3f\n", name, ll, mse)
end

include("05c_video_overlay.jl")
include("06_kernel_eval_performance.jl")
