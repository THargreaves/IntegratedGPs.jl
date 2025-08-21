using LinearAlgebra
using StaticArrays
using Plots
using Statistics
using ProgressMeter

Ds = 1:37

# TODO: real time includes compilation time (really slow for large D) but benchmark doesn't.
# Not sure how to benchmark with compilation time included or how to interpret this.

array_times = Vector{Float64}(undef, length(Ds))
@showprogress desc = "Benchmarking Arrays" for (i, d) in enumerate(Ds)
    X = rand(d, d)
    U = UpperTriangular(X)
    y = rand(d)

    res = @benchmark $U \ $y
    array_times[i] = median(res.times)
end

sarray_times = Vector{Float64}(undef, length(Ds))
@showprogress desc = "Benchmarking SArrays" for (i, D) in enumerate(Ds)
    X = @SMatrix rand(D, D)
    U = UpperTriangular(X)
    y = @SVector rand(D)

    res = @benchmark $U \ $y
    sarray_times[i] = median(res.times)
end

ratios = array_times ./ sarray_times

plot(
    Ds,
    ratios;
    ylabel="Relative Speedup",
    xlabel="Matrix Size (D)",
    yscale=:log10,
    ylims=(0.1, 100),
)
