using Struve
using Bessels
using ArbNumerics
using Plots

# ν = 3.5
# z = 400.0
# prec = 2048
# println("Large arg: ", Struve.struvem_large_arg_cutoff(ν, z))

# L_pkg = struvel(ν, z)

# L_big = Struve.struvel_power_series(BigFloat(ν, prec), BigFloat(z, prec))

# println("Pkg Error: ", round(Float64(log10(abs(L_pkg - L_big))), digits=1))

# # Now compute besselk-struvel product
# # KL_big = besselk(BigFloat(ν, prec), BigFloat(z, prec)) * L_big

# using ArbNumerics
# setworkingprecision(ArbReal; bits=prec)
# besselk_arb = ArbNumerics.besselk(ArbReal(ν), ArbReal(z))

# KL_arb = besselk_arb * ArbReal(L_big)
# KL_pkg = Bessels.besselk(ν, z) * L_pkg

# println()
# println("KL Pkg Error: ", round(Float64(log10(abs(KL_arb - KL_pkg))), digits=1))

# # Use L(x) = M(x) + I(x)
# KL_stable = (
#     Struve.struvem_large_argument(ν, z) * Bessels.besselk(ν, z) +
#     besselkx(ν, z) * besselix(ν, z)
# )

# println("KL Stable Error: ", round(Float64(log10(abs(KL_stable - KL_arb))), digits=1))

# # What is the error in Struve.struvem_large_argument(ν, z)?
# M_large = Struve.struvem_large_argument(ν, z)
# M_arb = ArbReal(L_big) - ArbNumerics.besseli(ArbReal(ν), ArbReal(z))
# println("M Large Arg Error: ", round(Float64(log10(abs(M_arb - M_large))), digits=1))

# # Try using log_struvem_large_argument — they're the same thing once large enough
# M_log = Struve.log_struvem_large_argument(ν, z)

# Compare errors across a range of values
ν = 3.5
zs = 10 .^ range(0; stop=4.0, length=100)
prec = 512

setworkingprecision(ArbReal; bits=prec)

function bessel_struvel_big(ν, z)
    L_big = Struve.struvel_power_series(BigFloat(ν - 1, prec), BigFloat(z, prec))
    K_arb = ArbNumerics.besselk(ArbReal(ν), ArbReal(z))
    return z * K_arb * ArbReal(L_big)
end

big_xs = [bessel_struvel_big(ν, z) for z in zs]

function bessel_struvel_pkg_product(ν, z)
    try
        L_pkg = Struve.struvel(ν - 1, z)
        K_pkg = Bessels.besselk(ν, z)
        return z * K_pkg * L_pkg
    catch e
        return NaN
    end
end

pkg_xs = [bessel_struvel_pkg_product(ν, z) for z in zs]

pkg_log_errors = [
    isnan(pkg_xs[i]) ? NaN : Float64(log10(abs(big_xs[i] - pkg_xs[i]))) for
    i in 1:length(zs)
]

function bessel_struvel_stable(ν, z)
    try
        M_large = Struve.struvem_large_argument(ν - 1, z)
        # Sign correction
        M_large = -M_large
        return z * (M_large * Bessels.besselk(ν, z) + besselkx(ν, z) * besselix(ν - 1, z))
    catch e
        return NaN
    end
end

stable_xs = [bessel_struvel_stable(ν, z) for z in zs]
stable_log_errors = [
    isnan(stable_xs[i]) ? NaN : Float64(log10(abs(big_xs[i] - stable_xs[i]))) for
    i in 1:length(zs)
]

large_arg = [Struve.struvem_large_arg_cutoff(ν, z) for z in zs]
large_arg_z = zs[findfirst(large_arg)]

p = plot(;
    xscale=:log10,
    xlabel="z",
    ylabel="log10 error",
    title="Error in Bessel-Struve product vs ArbNumerics",
    xlims=(minimum(zs), maximum(zs)),
)
plot!(zs, pkg_log_errors; label="Pkg", lw=2)
plot!(zs, stable_log_errors; label="Stable", lw=2, ls=:dash)
vline!([large_arg_z]; label="", lw=2, ls=:dash, color=:black)
display(p)

# Plot xs on log scale
p = plot(;
    xscale=:log10,
    yscale=:log10,
    xlabel="z",
    ylabel="value",
    title="Bessel-Struve product",
    xlims=(minimum(zs), maximum(zs)),
)
plot!(zs, Float64.(big_xs); label="BigFloat", lw=2)
plot!(zs, Float64.(pkg_xs); label="Pkg", lw=2, ls=:dash)
plot!(
    zs[stable_xs .> 0],
    Float64.(stable_xs[stable_xs .> 0]);
    label="Stable form",
    lw=2,
    ls=:dot,
)
vline!([large_arg_z]; label="", lw=2, ls=:dash, color=:black)
display(p)

# # Plot the logarithms of struvem, struvel, besselk and besseli using big floats
# log_struvems = [Float64(log10(abs(Struve.struvem_large_argument(ν, z)))) for z in zs]
# log_struvels = [Float64(log10(abs(Struve.struvel_power_series(ν, z)))) for z in zs]
# log_besselk = [Float64(log10(abs(ArbNumerics.besselk(ArbReal(ν), ArbReal(z))))) for z in zs]
# log_besseli = [Float64(log10(abs(ArbNumerics.besseli(ArbReal(ν), ArbReal(z))))) for z in zs]

# # Plot these
# p2 = plot(;
#     xscale=:log10,
#     xlabel="z",
#     ylabel="log10 value",
#     title="Logarithms of Struve and Bessel functions",
#     xlims=(minimum(zs), maximum(zs)),
# )
# plot!(zs, log_struvems; label="log10 Struve M", lw=2)
# plot!(zs, log_struvels; label="log10 Struve L", lw=2)
# plot!(zs, log_besselk; label="log10 Bessel K", lw=2)
# plot!(zs, log_besseli; label="log10 Bessel I", lw=2, ls=:dash)

# # Add bessel-struve product
# plot!(zs, log.(big_xs); label="log10 Bessel K * Struve L", lw=2, ls=:dash)

# Plot struvem on normal scale using package and big float
struvem_pkgs = [Struve.struvem(ν, z) for z in zs]

function struvem_big(ν, z)
    L_big = Struve.struvel_power_series(BigFloat(ν, prec), BigFloat(z, prec))
    I_arb = ArbNumerics.besseli(ArbReal(ν), ArbReal(z))
    return ArbReal(L_big) - I_arb
end
struvem_bigs = [struvem_big(ν, z) for z in zs]

p3 = plot(;
    xscale=:log10,
    xlabel="z",
    ylabel="Struve M",
    title="Struve M function values",
    xlims=(minimum(zs), maximum(zs)),
)
plot!(zs, struvem_pkgs; label="Struve M (pkg)", lw=2)
# plot!(zs, Float64.(struvem_bigs); label="Struve M (big)", lw=2, ls=:dash)

# [Struve.struvel_power_series(BigFloat(ν, prec), BigFloat(z, prec)) for z in zs]
# [ArbNumerics.besseli(ArbReal(ν), ArbReal(z)) for z in zs]

# Seems to be off by a minus sign
struve_m_assymptoptic(ν, z) = 2^*(1 - ν) / (sqrt(π) * Bessels.gamma(ν + 0.5)) * z^(ν - 1)
plot!(zs, struve_m_assymptoptic.(ν, zs); label="Struve M assymptotic", lw=2, ls=:dash)
