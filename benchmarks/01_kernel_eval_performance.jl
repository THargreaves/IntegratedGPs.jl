using IntegratedMaternGPs
using Polynomials
using Crayons
using ProgressMeter

highlight = Crayon(; foreground=:green)

function evals_per_second(f, timeout::Float64=1.0)
    cnt::Int = 0
    start_time = time()
    while (time() - start_time) < timeout
        f()
        cnt += 1
    end
    return cnt / timeout
end

S = 1.0
T = 2.0
bench(gp::GPT) where {GPT<:AbstractGPKernel} = evals_per_second(() -> kernel(gp, S, T))

SIGDIGITS = 3
function display(x::T) where {T<:AbstractFloat}
    x_exp10::Int = floor(log10(x))
    ref = x / (10.0^x_exp10)
    ref = round(ref; sigdigits=SIGDIGITS)
    digs = zeros(Int, SIGDIGITS)
    for i in 1:SIGDIGITS
        dig = floor(ref)
        digs[i] = Int(dig)
        ref = 10 * (ref - dig)
    end
    return "$(digs[1]).$(join(digs[2:end]))e$(x_exp10)"
end

println(highlight("\n\nSTARTING BENCHMARK TEST\n"))

pure_poly(x) = 1 + 2 * x + 3 * x * x
poly_poly = Polynomial([1, 2, 3])
pe_poly = PolynomialExp([1, 2, 3], 0)
cpe_poly = CompoundPolynomialExp(0 => [1, 2, 3])

evals_cpe = evals_per_second(() -> cpe_poly(1.0))
evals_poly = evals_per_second(() -> poly_poly(1.0))
evals_pe = evals_per_second(() -> pe_poly(1.0))
evals_pure = evals_per_second(() -> pure_poly(1.0))

println("Polynomials")
println("CPE:  $(round(evals_cpe, sigdigits=SIGDIGITS))/s")
println("PE:   $(round(evals_pe, sigdigits=SIGDIGITS))/s")
println("Poly: $(round(evals_poly, sigdigits=SIGDIGITS))/s")
println("Pure: $(round(evals_pure, sigdigits=SIGDIGITS))/s")
println("")

pexp_foo(x) = (1.0 + 2 * x + 3 * x * x) * exp(-x)
cpe_poly = PolynomialExp([1, 2, 3], 1)

evals_cpe = evals_per_second(() -> cpe_poly(1.0))
evals_foo = evals_per_second(() -> pexp_foo(1.0))

println("PolyExp")
println("PolyExp:  $(round(evals_cpe, sigdigits=SIGDIGITS))/s")
println("Foo:      $(round(evals_foo, sigdigits=SIGDIGITS))/s")
println("")

println(highlight("<> Evaluating kernels..."))

cpe_ν = 3.5
cpe_ρ = 2.4
cpe_σ2 = 5.4

rq_α = 5.7
rq_l = 2.3
rq_σ2 = 3.1

# TODO: include Squared Exponential

base_kernels = [
    ("General Matern", MaternGP(3.7, 2.4, 5.4)),
    ("General CPE Matern", MaternGP(cpe_ν, cpe_ρ, cpe_σ2)),
    ("Pure CPE Matern", CPEMaternGP(cpe_ν, cpe_ρ, cpe_σ2)),
    ("Rational Quadratic", RationalQuadraticGP(rq_α, rq_l, rq_σ2)),
]

kernel_types = Array{Type}(undef, (length(base_kernels), 3))
arr = zeros(Float64, (length(base_kernels), 3))

@showprogress for (ind, (name, gp)) in enumerate(base_kernels)
    kernel_types[ind, 1] = typeof(gp)
    arr[ind, 1] = bench(gp)

    naive_igp = Integrated(gp)
    kernel_types[ind, 2] = typeof(naive_igp)
    arr[ind, 2] = bench(naive_igp)

    igp = IntegratedMaternGPs.integrate(gp; cache_size=0)
    kernel_types[ind, 3] = typeof(igp)
    arr[ind, 3] = bench(igp)
end
longest_name = maximum([length(name) for (name, _) in base_kernels])

baseline = arr[1, 1]
println(highlight("</> Done.\n"))

LENGTH = longest_name + 2
SPACER = join([" " for _ in 1:4])
HLINE = join(["-" for _ in 1:50])

println("Types: ")
for ind in 1:length(base_kernels)
    println(
        kernel_types[ind, 1], SPACER, kernel_types[ind, 2], SPACER, kernel_types[ind, 3]
    )
end
println("\n\n")

println("Kernel evaluation benchmarks:")
println(HLINE)
println(
    rpad("BASE KERNEL", LENGTH),
    rpad("   K", 6),
    SPACER,
    rpad("Naive I", 6),
    SPACER,
    rpad("    I", 6),
)
println(HLINE)
for (ind, (name, _)) in enumerate(base_kernels)
    println(
        rpad(name * ": ", LENGTH),
        display(arr[ind, 1] / baseline),
        SPACER,
        display(arr[ind, 2] / baseline),
        SPACER,
        display(arr[ind, 3] / baseline),
    )
end
println(HLINE)

println("")
println(rpad("Baseline:", LENGTH), display(baseline))
println("")
