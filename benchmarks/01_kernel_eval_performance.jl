using IntegratedMaternGPs
using Polynomials
using Crayons
using ProgressMeter
using BenchmarkTools

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
MAX_SAMPLES = 100_000
MAX_SECONDS = 10

function bench_gp(gp::GPT) where {GPT<:AbstractGPKernel}
    trial = @benchmark kernel(_gp, S, T) setup = (_gp = $gp) samples = MAX_SAMPLES seconds =
        MAX_SECONDS
    return median(trial).time
end#evals_per_second(() -> kernel(gp, S, T))

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
    return "$(digs[1]).$(join(digs[2:end]))e$(x_exp10>=0 ? "+" : "")$(x_exp10)"
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

cpe_ν, cpe_ρ, cpe_σ2 = 3.5, 2.4, 5.4
se_ℓ, se_σ2 = 3.2, 0.4
rq_α, rq_l, rq_σ2 = 5.7, 2.3, 3.1

# TODO: Use a 4th order model
ssm_A = [0.9;;]
ssm_Q = [1.0;;]
ssm_H = [1.0;;]
ssm = SSM(ssm_A, ssm_Q, ssm_H)

base_kernels = [
    ("General Matern", MaternGP(3.7, 2.4, 5.4)),
    ("General CPE Matern", MaternGP(cpe_ν, cpe_ρ, cpe_σ2)),
    ("Pure CPE Matern", CPEMaternGP(cpe_ν, cpe_ρ, cpe_σ2)),
    ("SSM GP", SSMGP(ssm)),
    ("CPE SSM GP", ssm2GPKernel(ssm)),
    ("Squared Exponential", SquaredExponentialGP(se_ℓ, se_σ2)),
    ("Rational Quadratic", RationalQuadraticGP(rq_α, rq_l, rq_σ2)),
]

kernel_types = Array{Type}(undef, (length(base_kernels), 3))
arr = zeros(Float64, (length(base_kernels), 3))

prog = Progress(length(kernel_types))
for (ind, (name, gp)) in enumerate(base_kernels)
    kernel_types[ind, 1] = typeof(gp)
    arr[ind, 1] = bench_gp(gp)
    next!(prog)

    naive_igp = Integrated(gp)
    kernel_types[ind, 2] = typeof(naive_igp)
    arr[ind, 2] = bench_gp(naive_igp)
    next!(prog)

    if gp isa SSMGP
        arr[ind, 3] = 1E12
        next!(prog)
        continue
    end
    igp = IntegratedMaternGPs.integrate(gp; cache_size=0)
    kernel_types[ind, 3] = typeof(igp)
    arr[ind, 3] = bench_gp(igp)
    next!(prog)
end
longest_name = maximum([length(name) for (name, _) in base_kernels])

baseline = arr[1, 1]
println(highlight("</> Done.\n"))

LENGTH = longest_name + 2
SPACER = join([" " for _ in 1:4])
HLINE = join(["-" for _ in 1:50])

println("Types: ")
for ind in 1:length(base_kernels)
    if base_kernels[ind][2] isa SSMGP
        println(kernel_types[ind, 1], SPACER, kernel_types[ind, 2])
    else
        println(
            kernel_types[ind, 1], SPACER, kernel_types[ind, 2], SPACER, kernel_types[ind, 3]
        )
    end
end
println("\n\n")

min_times = minimum.(eachcol(arr))

println("Kernel evaluation benchmarks:")
println(HLINE)
println(
    rpad("BASE KERNEL", LENGTH),
    rpad("   K", 7),
    SPACER,
    rpad("Naive I", 7),
    SPACER,
    rpad("   I", 7),
)
println(HLINE)
for (ind, (name, _)) in enumerate(base_kernels)
    text1 = display(arr[ind, 1] / baseline)
    h1 = arr[ind, 1] == min_times[1]
    text2 = display(arr[ind, 2] / baseline)
    text3 = display(arr[ind, 3] / baseline)

    h1 = arr[ind, 1] == min_times[1]
    h2 = arr[ind, 2] == min_times[2]
    h3 = arr[ind, 3] == min_times[3]

    println(
        rpad(name * ": ", LENGTH),
        h1 ? highlight(text1) : text1,
        SPACER,
        h2 ? highlight(text2) : text2,
        SPACER,
        h3 ? highlight(text3) : text3,
    )
end
println(HLINE)

println("")
println(rpad("Baseline:", LENGTH), display(baseline))
println("")
