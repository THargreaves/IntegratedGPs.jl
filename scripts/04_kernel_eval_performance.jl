using IntegratedMaternGPs
using Polynomials

function evals_per_second(f, timeout::Float64=1.0)
    cnt::Int32 = 0
    start_time = time()
    while (time() - start_time) < timeout
        f()
        cnt += 1
    end
    cnt / timeout
end
println("\n\nRUNNING TEST\n")

SIGDIGITS = 4

pure_poly(x) = 1 + 2 * x + 3 * x * x 
poly_poly = Polynomial([1, 2, 3])
cpe_poly = CompoundPolynomialExp(0 => [1, 2, 3])

evals_cpe = evals_per_second(() -> cpe_poly(1.0))
evals_poly = evals_per_second(() -> poly_poly(1.0))
evals_pure = evals_per_second(() -> pure_poly(1.0))

# TODO: CPE Evaluation seems to be way too slow across the board.

println("Polynomials")
println("CPE:  $(round(evals_cpe, sigdigits=SIGDIGITS))/s")
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



ν = 3.5
ρ = 2.4
σ2 = 5.4
gp_gen = MaternGP(ν, ρ, σ2)
gp_cpe = CPEMaternGP(ν, ρ, σ2)

evals_gen = evals_per_second(() -> kernel(gp_gen, 0.0, 1.0))
evals_cpe = evals_per_second(() -> kernel(gp_cpe, 0.0, 1.0))

println("Standard kernel:")
println("General:   $(round(evals_gen, sigdigits=SIGDIGITS))/s")
println("CPE:       $(round(evals_cpe, sigdigits=SIGDIGITS))/s")
println("")


naive_igp_gen = Integrated(gp_gen)
naive_igp_cpe = Integrated(gp_cpe)
igp_gen = IntegratedMaternGP(gp_gen, cache_size=0)
igp_cpe = IntegratedCPEMaternGP(gp_cpe, cache_size=0)
igp_cache = IntegratedCPEMaternGP(gp_cpe, cache_size=1000)

evals_naive_gen = evals_per_second(() -> kernel(naive_igp_gen, 1.0, 2.0))
evals_naive_cpe = evals_per_second(() -> kernel(naive_igp_cpe, 1.0, 2.0))
evals_gen = evals_per_second(() -> kernel(igp_gen, 1.0, 2.0))
evals_cpe = evals_per_second(() -> kernel(igp_cpe, 1.0, 2.0))
evals_cache = evals_per_second(() -> kernel(igp_cache, 1.0, 2.0))

println("Integrated kernel:")
println("Naive General: $(round(evals_naive_gen, sigdigits=SIGDIGITS))/s")
println("Naive CPE:     $(round(evals_naive_cpe, sigdigits=SIGDIGITS))/s")
println("General:       $(round(evals_gen, sigdigits=SIGDIGITS))/s")
println("CPE:           $(round(evals_cpe, sigdigits=SIGDIGITS))/s")
println("Cached:        $(round(evals_cache, sigdigits=SIGDIGITS))/s")
println("")
