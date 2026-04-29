using HCubature
using IntegratedGPs
using SpecialFunctions

α = 0.5

ν = 3.5
ρ = 2.0
σ2 = 1.0

S = 0.9
T = 0.8
Δ = abs(S - T)

gp = CPEMaternGP(ν, ρ, σ2)

K_numerical =
    exp(α * (S + T)) * HCubature.hcubature(
        x -> exp(-α * (x[1] + x[2])) * kernel(gp, x[1], x[2]), [0.0, 0.0], [S, T]
    )[1]

I_plus(t) = HCubature.hcubature(x -> exp(α * x[1]) * kernel(gp, 0.0, x[1]), [0.0], [t])[1]
I_minus(t) = HCubature.hcubature(x -> exp(-α * x[1]) * kernel(gp, 0.0, x[1]), [0.0], [t])[1]

contribution(t) = I_minus(t) - exp(-2α * t) * I_plus(t)

K_analytical =
    exp(α * (S + T)) *
    (contribution(S) + contribution(T) - exp(-2α * S) * contribution(Δ)) / (2α)

K_analytical =
    exp(α * (S + T)) *
    (contribution(S) + contribution(T) - exp(-2α * S) * contribution(Δ)) / (2α)

println("Numerical:  $K_numerical")
println("Analytical: $K_analytical")

# Now try to compute I_plus in closed form
I_plus_numerical = I_plus(T)

gamma_inc_lower(a, x) = gamma(a) * gamma_inc(a, x)[1]

beta, poly = only(gp.cpe.polynomials)
# 1.3228756555322954 => ImmutablePolynomial(1.0 + 1.3228756555322954*x + 0.7000000000000001*x^2 + 0.15433549314543446*x^3)

I_plus_analytical = 0.0
k = beta - α
for (i, c) in enumerate(poly.coeffs)
    n = i - 1
    I_plus_analytical += c / k^(n + 1) * gamma_inc_lower(n + 1, k * T)
end

println("I_plus numerical:  $I_plus_numerical")
println("I_plus analytical: $I_plus_analytical")

I_minus_numerical = I_minus(T)
k = beta + α
I_minus_analytical = 0.0
for (i, c) in enumerate(poly.coeffs)
    n = i - 1
    I_minus_analytical += c / k^(n + 1) * gamma_inc_lower(n + 1, k * T)
    (n + 1, k * T)
end
println("I_minus numerical:  $I_minus_numerical")
println("I_minus analytical: $I_minus_analytical")
