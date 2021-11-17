using Kronecker, Random, LinearAlgebra, Distributions, IntervalOptimisation, IntervalArithmetic, DelimitedFiles

Random.seed!(1234)
const PPI = 1.0 / sqrt(2π)

# transition probability matrix
function Φ(b, γₖ, k::Integer)
    γ = [1 - (1 - γₖ)^(b^(j - k)) for j ∈ 1:k] .* 0.5
    return Matrix(reduce(⊗, [[1.0 - u u; u 1.0 - u] for u ∈ γ]))
end

# state dependent volatility
Σ(σ₀, m₀, k::Integer, M::Vector{Int}) = [σ₀ * sqrt(((2.0 - m₀)^u) * (m₀^(k - u))) for u ∈ M]  

function nll(p, k, M, A, Π, rt)
    b, m₀, γₖ, σ₀ = p
    A .= Φ(b, γₖ, k)
    σ = Σ(σ₀, m₀, k, M)
    ll = 0.0
    for r in rt
        ωr = [PPI * exp(-0.5 * r * r / (s * s)) / s for s in σ]
        w = ωr .* (A * Π)
        sw = sum(w)
        ll += log(sw)
        Π = w ./ sw
    end
    return -ll
end

x = vec(readdlm("/users/balaji/projects/MSM.jl/data/btc.csv", ',', header=true)[1])
k = 4
b = 1..20
m₀ = 1..2
γₖ = 0..1
σ₀ = 0..5
p = IntervalBox([b, m₀, γₖ, σ₀])
M = sum.([digits(q, base=2, pad=k) for q in 0:2^k - 1])
N = length(M)
Π = fill(1.0 / N, N)
A = Φ(b, γₖ, k)

res, xmin = minimise(p -> nll(p, k, M, A, Π, x), p)