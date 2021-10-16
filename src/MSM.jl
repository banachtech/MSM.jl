"""
Module MSM implements Calvet and Fisher's Markov Switching Multifractal model.

# Example usge:
```jldocs
julia> m = MSMmodel()
MSMmodel
  k: Int64 4
  b: Float64 5.0
  m₀: Float64 1.7
  γₖ: Float64 0.9
  σ₀: Float64 0.01
# Fit model to data x
julia> fit!(m, x);
# Predict volatility of x
julia> s = predict(m);
julia> histogram(s)
# Simulate a model
julia> simulate(m, nsims = 10000)
```
"""
module MSM

export fit!, fitglobal!, predict, simulate, tune

using Parameters, DataFrames, Statistics, Random, Optim, Kronecker, LinearAlgebra, Distributions, BlackBoxOptim

Random.seed!(1234)
const PPI = 1.0 / sqrt(2π)

"""
    MSMmodel(k, b, m₀, γₖ, σ₀)

Constructor for Markov Switching Multifractal (MSM) model.

# Example:
```jldocs
julia> m = MSMmodel()
MSMmodel
  k: Int64 4
  b: Float64 5.0
  m₀: Float64 1.7
  γₖ: Float64 0.9
  σ₀: Float64 0.01
```
"""
@with_kw mutable struct MSMmodel @deftype Float64
    k::Int = 4; @assert k >= 1
    b = 5.0; @assert b > 1.0
    m₀ = 1.7; @assert m₀ > 1. && m₀ < 2.
    γₖ = 0.90; @assert γₖ > 0. && γₖ < 1.
    σ₀ = 0.01; @assert σ₀ > 0.
end

# transition probability matrix
function Φ(b, γₖ, k::Integer)
    γ = [1 - (1 - γₖ)^(b^(j - k)) for j ∈ 1:k] .* 0.5
    return Matrix(reduce(⊗, [[1.0 - u u; u 1.0 - u] for u ∈ γ]))
end

# state dependent volatility
Σ(σ₀, m₀, k::Integer, M::Vector{Int}) = [σ₀ * sqrt(((2.0 - m₀)^u) * (m₀^(k - u))) for u ∈ M]  

Σ(σ₀, m₀, k::Integer, m::Int) = σ₀ * sqrt(((2.0 - m₀)^m) * (m₀^(k - m)))

"""
    fit!(model::MSMmodel, x::Vector{Float64}; time_limit=60., solver=NelderMead(), g_tol=1.0e-6)

Fit MSM model to a vector of real-valued, de-meaned data using Optim's optimization methods. Recommended method is the default method.

# Example:
```jldocs
julia> using MSM; 
julia> m = MSMmodel(); fit!(m, x)
```
"""
function fit!(model::MSMmodel, x::Vector{Float64}; time_limit=60., solver=NelderMead(), g_tol=1.0e-6)
    @unpack k, b, m₀, γₖ, σ₀ = model
    p0 = [b, m₀, γₖ, σ₀]
    l = [1.001, 1.001, 0.001, 0.0001]
    h = [50.0, 1.999, 0.999, 5.0]
    M = sum.([digits(q, base=2, pad=k) for q in 0:2^k - 1])
    N = length(M)
    Π = fill(1.0 / N, N)
    A = Φ(b, γₖ, k)
    res = optimize(p -> nll(p, k, M, A, Π, x), l, h, p0, Fminbox(solver), Optim.Options(time_limit=time_limit, g_tol=g_tol))
    b, m₀, γₖ, σ₀ = res.minimizer
    @pack! model = b, m₀, γₖ, σ₀
    return res
end

"""
    fitglobal!(model::MSMmodel, x::Vector{Float64}; method = :adaptive_de_rand_1_bin_radiuslimited,  maxsteps = 10000, tracemode = :verbose)

Fit MSM model to a vector of real-valued, de-meaned data using BlackBoxOptim's global optimization methods. Recommended method is the default method.

# Example:
```jldocs
julia> using MSM; 
julia> m = MSMmodel(); fitglobal!(m, x)
```

tracemode = :silent suppresses optimizer trace messages.
"""
function fitglobal!(model::MSMmodel, x::Vector{Float64}; method=:adaptive_de_rand_1_bin_radiuslimited,  maxsteps=10000, tracemode=:verbose)
    @unpack k, b, m₀, γₖ, σ₀ = model
    l = [1.001, 1.001, 0.001, 0.0001]
    h = [50.0, 1.999, 0.999, 5.0]
    M = sum.([digits(q, base=2, pad=k) for q in 0:2^k - 1])
    N = length(M)
    Π = fill(1.0 / N, N)
    A = Φ(b, γₖ, k)
    res = bboptimize(p -> nll(p, k, M, A, Π, x), SearchRange=collect(zip(l, h)), Method=:adaptive_de_rand_1_bin_radiuslimited, MaxSteps=maxsteps, TraceMode=tracemode)
    b, m₀, γₖ, σ₀ = best_candidate(res)
    @pack! model = b, m₀, γₖ, σ₀
    return res
end

"""
    nll(p::Vector{<:Real}, k::Integer, M::Vector{Int}, A::Matrix{Float64}, Π::Vector{Float64}, rt::Vector{Float64})

compute negative log-likelihood of MSM. 
"""
function nll(p::Vector{<:Real}, k::Integer, M::Vector{Int}, A::Matrix{Float64}, Π::Vector{Float64}, rt::Vector{Float64})
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

"""
    predict(model::MSMmodel; window=100, nsamples=100)

Predict volatility implied by an MSM model through simulation. Window is the length of observations over which volatility (standard deviation) is computed and nsamples is the number of such windows simulated. Returns a vector of length nsamples containing volatilities.
"""
function predict(model::MSMmodel; window=100, nsamples=100)
    s = fill(0., nsamples)
    @inbounds for i = 1:nsamples
        r = simulate(model, nsims=window)
        s[i] = std(r)
    end
return s
end

"""
    simulate(model::MSMmodel; nsims=1000)

Simulate an MSM model's trajectory of length nsims. Returns a vector.
"""
function simulate(model::MSMmodel; nsims=1000)
    @unpack k, b, m₀, γₖ, σ₀ = model
    A = Φ(b, γₖ, k)
    M = sum.([digits(q, base=2, pad=k) for q in 0:2^k - 1])
    N = length(M)
    u = rand(Categorical(N))
    r = Vector{Float64}(undef, nsims)
    r[1] = Σ(σ₀, m₀, k, M[u]) * randn()
    @inbounds for j = 2:nsims
        u = rand(Categorical(A[u,:]))
        r[j] = Σ(σ₀, m₀, k, M[u]) * randn()
    end
return r
end

"""
    tune(x::Vector{<:Real}; ks=2:12)

Fit MSM models for different k's and output results. k > 10 may take more than a minute to fit. Returns a dataframe.

"""
function tune(x::Vector{<:Real}; ks=2:12)
    res = DataFrame()
    for j in ks
        model = MSM(k=j, σ₀=std(x))
        out = @timed fit!(model, x)
        @unpack k, b, m₀, γₖ, σ₀ = model
        push!(res, (;([:exec_time, :loglik, :k, :b, :m₀, :γₖ, :σ₀] .=> [out.time, -out.value.minimum, k, b, m₀, γₖ, σ₀])...))
        println("k = $(k) took $(out.time) seconds.")
    end
    return res
end

end