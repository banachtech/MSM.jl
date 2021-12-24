using DataFrames, CSV, DelimitedFiles, Plots, MSM, Statistics

x = readdlm("data/btc.csv", ',')
x = vec(x)

n = length(x) - 240
res = DataFrame()
c = 0
for j in 2001:240:n
    m = MSMmodel(k = 6)
    fitglobal!(m, x[j-2000:j-1])
    s = predict(m, 500, 240)
    v = std(x[j:j+240])
    push!(res, (realised = v, msmmean = mean(s), msmsd = std(s)))
    println("$c done")
    c += 1
end

CSV.write("data/btcvol.csv", res)

plt = plot(res.realised, label = "realised", lw = 2)
plot!(plt, res.msmmean .- res.msmsd, label = "msm -1sd", lw = 2)
plot!(plt, res.msmmean .+ res.msmsd, label = "msm +1sd", lw = 2)

plt1 = plot(res.msmmean .- res.msmsd, fillrange = res.msmmean .+ res.msmsd, fillalpha = 0.35, color = :orange, label = "msm +/- 1σ")
plot!(plt1, res.msmmean, label = "msm mean", lw = 2, color = :blue)
plot!(plt1, res.realised, label = "realised", lw = 2, color = :red)
savefig(plt1, "btcvoljulia.png")

begin
    rv = readdlm("data/hvol.csv", ',')
    mv = readdlm("data/msmvol_mean.csv", ',')
    sd = readdlm("data/msmvol_sd.csv", ',')
    plt2 = plot(mv .- sd, fillrange = mv .+ sd, fillalpha = 0.35, color = :orange, label = "msm +/- 1σ")
    plot!(plt2, mv, label = "msm mean", lw = 2, color = :blue)
    plot!(plt2, rv, label = "realised", lw = 2, color = :red)
    savefig(plt2, "btcvolpy.png")
end