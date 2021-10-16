using MSM

# initialize a MSM model
model = MSMmodel()

# generate fake data
x = simulate(model, nsims=2000)

# fit MSM to x
fit!(model, x)

fitglobal!(model, x, maxsteps=5000)