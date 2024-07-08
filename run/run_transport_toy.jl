include("../src/forward.jl")
include("../src/mcmc_simple.jl")
include("../src/mcmc_1d.jl")

Random.seed!(123)
using TransportBasedInference, StatsPlots


O = [1 0 0]
p, n = size(O)
# Goal oriented 

# Inference parameters
μ_x = 0.05*ones(n)#zeros(n) #
μ_z = O * μ_x
σ_X²= 0.1
σ_ϵ² = 1e-4
Γ_ϵ = diagm(σ_ϵ² * ones(n))
Γ_x = diagm(σ_X² * ones(n))
Γ_z = O * Γ_x * O'
invΓ_x, invΓ_z, invΓ_ϵ = inv(Γ_x), inv(Γ_z), inv(Γ_ϵ)
Q = Γ_x * O' * invΓ_z

normDist = MvNormal(μ_x, Γ_x)
noiseDist = MvNormal(zeros(n), Γ_ϵ)
x_true = [-0.06457306721039767; -0.14632513788889215; -0.16236037455860808] #rand(normDist)
z_true = O * x_true
# Apply forward model
eps = sqrt(σ_ϵ²) .* randn(n)
y = fwdtoy(x_true) + eps

m = 1000
Nx = 1 # Dimension of the state
Ny = 3 # Dimension of the observation
x_prsamp = rand(normDist, m)
z_prsamp = O * x_prsamp
y_prsamp_pred = rand(noiseDist, m)

for i = 1:m
    y_prsamp_pred[:,i] = y_prsamp_pred[:,i] + fwdtoy(x_prsamp[:,i])
end


X = vcat(y_prsamp_pred, (z_prsamp .- O * μ_x))


S = HermiteMap(10, X; diag = true, factor = 0.5, α = 1e-6, b = "ProHermiteBasis");

@time S = optimize(S, X, "split"; maxterms = 30, withconstant = true, withqr = true, verbose = true, 
                                  maxpatience = 30, start = 1, hessprecond = true)

F = evaluate(S, X; start = Ny+1)

# Let's generate the posterior samples by partially inverting the map $\boldsymbol{S}^{\boldsymbol{\mathcal{X}}}$, for $\boldsymbol{y}^\star = 0.25$
Ystar = y

Xa = deepcopy(X)
@time hybridinverse!(Xa, F, S, Ystar; apply_rescaling = true, start = 2)

histogram(sort(Xa[4,:]), xlim = (-Inf, Inf), bins = 90, normalize = :pdf, label = "Posterior samples", color = "skyblue2")
z_possamp_transport = Xa[4,:]                           

## MCMC ##
x0 = x_true
@time x_possamp = mcmc_amm_simple(x0, μ_x, Γ_x, Γ_ϵ, y, m*10)
z_possamp = O * x_possamp

# @time z_possamp_gmm = mcmc_lis_1d(x0, μ_x, Γ_x, Γ_ϵ, Q, O, y; N=Int(m/10))
@time z_possamp_covexpand = mcmc_lis_1d(x0, μ_x, Γ_x, Γ_ϵ, Q, O, y; N=m)


density(z_possamp[100000:1:end], color=:black, linewidth=2, label="Naive", title="1D Goal Posterior - Marginal Density")#, xlims=(-0.1,0.06))
density!(z_possamp_transport[10000:10:end], color=:blue, linewidth=2, label="Transport")
# density!(z_possamp_cdr[10000:10:end], color=:red, linewidth=2, label="CDR")
density!(z_possamp_covexpand[10000:10:end], color=:green, linewidth=2, label="CovExpand")
density!(z_prsamp[1:10:end], color=:black, linewidth=1, label="Prior")
plot!([z_true], seriestype="vline", color=:black, linewidth=3, label="Truth")

plot(z_possamp[1:50000], xlabel="Sample number", ylabel="Z", title="Naive MCMC", label=false)
plot(z_possamp_lowrank[1:50000], xlabel="Sample number", ylabel="Z", title="Goal-oriented MCMC", label=false)
plot(z_possamp_transport)
# npzwrite("z_possamp_cdr.npy",z_possamp_cdr)