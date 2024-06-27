include("inverseProblem.jl")
include("mcmc.jl")
include("mcmc_1d.jl")

Random.seed!(123)

## Test run_godr_7 on linear toy problem 
## results of this test : 
## Naive MCMC works 
## Certified dimension reduction works if we use 1000 prior samples
using StatsPlots


O = [1 0]# 0]
p, n = size(O)

# Inference parameters
μ_x = 0.05*ones(n)#zeros(n) #
μ_z = O * μ_x
σ_X² = 0.1
σ_ϵ² = 1e-4
Γ_ϵ = diagm(σ_ϵ² * ones(n))
Γ_x = diagm(σ_X² * ones(n))
Γ_z = O * Γ_x * O'
invΓ_x, invΓ_z, invΓ_ϵ = inv(Γ_x), inv(Γ_z), inv(Γ_ϵ)
Q = Γ_x * O' * invΓ_z

normDist = MvNormal(μ_x, Γ_x)
x_true = [-0.06457306721039767; -0.14632513788889215]
z_true = O * x_true
# Apply forward model
eps = sqrt(σ_ϵ²) .* randn(n)
y = fwdtoy(x_true) + eps

G = [1 10; 1 -1]
Γ_xgy = inv(G' * invΓ_ϵ * G + invΓ_x)
μ_xgy = Γ_xgy * (invΓ_x * μ_x + G' *invΓ_ϵ * y)
μ_zgy = (O * μ_xgy)[1]
Γ_zgy = (O * Γ_xgy * O')[1,1]
z_possamp_true = rand(Normal(μ_zgy, sqrt(Γ_zgy)), 1000000)



m = 500000
x_prsamp = rand(normDist, m)
z_prsamp = O * x_prsamp


@time x_possamp = mcmc_amm_simple(μ_x, μ_x, Γ_x, Γ_ϵ, y, m+20000)[:,20001:end]
x_pos_mean = mean(x_possamp[:,Int(m/2):end], dims=2)
z_possamp = O * x_possamp

# low rank 1D
x0 = x_true
@time z_possamp_lowrank = mcmc_lis_1d(x0, μ_x, Γ_x, Γ_ϵ, Q, O, y; N=Int(m/10))


density(z_possamp[100000:10:end], color=:red, linewidth=2, label="Naive", title="1D Goal Posterior - Marginal Density", xlims=(-0.15,0.03))
density!(z_possamp_lowrank[1000:1:end], color=:blue, linewidth=2, label="Low Rank")
density!(z_possamp_true, color=:green, linewidth=2, label="True Posterior")
# density!(z_prsamp[1:10:end], color=:black, linewidth=1, label="Prior")
plot!([z_true], seriestype="vline", color=:black, linewidth=3, label="Truth")

# ## MCMC Chain plots
plot(z_possamp[1:10:end], xlabel="Sample number", ylabel="Z", title="Goal-oriented MCMC", label=false)
plot(z_possamp_lowrank[1:end], xlabel="Sample number", ylabel="Z", title="Goal-oriented MCMC", label=false)
# 

# npzwrite("data_pseudomarg_tests/z_pos_true.npy", z_possamp_true)
# npzwrite("data_pseudomarg_tests/z_pos_cdr_10prsamp.npy", z_possamp_lowrank)
# npzwrite("data_pseudomarg_tests/z_pos_naive.npy", z_possamp)



density(npzread("data_pseudomarg_tests/z_pos_naive.npy")[100000:10:end], color=:red, linewidth=2, label="Naive", title="1D Goal Posterior - Marginal Density")#, xlims=(-0.15,0.03))
density!(npzread("data_pseudomarg_tests/z_pos_true.npy"), color=:black, linewidth=2, label="True Posterior")
density!(npzread("data_pseudomarg_tests/z_pos_cdr_10prsamp.npy"), color=:blue, linewidth=2, label="CDR – 10 PrSamp", linestyle=:solid)
density!(npzread("data_pseudomarg_tests/z_pos_cdr_100prsamp.npy"), color=:blue, linewidth=2, label="CDR – 100 PrSamp", linestyle=:dash)
density!(npzread("data_pseudomarg_tests/z_pos_cdr_1000prsamp.npy"), color=:blue, linewidth=2, label="CDR – 1000 PrSamp", linestyle=:dashdot)
display(plot!([z_true], seriestype="vline", color=:black, linewidth=3, label="Truth"))

density(npzread("data_pseudomarg_tests/z_pos_naive.npy")[100000:10:end], color=:red, linewidth=2, label="Naive", title="1D Goal Posterior - Marginal Density", xlims=(-0.12,0.))
density!(npzread("data_pseudomarg_tests/z_pos_true.npy"), color=:black, linewidth=2, label="True Posterior")
density!(npzread("data_pseudomarg_tests/z_pos_cdr_10prsamp.npy"), color=:blue, linewidth=2, label="CDR – 10 PrSamp", linestyle=:solid)
density!(npzread("data_pseudomarg_tests/z_pos_cdr_100prsamp.npy"), color=:blue, linewidth=2, label="CDR – 100 PrSamp", linestyle=:dash)
density!(npzread("data_pseudomarg_tests/z_pos_cdr_1000prsamp.npy"), color=:blue, linewidth=2, label="CDR – 1000 PrSamp", linestyle=:dashdot)
display(plot!([z_true], seriestype="vline", color=:black, linewidth=3, label="Truth"))