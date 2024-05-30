include("inverseProblem.jl")
include("mcmc.jl")
using StatsPlots

Random.seed!(123)



n, p = 326, 1

O = vcat(zeros(2), npzread("data_canopy/goal_op_8_unscaled.npy"))' / npzread("data_canopy/prscale_1_1.npy")
O_offset = npzread("data_canopy/goal_op_const_8_unscaled.npy")
x_true = npzread("data_canopy/s_true.npy")[1,1,:] #x_true atm = 0.19, 1.31
z_true = O[3:end]' * x_true + O_offset
y = npzread("data_canopy/y.npy")[1,1,:]

# Inference parameters
μ_x = vcat([0.2; 1.3], npzread("data_canopy/prmean_1_1.npy"))
Γ_x = zeros((328, 328))
Γ_x[1:2,1:2] = [0.01 0; 0 0.04]
Γ_x[3:end,3:end] = npzread("data_canopy/prcov_1_1.npy")
Γ_ϵ = diagm(y * 1e-4)
μ_z = O * μ_x
Γ_z = O * Γ_x * O'
invΓ_x, invΓ_z, invΓ_ϵ = inv(Γ_x), inv(Γ_z), inv(Γ_ϵ)
Q = Γ_x * O' * invΓ_z

m = 10000

normDist = MvNormal(μ_x, Γ_x)
x_prsamp = rand(normDist, m)
z_prsamp = (O * x_prsamp)' .+ O_offset


# @time x_possamp = mcmc_amm_simple(μ_x, μ_x, Γ_x, Γ_ϵ, y, m+20000)[:,20001:end]
# x_pos_mean = mean(x_possamp[:,Int(m/2):end], dims=2)
# z_possamp = O * x_possamp


# low rank 1D
@time z_possamp_lowrank = mcmc_lis_1d(μ_z[1], μ_x, Γ_x, Γ_ϵ, Q, O, y; N=m) .+ O_offset



# ## SAVE THIS TO NPY!!!!!!!
# npzwrite("data_canopy/z_chain_1_1_may29.npy", z_possamp_lowrank)







density(z_possamp_lowrank[2500:10:end], color=:blue, linewidth=2, label="Low Rank",  title="1D Goal Posterior - Marginal Density", xlim=[0.1,0.4])
density!(z_prsamp[2500:10:end], color=:black, linewidth=1, label="Prior")
# plot!([mean(z_possamp[1:100:end])], seriestype="vline", color=:red3, linewidth=3, label=false)
# plot!([mean(z_possamp_lowrank[1:100:end])], seriestype="vline", color=:blue3, linewidth=3, label=false)
# plot!([mean(z_prsamp[1:100:end])], seriestype="vline", color=:black, linewidth=2, label=false)
plot!([z_true], seriestype="vline", color=:black, linewidth=3, label="Truth")


# ## MCMC Chain plots
plot(z_possamp, xlabel="Sample number", ylabel="Z", title="Naive MCMC", label=false)
plot(z_possamp_lowrank[1:10000], xlabel="Sample number", ylabel="Z", title="Goal-oriented MCMC", label=false)


