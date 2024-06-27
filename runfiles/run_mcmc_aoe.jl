

include("mcmc_aoe.jl")
using StatsPlots, NPZ

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
Γ_x[1:2,1:2] = [0.01 0; 0 4]
Γ_x[3:end,3:end] = npzread("data_canopy/prcov_1_1.npy")
Γ_ϵ = diagm(y * 1e-4)
μ_z = O * μ_x
Γ_z = O * Γ_x * O'
invΓ_x, invΓ_z, invΓ_ϵ = inv(Γ_x), inv(Γ_z), inv(Γ_ϵ)
Q = Γ_x * O' * invΓ_z

m = 100000

normDist = MvNormal(μ_x, Γ_x)
x_prsamp = rand(normDist, m)
z_prsamp = (O * x_prsamp)' .+ O_offset

@time x_possamp = mcmc_bm(μ_x, Γ_x, Γ_ϵ, y, m)
# x_pos_mean = mean(x_possamp[:,Int(m/2):end], dims=2)
z_possamp = (O * x_possamp)' .+ O_offset


# low rank 1D

# @time z_possamp_lowrank = mcmc_lis_1d(vcat(xa,xs), μ_x, Γ_x, Γ_ϵ, Q, O, y; N=Int(m/10)) .+ O_offset

# @time z_possamp_lowrank = mcmc_lis_1d(μ_z[1], μ_x, Γ_x, Γ_ϵ, Q, O, y; N=m) 

# @time z_possamp_lowrank = mcmc_lis_unified(μ_z[1], μ_x, Γ_x, Γ_ϵ, Q, O, y; N=m) .+ O_offset




# ## SAVE THIS TO NPY!!!!!!!
# npzwrite("data_canopy/z_chain_1_1_jun29.npy", z_possamp_lowrank)

scatter(x_possamp[1,50000:100:end], x_possamp[2,50000:100:end])

density(z_possamp_lowrank[2500:10:end], color=:blue, linewidth=2, label="Low Rank",  title="1D Goal Posterior - Marginal Density")#, xlim=[0.1,0.4])
density!(z_prsamp[2500:10:end], color=:black, linewidth=1, label="Prior")
density(z_possamp[5000:10:end], color=:red, linewidth=2, label="Naive")#, xlim=[0.1,0.4])
plot!([z_true], seriestype="vline", color=:black, linewidth=3, label="Truth")



# ## MCMC Chain plots
plot(1:100:m,z_possamp[1:100:m], xlabel="Sample number", ylabel="Z", title="Naive MCMC", label=false)
plot(z_possamp_lowrank[1:10000], xlabel="Sample number", ylabel="Z", title="Goal-oriented MCMC", label=false)


