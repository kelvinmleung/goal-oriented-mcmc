include("inverseProblem.jl")
include("mcmc_1d.jl")
using StatsPlots, NPZ

Random.seed!(123)





n, p = 326, 1

O = vcat(zeros(2), npzread("data_canopy/goal_op_4_unscaled.npy"))' / npzread("data_refl/177/prscale_177.npy")
O_offset = npzread("data_canopy/goal_op_const_4_unscaled.npy")
x_true = npzread("data_refl/177/x_177.npy")
z_true = O[3:end]' * x_true + O_offset
y = npzread("data_refl/177/y_177.npy")



λ_ranges = [400.0 1300.0; 1450.0 1780.0; 2051.0 2451.0]
λ_idx = AOE.get_λ_idx(collect(346.29958:5.0086700464037115:2505.03637), λ_ranges)
priormodel, wls = get_priormodel(:standard; λ_ranges) # PriorModel instance
rtmodel = AOE.get_radiative_transfer_modtran(:LUTRT1; λ_ranges);
rdbufs = get_RetrievalData_bufs(nλ) 
xa, xs = AOE.invert(y, rdbufs[1], rtmodel, priormodel)
fx = AOE.fwdfun(xa, xs, rtmodel) 
dfx = AOE.gradfwd_accel(xa, xs, rtmodel, fx)[:,3:end]
x_map = vcat(xa,xs)

# Inference parameters
μ_x = vcat([0.2; 1.3], npzread("data_refl/177/prmean_177.npy"))
Γ_x = zeros((328, 328))
Γ_x[1:2,1:2] = [0.01 0; 0 0.04]
Γ_x[3:end,3:end] = npzread("data_refl/177/prcov_177.npy")
Γ_ϵ = diagm(y * 1e-4)
μ_z = O * μ_x
Γ_z = O * Γ_x * O'
invΓ_x, invΓ_z, invΓ_ϵ = inv(Γ_x), inv(Γ_z), inv(Γ_ϵ)
Q = Γ_x * O' * invΓ_z

m = 10000

normDist = MvNormal(μ_x, Γ_x)
x_prsamp = rand(normDist, m)
z_prsamp = (O * x_prsamp)' .+ O_offset


# Load MCMC chain from Master's paper
mcmcchain_refl = vcat(npzread("/Users/kmleung/Documents/JPLproject/resultsGibbs/177_SNR50_RandWalkIsofitCovEps0_11_2M/mcmcchain.npy")[λ_idx,1:4:end], npzread("/Users/kmleung/Documents/JPLproject/resultsGibbs/177_SNR50_RandWalkIsofitCovEps0_11_2M/mcmcchain.npy")[end-1:end,1:4:end])
z_fullrank = O*mcmcchain_refl

# low rank 1D
# @time z_possamp_lowrank_177 = mcmc_lis_1d(μ_z[1], μ_x, Γ_x, Γ_ϵ, Q, O, y; N=m) .+ O_offset
# @time z_possamp_lowrank = mcmc_lis_unified(μ_z[1], μ_x, Γ_x, Γ_ϵ, Q, O, y; N=m) .+ O_offset
@time z_possamp_lowrank_177 = mcmc_lis_1d(vcat(xa,xs), μ_x, Γ_x, Γ_ϵ, Q, O, y; N=m) .+ O_offset

# MAP
Γ_laplace = inv(invΓ_x[3:end,3:end] + dfx' * invΓ_ϵ * dfx)
laplaceDist = MvNormal(xs, (tril(Γ_laplace)+ tril(Γ_laplace,-1)'))
x_laplacesamp_177 = rand(laplaceDist, m)
z_laplacesamp_177 = (O[3:end]' * x_laplacesamp_177)' .+ O_offset




density(z_possamp_lowrank_177[2000:1:end], color=:blue, linewidth=2, label="Low Rank",  title="1D Goal Posterior - Marginal Density", xlim=[-0.1,0.1])
density!(z_prsamp[1:1:end], color=:black, linewidth=1, label="Prior")
# density!(z_laplacesamp_177[1:1:end], color=:red, linewidth=1, label="Laplace")
density!(z_fullrank[2500:10:end], color=:red, linewidth=2, label="Naive")
# density!(z_possamp[950000:10:end], color=:red, linewidth=2, label="Naive")#, xlim=[0.1,0.4])
plot!([z_true], seriestype="vline", color=:black, linewidth=3, label="Truth")
# plot!([O*x_map+O_offset], seriestype="vline", color=:red, linewidth=3, label="MAP")
plot!([mean(z_possamp_lowrank_177[2000:1:end])], seriestype="vline", color=:blue, linewidth=3, label="Pos Mean")



# ## MCMC Chain plots
plot(1:20:50000*20,z_fullrank[1:1:end], xlabel="Sample number", ylabel="Z", title="Naive MCMC", label=false)
plot(z_possamp_lowrank_177[1:10000], xlabel="Sample number", ylabel="Z", title="Goal-oriented MCMC", label=false)


