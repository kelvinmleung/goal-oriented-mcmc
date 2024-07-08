include("../src/forward.jl")
include("../src/mcmc_1d.jl")
include("../src/mcmc_aoe.jl")
using StatsPlots, NPZ, AOE

Random.seed!(123)


λ_ranges = [400.0 1300.0; 1450.0 1780.0; 2051.0 2451.0]
priormodel, wls = get_priormodel(:standard; λ_ranges) # PriorModel instance
rtmodel = AOE.get_radiative_transfer_modtran(:LUTRT1; λ_ranges);
rdbufs = get_RetrievalData_bufs(326) 


n, p = 326, 1
nλ=326

O = vcat(zeros(2), npzread("data/data_canopy/goal_op_8_unscaled.npy"))' / npzread("data/data_canopy/prscale_1_1.npy")
O_offset = npzread("data/data_canopy/goal_op_const_8_unscaled.npy")
x_true = npzread("data/data_canopy/s_true.npy")[1,1,:] #x_true atm = 0.19, 1.31
z_true = O[3:end]' * x_true + O_offset
y = npzread("data/data_canopy/y.npy")[1,1,:] + rand(MvNormal(zeros(n), diagm(AOE.dummy_noisemodel(npzread("data/data_canopy/y.npy")[1,1,:]))))


xa, xs = AOE.invert(y, rdbufs[1], rtmodel, priormodel)
fx = AOE.fwdfun(xa, xs, rtmodel) 
dfx = AOE.gradfwd_accel(xa, xs, rtmodel, fx)[:,3:end]
x_map = vcat(xa,xs)

# Inference parameters
μ_x = vcat([0.2; 1.3], npzread("data/data_canopy/prmean_1_1.npy"))
Γ_x = zeros((328, 328))
Γ_x[1:2,1:2] = [0.01 0; 0 0.04]
Γ_x[3:end,3:end] = npzread("data/data_canopy/prcov_1_1.npy")

Γ_ϵ = diagm(AOE.dummy_noisemodel(y))# diagm(y * 1e-4)

μ_z = O * μ_x #+ O_offset
Γ_z = O * Γ_x * O'
invΓ_x, invΓ_z, invΓ_ϵ = inv(Γ_x), inv(Γ_z), inv(Γ_ϵ)
Q = Γ_x * O' * invΓ_z

m = 10000

normDist = MvNormal(μ_x, Γ_x)
x_prsamp = rand(normDist, m)
z_prsamp = (O * x_prsamp)' .+ O_offset

# @time x_possamp = mcmc_amm_simple(vcat([0.2; 1.3],x_true), μ_x, Γ_x, Γ_ϵ, y, m)
# x_pos_mean = mean(x_possamp[:,Int(m/2):end], dims=2)
# z_possamp = (O * x_possamp)' .+ O_offset
# # npzwrite("data_canopy/z_chain_1_1_jun4_naiveMCMC.npy", z_possamp)

x_prsamp = rand(normDist, m)
z_prsamp = O * x_prsamp
noiseDist = MvNormal(zeros(n), Γ_ϵ)
y_prsamp_pred = rand(noiseDist, m)

for i = 1:m
    y_prsamp_pred[:,i] = y_prsamp_pred[:,i] + aoe_fwdfun(x_prsamp[:,i])
end

yz_prsamp = hcat(y_prsamp_pred', (z_prsamp .- O * μ_x)')
nComp = 10

gmm = GMM(nComp, yz_prsamp, method=:kmeans, kind=:full, nInit=100, nIter=50, nFinal=50)

@time x_possamp = mcmc_bm_3block(μ_x, Γ_x, Γ_ϵ, y, 1000000)
# x_pos_mean = mean(x_possamp[:,Int(m/2):end], dims=2)
z_possamp_naive = (O * x_possamp)' .+ O_offset


# low rank 1D
@time z_possamp_lowrank_covexpand = mcmc_lis_1d(vcat(xa,xs), μ_x, Γ_x, Γ_ϵ, Q, O, y; N=m, logposmethod="covexpand") .+ O_offset
npzwrite("data/data_canopy/z_covexpand_june28.npy", z_possamp_lowrank_covexpand)

@time z_possamp_lowrank_gmm = mcmc_lis_1d(vcat(xa,xs), μ_x, Γ_x, Γ_ϵ, Q, O, y; N=m, logposmethod="gmm") .+ O_offset
npzwrite("data/data_canopy/z_gmm_june28.npy", z_possamp_lowrank_gmm)

@time z_possamp_lowrank_pseudomarg = mcmc_lis_1d(vcat(xa,xs), μ_x, Γ_x, Γ_ϵ, Q, O, y; N=m, logposmethod="pseudomarg") .+ O_offset
npzwrite("data/data_canopy/z_naive_june28.npy", z_possamp_naive)



density(z_possamp_lowrank_gmm[2000:1:end], color=:blue, linewidth=2, label="Low Rank - GMM",  title="1D Goal Posterior - Marginal Density")
density!(z_possamp_lowrank_covexpand[2000:1:end], color=:red, linewidth=2, label="Low Rank - CovExpand")
# density!(z_possamp_lowrank_pseudomarg[2000:1:end], color=:green, linewidth=2, label="Low Rank - Pseudomarg")

# density!(z_prsamp[1:1:end], color=:black÷ linewidth=1, label="Prior")
# density!(z_laplacesamp[1:1:end], color=:red, linewidth=1, label="Laplace")

density!(z_possamp_naive[20000:10:end], color=:black, linewidth=2, label="Naive")#, xlim=[0.1,0.4])
plot!([z_true], seriestype="vline", color=:black, linewidth=3, label="Truth")





## previous stuff
# @time z_possamp_lowrank_initpr = mcmc_lis_1d(μ_x, μ_x, Γ_x, Γ_ϵ, Q, O, y; N=m) .+ O_offset

# @time z_possamp_lowrank = mcmc_lis_unified(μ_z[1], μ_x, Γ_x, Γ_ϵ, Q, O, y; N=m) .+ O_offset
# npzwrite("data_canopy/z_chain_1_1_jun29.npy", z_possamp_lowrank)

# MAP
Γ_laplace = inv(invΓ_x[3:end,3:end] + dfx' * invΓ_ϵ * dfx)
laplaceDist = MvNormal(xs, (tril(Γ_laplace)+ tril(Γ_laplace,-1)'))
x_laplacesamp = rand(laplaceDist, m)
z_laplacesamp = (O[3:end]' * x_laplacesamp)' .+ O_offset




density(z_possamp_lowrank[2000:1:end], color=:blue, linewidth=2, label="Low Rank",  title="1D Goal Posterior - Marginal Density", xlim=[0.1,0.3])
density!(z_possamp_lowrank_initpr[1:1:end], color=:blue, linewidth=2, label="Low Rank - Init Prior")

density!(z_prsamp[1:1:end], color=:black, linewidth=1, label="Prior")
density!(z_laplacesamp[1:1:end], color=:red, linewidth=1, label="Laplace")

# density!(z_possamp[950000:10:end], color=:red, linewidth=2, label="Naive")#, xlim=[0.1,0.4])
plot!([z_true], seriestype="vline", color=:black, linewidth=3, label="Truth")
# plot!([O*x_map+O_offset], seriestype="vline", color=:red, linewidth=3, label="MAP")
# plot!([mean(z_possamp_lowrank[2000:1:end])], seriestype="vline", color=:blue, linewidth=3, label="Pos Mean")




# # ## MCMC Chain plots
# plot(1:100:m,z_possamp[1:100:m], xlabel="Sample number", ylabel="Z", title="Naive MCMC", label=false)
plot(z_possamp_lowrank_gmm[1:end], xlabel="Sample number", ylabel="Z", title="Goal-oriented MCMC", label=false)


