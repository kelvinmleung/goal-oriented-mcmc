include("inverseProblem.jl")
include("mcmc_1d.jl")
using StatsPlots, NPZ

Random.seed!(123)



site_label = ["177", "306", "mars", "dark"]
site = 1

n, p = 326, 1

O = vcat(zeros(2), npzread("data_canopy/goal_op_4_unscaled.npy"))' / npzread("data_refl/" * site_label[site] * "/prscale_" * site_label[site] * ".npy")
O_offset = npzread("data_canopy/goal_op_const_4_unscaled.npy")
x_true = npzread("data_refl/" * site_label[site] * "/x_" * site_label[site] * ".npy")
z_true = O[3:end]' * x_true + O_offset
# y = npzread("data_refl/" * site_label[site] * "/y_" * site_label[site] * ".npy")
# y = npzread("/Users/kmleung/Documents/Github/transport-retrieval/data/y_" * site_label[site] * ".npy")
y = npzread("/Users/kmleung/Documents/JPLspatial/AOE.jl/examples/transport/ensembles_a=[0.2,1.5]/yobs_sim_" * site_label[site] * ".npy")[:,1]



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
μ_x = vcat([0.2; 1.3], npzread("data_refl/" * site_label[site] * "/prmean_" * site_label[site] * ".npy"))
Γ_x = zeros((328, 328))
Γ_x[1:2,1:2] = [0.01 0; 0 0.04]
Γ_x[3:end,3:end] = npzread("data_refl/" * site_label[site] * "/prcov_" * site_label[site] * ".npy")
Γ_ϵ = diagm(y * 1e-4)
μ_z = O * μ_x
Γ_z = O * Γ_x * O'
invΓ_x, invΓ_z, invΓ_ϵ = inv(Γ_x), inv(Γ_z), inv(Γ_ϵ)
Q = Γ_x * O' * invΓ_z

m = 10000

normDist = MvNormal(μ_x, Γ_x)
x_prsamp = rand(normDist, m)
z_prsamp = (O * x_prsamp)' .+ O_offset



x_prsamp = rand(normDist, m)
z_prsamp = O * x_prsamp
noiseDist = MvNormal(zeros(nλ), Γ_ϵ)
y_prsamp_pred = rand(noiseDist, m)

for i = 1:m
    y_prsamp_pred[:,i] = y_prsamp_pred[:,i] + aoe_fwdfun(x_prsamp[:,i])
end

yz_prsamp = hcat(y_prsamp_pred', (z_prsamp .- O * μ_x)')
nComp = 10

gmm = GMM(nComp, yz_prsamp, method=:kmeans, kind=:full, nInit=100, nIter=50, nFinal=50)
# gmm.hist







## Log Likelihood Plot ##
z_likelihood = collect(-0.5:0.1:0.5)
n_likelihood = length(z_likelihood)
likelihood_gmm = zeros(n_likelihood)
likelihood_cdr = zeros(n_likelihood)
likelihood_covexpand = zeros(n_likelihood)
for i = 1:n_likelihood
    likelihood_gmm[i] = gmm_likelihood(gmm, z_likelihood[i], y)
    likelihood_cdr[i] = @time cdr_likelihood(z_likelihood[i], μ_x, invΓ_ϵ, Q, O, y; offset=O_offset)
    likelihood_covexpand[i] = covexpand_likelihood(z_likelihood[i], μ_x, Γ_x, Γ_ϵ, Q, O, y; offset=O_offset)

end
plot(z_likelihood, likelihood_gmm, linewidth=2, label="GMM", xlabel="Z", ylabel="Log Likelihood")
# plot!(z_likelihood, likelihood_cdr, linewidth=2, label="CDR")
plot!(z_likelihood, likelihood_covexpand, linewidth=2, label="Expanded Cov")

# npzwrite("data_canopy/pseudomarg_likelihood_cwt_177sim_-0.5to0.5.npy", likelihood_cdr)



# Load MCMC chain from Master's paper
# mcmcchain_refl = vcat(npzread("/Users/kmleung/Documents/JPLproject/resultsGibbs/" * site_label[site] * "_SNR50_RandWalkIsofitCovEps0_11_2M/mcmcchain.npy")[λ_idx,1:4:end], npzread("/Users/kmleung/Documents/JPLproject/resultsGibbs/" * site_label[site] * "_SNR50_RandWalkIsofitCovEps0_11_2M/mcmcchain.npy")[end-1:end,1:4:end])
# z_fullrank = O*mcmcchain_refl .+ O_offset

x_possamp = npzread("data_refl/mcmc_results_fullrank/3block/mcmc_simobs_" * site_label[site] * ".npy")
z_fullrank = O * x_possamp .+ O_offset

# low rank 1D
# @time z_possamp_lowrank_" * site_label[site] * " = mcmc_lis_1d(μ_z[1], μ_x, Γ_x, Γ_ϵ, Q, O, y; N=m) .+ O_offset
# @time z_possamp_lowrank = mcmc_lis_unified(μ_z[1], μ_x, Γ_x, Γ_ϵ, Q, O, y; N=m) .+ O_offset
@time z_possamp_lowrank_177 = mcmc_lis_1d(vcat(xa,xs), μ_x, Γ_x, Γ_ϵ, Q, O, y; N=m) .+ O_offset

# MAP
Γ_laplace = inv(invΓ_x[3:end,3:end] + dfx' * invΓ_ϵ * dfx)
laplaceDist = MvNormal(xs, (tril(Γ_laplace)+ tril(Γ_laplace,-1)'))
x_laplacesamp_177 = rand(laplaceDist, m)
z_laplacesamp_177 = (O[3:end]' * x_laplacesamp_177)' .+ O_offset



density(z_possamp_lowrank_177[2000:1:end], color=:blue, linewidth=2, label="Low Rank",  title="1D Goal Posterior - Marginal Density", xlim=[-0.1,0.1])
density!(z_prsamp[1:1:end], color=:black, linewidth=1, label="Prior")
# density!(z_laplacesamp_" * site_label[site] * "[1:1:end], color=:red, linewidth=1, label="Laplace")
density!(z_fullrank[2500:10:end], color=:red, linewidth=2, label="Naive")
# density!(z_possamp[950000:10:end], color=:red, linewidth=2, label="Naive")#, xlim=[0.1,0.4])
plot!([z_true], seriestype="vline", color=:black, linewidth=3, label="Truth")
# plot!([O*x_map+O_offset], seriestype="vline", color=:red, linewidth=3, label="MAP")
plot!([mean(z_possamp_lowrank_177[2000:1:end])], seriestype="vline", color=:blue, linewidth=1, label="Pos Mean")



# ## MCMC Chain plots
plot(z_fullrank[1:1:end], xlabel="Sample number", ylabel="Z", title="Naive MCMC", label=false)
plot(z_possamp_lowrank_177[1:10000], xlabel="Sample number", ylabel="Z", title="Goal-oriented MCMC", label=false)



# 