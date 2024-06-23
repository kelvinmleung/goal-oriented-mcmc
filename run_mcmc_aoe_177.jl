

include("mcmc_aoe.jl")
using StatsPlots, NPZ

Random.seed!(123)

n, p = 326, 1

λ_ranges = [400.0 1300.0; 1450.0 1780.0; 2051.0 2451.0]
priormodel, wls = get_priormodel(:standard; λ_ranges) # PriorModel instance
rtmodel = AOE.get_radiative_transfer_modtran(:LUTRT1; λ_ranges);
λ_idx = AOE.get_λ_idx(wls, λ_ranges)



site_label = ["177", "306", "mars", "dark"]

ind = 1
y = npzread("/Users/kmleung/Documents/Github/transport-retrieval/data/y_177.npy")
# y = npzread("/Users/kmleung/Documents/JPLspatial/AOE.jl/examples/transport/ensembles_a=[0.2,1.5]/yobs_sim_" * site_label[ind] * ".npy")[:,1]

x_true = npzread("/Users/kmleung/Documents/Github/transport-retrieval/data/x_" * site_label[ind] * ".npy")
rdbufs = get_RetrievalData_bufs(326) 

xa, xs = AOE.invert(y, rdbufs[1], rtmodel, priormodel)


mu_pr = npzread("/Users/kmleung/Documents/JPLspatial/AOE.jl/examples/transport/data/" * site_label[ind] * "/mu_pr_" * site_label[ind] * ".npy")
cov_pr = npzread("/Users/kmleung/Documents/JPLspatial/AOE.jl/examples/transport/data/" * site_label[ind] * "/cov_pr_" * site_label[ind] * ".npy")


# Inference parameters
μ_x = vcat([0.2; 1.5], mu_pr[λ_idx])
Γ_x = zeros((328, 328))
Γ_x[1:2,1:2] = [0.04 0; 0 1.]
Γ_x[3:end,3:end] = cov_pr[λ_idx, λ_idx]
# diagΓ_ϵ = (y /50).^2 
# diagΓ_ϵ[diagΓ_ϵ .< 1e-5] .= 1e-5
# Γ_ϵ = diagm(diagΓ_ϵ)

Γ_ϵ = diagm(AOE.dummy_noisemodel(y))

m = 1000000


@time x_possamp = mcmc_bm(μ_x, Γ_x, Γ_ϵ, y, m)



fx = AOE.fwdfun(xa, xs, rtmodel) 
dfx = AOE.gradfwd_accel(xa, xs, rtmodel, fx)[:,3:end]
heatmap(inv(inv(Γ_x[3:end,3:end]) + dfx' * inv(Γ_ϵ) * dfx))
plot(diag(inv(inv(Γ_x[3:end,3:end]) + dfx' * inv(Γ_ϵ) * dfx)))


mcmcmean = mean(x_possamp[3:end,Int(m/2):100:end], dims=2)
# posvar = mean(x_possamp[3:end,Int(m/2):end], dims=2)
plot(wls, x_true, label="Truth", title="Retrieval Comparison")
plot!(wls, xs, label="Isofit")
plot!(wls, mcmcmean, label="MCMC")


scatter(x_possamp[1,Int(m/2):100:end], x_possamp[2,Int(m/2):100:end], alpha=0.2)

# ## MCMC Chain plots
plot(x_possamp[1,1:100:m], xlabel="Sample number", ylabel="AOD", title="MCMC Chain", label=false)
plot(x_possamp[2,1:100:m], xlabel="Sample number", ylabel="H2O", title="MCMC Chain", label=false)
# histogram(x_possamp[1,100000:100:m], xlabel="Sample number", ylabel="AOD", title="MCMC Chain", label=false)

npzwrite("data_refl/mcmc_results_fullrank/mcmc_simobs_" * site_label[ind] * ".npy", x_possamp[:,Int(m/2):10:end])


