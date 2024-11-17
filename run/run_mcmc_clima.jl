include("../src/forward.jl")
include("../src/goalorientedtransport.jl")
include("../src/mcmc_aoe.jl")

Random.seed!(123)

λ_ranges = [400.0 1300.0; 1450.0 1780.0; 2051.0 2451.0]
priormodel, wls = get_priormodel(:standard; λ_ranges) # PriorModel instance
rtmodel = AOE.get_radiative_transfer(:modtran; λ_ranges);
n, p = 326, 4
m = 100000

setup = initialize_GODRdata(n, p)
GODRdata_pr8!(setup; n=n, p=p);
prsamp = gen_pr_samp(setup, m; n=n, p=p);


setup.x_true .= load("data/data_CliMA/truth_obs_pairs.jld", "x")
setup.y .= load("data/data_CliMA/truth_obs_pairs.jld", "y")
setup.z_true .= load("data/data_CliMA/truth_obs_pairs.jld", "z")



# naive
m_naive = 3000000
@time x_possamp = mcmc_bm_3block(setup.μ_x, setup.Γ_x, setup.Γ_ϵ, setup.y, m_naive)
z_possamp_naive = setup.O * x_possamp .+ setup.O_offset
# npzwrite("data/data_clima/z_naive_mcmc.npy", z_possamp_naive)
# z_possamp_naive = npzread("data/data_canopy/aug2/10pix_ind("*string(indx)*","*string(indy)*")/z_naive.npy")


density(z_possamp_naive[1,1:10:end], color=:black, linewidth=2, label="Naive MCMC")#, xlim=[0.1,0.4])
display(plot!([setup.z_true[1]], seriestype="vline", color=:black, linewidth=3, label="Truth"))


