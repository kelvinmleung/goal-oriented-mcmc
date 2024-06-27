include("inverseProblem.jl")
include("mcmc.jl")
include("mcmc_1d.jl")

Random.seed!(123)

## Test run_godr_7 on linear toy problem 
## results of this test : 
## Naive MCMC works 
## Certified dimension reduction works if we use 1000 prior samples
using StatsPlots, GaussianMixtures

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

m = 500000
x_prsamp = rand(normDist, m)
z_prsamp = O * x_prsamp
y_prsamp_pred = rand(noiseDist, m)

for i = 1:m
    y_prsamp_pred[:,i] = y_prsamp_pred[:,i] + fwdtoy(x_prsamp[:,i])
end

yz_prsamp = hcat(y_prsamp_pred', (z_prsamp .- O * μ_x)')
nComp = 8

gmm = GMM(nComp, yz_prsamp, method=:kmeans, kind=:full, nInit=100, nIter=100, nFinal=100)
# gmm.hist





## Samples after conditioning ##
yz_weights = weights(gmm)
yz_means = means(gmm)
yz_covs = covars(gmm)
ygz_means = zeros((nComp, n))
ygz_covs = [zeros(n,n) for _ in 1:nComp]
for i in 1:nComp
    ygz_covs[i] = yz_covs[i][1:end-1,1:end-1] - yz_covs[i][1:end-1,end] * yz_covs[i][1:end-1,end]' / yz_covs[i][end,end]
    ygz_means[i,:] = yz_means[i,1:end-1] + yz_covs[i][1:end-1,end] * (-0.2 - yz_means[i,end])  / yz_covs[i][end,end] 
end
mixtureDef = []
for i in 1:nComp
    push!(mixtureDef, (ygz_means[i,:], ygz_covs[i]))
end
yz_weights
gmm_cond = MixtureModel(MvNormal, mixtureDef, yz_weights)

randgmmcond = rand(gmm_cond, 10000)
scatter(randgmmcond[1,:], randgmmcond[2,:], xlims=(-0.5,1.), ylims=(-1.,1.))

randgmmcond = rand(gmm, 10000)
scatter(randgmmcond[:,1], randgmmcond[:,2], xlims=(-0.5,1.), ylims=(-1.,1.))





## Log Likelihood Plot ##
z_likelihood = collect(-0.5:0.001:0.5)
n_likelihood = length(z_likelihood)
likelihood_gmm = zeros(n_likelihood)
likelihood_cdr = zeros(n_likelihood)
likelihood_covexpand = zeros(n_likelihood)
for i = 1:n_likelihood
    likelihood_gmm[i] = gmm_likelihood(gmm, z_likelihood[i], y)
    likelihood_cdr[i] = cdr_likelihood(z_likelihood[i], μ_x, invΓ_ϵ, Q, O, y)
    likelihood_covexpand[i] = covexpand_likelihood(z_likelihood[i], μ_x, Γ_x, Γ_ϵ, Q, O, y)

end
plot(z_likelihood, likelihood_gmm, linewidth=2, label="GMM", xlabel="Z", ylabel="Log Likelihood")
plot!(z_likelihood, likelihood_cdr, linewidth=2, label="CDR")
plot!(z_likelihood, likelihood_covexpand, linewidth=2, label="Expanded Cov")



## MCMC ##
x0 = x_true
@time x_possamp = mcmc_amm_simple(x0, μ_x, Γ_x, Γ_ϵ, y, m)
z_possamp = O * x_possamp

# @time z_possamp_gmm = mcmc_lis_1d(x0, μ_x, Γ_x, Γ_ϵ, Q, O, y; N=Int(m/10))
@time z_possamp_covexpand = mcmc_lis_1d(x0, μ_x, Γ_x, Γ_ϵ, Q, O, y; N=Int(m/10))


density(z_possamp[10000:10:end], color=:black, linewidth=2, label="Naive", title="1D Goal Posterior - Marginal Density")#, xlims=(-0.1,0.06))
density!(z_possamp_gmm[10000:10:end], color=:blue, linewidth=2, label="GMM")
# density!(z_possamp_cdr[10000:10:end], color=:red, linewidth=2, label="CDR")
density!(z_possamp_covexpand[10000:10:end], color=:green, linewidth=2, label="CovExpand")
# density!(z_prsamp[1:10:end], color=:black, linewidth=1, label="Prior")
plot!([z_true], seriestype="vline", color=:black, linewidth=3, label="Truth")

plot(z_possamp[1:50000], xlabel="Sample number", ylabel="Z", title="Naive MCMC", label=false)
plot(z_possamp_lowrank[1:50000], xlabel="Sample number", ylabel="Z", title="Goal-oriented MCMC", label=false)

# npzwrite("z_possamp_cdr.npy",z_possamp_cdr)