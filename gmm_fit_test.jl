include("inverseProblem.jl")
include("mcmc.jl")
include("mcmc_1d.jl")

Random.seed!(123)

## Test run_godr_7 on linear toy problem 
## results of this test : 
## Naive MCMC works 
## Certified dimension reduction works if we use 1000 prior samples
using StatsPlots, GaussianMixtures


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
noiseDist = MvNormal(zeros(n), Γ_ϵ)
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



m = 100000
x_prsamp = rand(normDist, m)
z_prsamp = O * x_prsamp
y_prsamp_pred = rand(noiseDist, m)

for i = 1:m
    y_prsamp_pred[:,i] = y_prsamp_pred[:,i] + fwdtoy(x_prsamp[:,i])
end

yz_prsamp = hcat(y_prsamp_pred', (z_prsamp .- O * μ_x)')
nComp = 2

gmm = GMM(nComp, yz_prsamp, method=:kmeans, kind=:full, nInit=50, nIter=10, nFinal=10)

gmm_samps = rand(gmm, m)






function gmm_likelihood(gmm, z_input, y)
    yz_weights = weights(gmm)
    yz_means = means(gmm)
    yz_covs = covars(gmm)

    ygz_means = zeros((nComp, n))
    ygz_covs = [zeros(n,n) for _ in 1:nComp]
    zeros((nComp, n, n))

    for i in 1:nComp
        ygz_covs[i] = yz_covs[i][1:2,1:2] -  yz_covs[i][1:2,3] * yz_covs[i][1:2,3]' / yz_covs[i][3,3]
        ygz_means[i,:] = yz_means[i,1:2] + yz_covs[i][1:2,3] / yz_covs[i][3,3] * (z_input - yz_means[i,3])
    end


    mixtureDef = []
    for i in 1:nComp
        push!(mixtureDef, (ygz_means[i,:], ygz_covs[i]))
    end

    gmm_cond = MixtureModel(MvNormal, mixtureDef, yz_weights)
    logpdf(gmm_cond, y)

end

# z_input = 0.5 #z_prsamp[1,1]


# check the actual logpdf for the true model




y_samp_cond = rand(gmm_cond, m)




# density(yz_prsamp[:,3], color=:red, linewidth=2, label="Training Samples", title="Z")
# density!(gmm_samps[:,3], color=:blue, linewidth=2, label="GMM Samples")

# scatter(yz_prsamp[:,1],yz_prsamp[:,2], color=:black, linewidth=2, label="Joint", xlabel="y1", ylabel="y2")
# scatter!(y_samp_cond[1,:], y_samp_cond[2,:], color=:red, linewidth=2, label="Cond Z=0")


x0 = x_true
@time x_possamp = mcmc_amm_simple(x0, μ_x, Γ_x, Γ_ϵ, y, m)
x_pos_mean = mean(x_possamp[:,Int(m/2):end], dims=2)
z_possamp = O * x_possamp

@time z_possamp_lowrank = mcmc_lis_1d(x0, μ_x, Γ_x, Γ_ϵ, Q, O, y; N=m)





density(z_possamp[10000:10:end], color=:red, linewidth=2, label="Naive", title="1D Goal Posterior - Marginal Density")
density!(z_possamp_lowrank[1000:10:end], color=:blue, linewidth=2, label="Low Rank")
# density!(z_possamp_true, color=:green, linewidth=2, label="True Posterior")
# density!(z_prsamp[1:10:end], color=:black, linewidth=1, label="Prior")
plot!([z_true], seriestype="vline", color=:black, linewidth=3, label="Truth")
