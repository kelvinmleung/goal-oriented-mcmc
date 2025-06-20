
using Plots, Distributions, JLD
using SRFTools
using AOE
using LinearAlgebra

data_goal, data_refl, wls_sobol = load("data/data_CliMA/rawdata.jld", "goal", "refl", "wls")
p, N = size(data_goal)
n = size(data_refl, 1)

N_qoi, N = size(data_goal)

λ_ranges = [400.0 1300.0; 1450.0 1780.0; 2051.0 2451.0]
priormodel, wls_326 = get_priormodel(:standard; λ_ranges) # PriorModel instance
rtmodel = AOE.get_radiative_transfer(:modtran; λ_ranges);

prjld = jldopen("./priordata/priors_standard.jld")
means = collect(eachcol(prjld["means"][:,:]'))
covs = [prjld["covs"][:,:,:][i, :, :] for i ∈ 1:length(means)]
pr_λs = load("./LUTdata/wls.jld")["wls"]


# p, n, N = length(keys_goal), length(keys_refl), length(ds["ind"][:])


prInd = 8
prsamp = rand(MvNormal(priormodel[prInd].mean, priormodel[prInd].cov))

idx_sobol = AOE.get_λ_idx(wls_sobol, λ_ranges)
wls_98 = wls_sobol[idx_sobol]
data_refl = data_refl[idx_sobol,:]

# how to  get the prior samples 
plot(wls_98, data_refl[:,1:100], linewidth=1, alpha=0.3, color=:blue, label="")
# plot!(pr_λs, means[prInd] * 3, alpha=1, linewidth=2, color=:red, label="")
plot!(wls_326, priormodel[prInd].mean * 3, alpha=0.5, color=:red, linewidth=2)
vline!(1300:1450, color=:white, label="",alpha=0.3)
vline!(1780:2050, color=:white, label="",alpha=0.3)


wls = wls_sobol
wls_spacing = vcat(wls[2] - wls[1], wls[2:end] - wls[1:end-1], wls[end] - wls[end-1])
σ_125 = FWHM_to_σ.(0.5 * (wls_spacing[1:end-1] + wls_spacing[2:end]))
srf_mat = srf_matrix(pr_λs, SRFSpec(wls, σ_125))[idx_sobol,:]


means_125 = [srf_mat * means[i] for i ∈ 1:length(means)]
covs_125 = [srf_mat * covs[i] * srf_mat' for i ∈ 1:length(means)]

nPr = 8

function ind_of_prior(x, means, nPr)
    
    γs = [(x' * x) / (x' * means[i]) for i in 1:nPr]
    costs = [sum((means[i] - x / γ).^2) for (γ,i) in zip(γs, 1:nPr)]
    # println(γs, costs)
    i0 = argmin(costs)
    # println(costs[i0])
    i0,  γs[i0]#, #costs[i0],
end

ind = 1
ind_of_prior(data_refl[:,ind], means_125, 8)
plot(wls_98, data_refl[:,ind], label="Data")
plot!(wls_98, means_125[8], label="Prior")


list_prior_types = zeros(nPr)
list_ind_pr = [[] for _ = 1:nPr]
list_ind_scaling = zeros(N)

for i in 1:N
    prior_ind, γ = ind_of_prior(data_refl[:,i], means_125, 8)
    list_ind_scaling[i] = γ
    push!(list_ind_pr[prior_ind], i)
    # push!(list_prior_types[i], ind_of_prior(data_refl[:,i], means_125, 8))
end

bar(collect(1:8), list_prior_types, label="")
plot!(xlabel="Surface Type", ylabel="Frequency", title="Frequency of surface types in CliMA data")




## TRAIN ON PRIOR 8 ONLY
indPr = 8
Ntrain = 24000
Ntest = length(list_ind_pr[indPr]) - Ntrain

Xtrain = zeros(Ntrain,length(idx_sobol));
Xtest = zeros(Ntest,length(idx_sobol));
for i in 1:Ntrain
    Xtrain[i,:] = data_refl[:,list_ind_pr[indPr][i]]' / list_ind_scaling[list_ind_pr[indPr][i]] #/ norm(data_refl[:,list_ind_pr[indPr][i]]')
end
Xtrain = hcat(ones(Ntrain, 1), Xtrain)


for (i,j) in enumerate(Ntrain+1 : Ntrain+Ntest)
    Xtest[i,:] = data_refl[:,list_ind_pr[indPr][j]]' / list_ind_scaling[list_ind_pr[indPr][j]]# / norm(data_refl[:,list_ind_pr[indPr][j]]')
end
Xtest = hcat(ones(Ntest, 1), Xtest)

plot(wls_98,Xtrain[1:10,2:end]')
plot(wls_98,Xtest[1:10,2:end]')

Ztrain = data_goal[:, list_ind_pr[indPr][1:Ntrain]]'
Ztest = data_goal[:, list_ind_pr[indPr][Ntrain+1:Ntrain+Ntest]]'
# log transform CHL and LWC
Ztrain[:,3] = log.(Ztrain[:,3])
Ztest[:,3] = log.(Ztest[:,3])

Ztrain[:,10] = log.(Ztrain[:,10])
Ztest[:,10] = log.(Ztest[:,10])

# sqrt transform LMA
Ztrain[:,7] = log.(Ztrain[:,7])
Ztest[:,7] = log.(Ztest[:,7])


plot(wls_98, Xtrain[1:1000,2:end]', color=:red, alpha=0.1, label="")


eps = 1e-1
β = inv(Xtrain' * Xtrain + eps * I) * Xtrain' * Ztrain
linoper = β[2:end,:]
offset = β[1,:]

selectQOI = [2,3,5,7,9,10]
# selectQOI = collect(1:16)
selectQOI = [2,3,7,10]


# Set up a 2x2 grid layout for the 4 subplots
# p = plot(layout=(1, 4), size=(400, 400), dpi=300)

rsq = zeros(length(selectQOI))
for (i, idx) ∈ enumerate(selectQOI)
    # y_true = Ztrain[:,idx]
    # y_pred = Xtrain * β[:,idx]
    y_true = Ztest[:,idx]
    y_pred = Xtest * β[:,idx]
    
    ss_res = sum((y_true - y_pred).^2)  # Residual sum of squares
    ss_tot = sum((y_true .- mean(y_true)).^2)  # Total sum of squares
    
    rsq[i] = 1 - ss_res / ss_tot
    
    # println("R² for $(keys_goal[indQOI]): ", r_squared)
    if idx in [3,7,10]
        scatter(exp.(y_true[1:2:end]),exp.(y_pred[1:2:end]), title=keys_goal[idx],  xlabel="Truth", ylabel="Predicted", legend=false, dpi=300, alpha=0.5, size=(300,300))
        plot!([0, maximum(exp.(y_true))], [0, maximum(exp.(y_true))], color=:red, linewidth=3)

    else
        scatter(y_true[1:2:end],y_pred[1:2:end], title=keys_goal[idx],  xlabel="Truth", ylabel="Predicted", legend=false, dpi=300, alpha=0.5, size=(300,300))
        plot!([0, maximum(y_true)], [0, maximum(y_true)], color=:red, linewidth=3)

    end
    # scatter(y_true[1:2:end],y_pred[1:2:end], title=keys_goal[idx],  xlabel="Truth", ylabel="Predicted", legend=false, dpi=300, alpha=0.5, size=(300,300))
    # plot!([0, maximum(y_true)], [0, maximum(y_true)], color=:red, linewidth=3)
    # plot!(p, scatter, subplot=i)
    savefig("plots/05262025/linreg_qoi_" * keys_goal[idx] * ".pdf")
end
rsq
# savefig(p, "plots/11112024/linreg_4qoi.png")


