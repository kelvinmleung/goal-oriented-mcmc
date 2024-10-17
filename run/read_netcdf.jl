using NCDatasets
using Plots, Distributions, JLD
using SRFTools
using AOE
using LinearAlgebra

ds = NCDataset("data/data_CliMA/JPL_SBG_sobol_example_v1.nc")


keys_goal = []
keys_refl = []
wls_sobol = Float64[]
for (key, value) in ds
    if key[1] ∈ "ABCDEFGHIJKLMNOPQRSTUVWXYZ" && key != "ind"
        push!(keys_goal, key)
    elseif key[1] ∉ "ABCDEFGHIJKLMNOPQRSTUVWXYZ" && key != "ind"
        push!(keys_refl, key)
        push!(wls_sobol, parse(Float64, strip(key, ['_', 'n', 'm'])))
    end
end

p, n, N = length(keys_goal), length(keys_refl), length(ds["ind"][:])


data_goal = zeros(p, N)
data_refl = zeros(n, N)
for i in 1:p
    data_goal[i,:] = ds[keys_goal[i]][:]
end
for i in 1:n
    data_refl[i,:] = ds[keys_refl[i]][:]
end
save("data/data_CliMA/rawdata.jld", "goal", data_goal, "refl", data_refl, "wls", wls_sobol)

Ntrain = 50000
Xtrain = hcat(ones(Ntrain, 1), data_refl[:, 1:Ntrain]')
Ztrain = data_goal[:, 1:Ntrain]'

β = inv(Xtrain' * Xtrain) * Xtrain' * Ztrain
linoper = β[2:end,:]
offset = β[1,:]


rsq = zeros(p)
for i in 1:16
    y_true = Ztrain[:,i]
    y_pred = Xtrain * β[:,i]
    
    ss_res = sum((y_true - y_pred).^2)  # Residual sum of squares
    ss_tot = sum((y_true .- mean(y_true)).^2)  # Total sum of squares
    
    rsq[i] = 1 - ss_res / ss_tot

    plot(title=keys_goal[i],  xlabel="Truth", ylabel="Predicted", dpi=300)
    display(plot!(y_true[1:1000],y_pred[1:1000], seriestype=:scatter, legend=false))
    # savefig("data/data_CliMA/regression_rawdata/$(keys_goal[indQOI]).png")
end
rsq



selectQOI = [1,2,3,5,7,8,9,10,11]
selectQOI = [3,5,7,9,10]
λ_ranges = [400.0 1300.0; 1450.0 1780.0; 2051.0 2451.0]
idx_sobol = AOE.get_λ_idx(wls_sobol, λ_ranges) #### ADD THIS INTO THE DATA GOAL SLICING
data_goal = zeros(length(selectQOI), N)
data_refl = zeros(length(idx_sobol), N)
for (i,idx) ∈ enumerate(selectQOI)
    data_goal[i,:] = ds[keys_goal[idx]][:]
end
for (i,idx) ∈ enumerate(idx_sobol)
    data_refl[i,:] = ds[keys_refl[idx]][:]
end
save("data/data_CliMA/relevantdata.jld", "goal", data_goal, "refl", data_refl, "wls", wls_sobol)

data_goal, data_refl = load("data/data_CliMA/relevantdata.jld", "goal", "refl")

Ntrain = 40000

Xtrain = zeros(Ntrain,length(idx_sobol));
Xtest = zeros(N - Ntrain,length(idx_sobol));
for i in 1:Ntrain
    Xtrain[i,:] = data_refl[:,i]'/ norm(data_refl[:,i]')
end
Xtrain = hcat(ones(Ntrain, 1), Xtrain)

for (i,j) in enumerate(Ntrain+1:N)
    Xtest[i,:] = data_refl[:,j]'/ norm(data_refl[:,j]')
end
Xtest = hcat(ones(N - Ntrain, 1), Xtest)


Ztrain = data_goal[:, 1:Ntrain]'
Ztest = data_goal[:, Ntrain+1:N]'

eps = 1e-6
β = inv(Xtrain' * Xtrain + eps * I) * Xtrain' * Ztrain
linoper = β[2:end,:]
offset = β[1,:]


# ind_outliers = []

rsq = zeros(length(selectQOI))
for (i, idx) ∈ enumerate(selectQOI)
    # y_true = Ztrain[:,i]
    # y_pred = Xtrain * β[:,i]
    y_true = Ztest[:,i]
    y_pred = Xtest * β[:,i]
    
    ss_res = sum((y_true - y_pred).^2)  # Residual sum of squares
    ss_tot = sum((y_true .- mean(y_true)).^2)  # Total sum of squares
    
    rsq[i] = 1 - ss_res / ss_tot
    
    # println("R² for $(keys_goal[indQOI]): ", r_squared)
    plot(title=keys_goal[idx],  xlabel="Truth", ylabel="Predicted", dpi=300)
    # if idx == 10
    #     for j ∈ 1:N-Ntrain
    #         if (y_true[j] < 5. && y_pred[j] > 7.) || y_pred[j] < 0. || y_pred[j] > 20. || (y_true[j] >16. && y_pred[j] < 14.)
    #             push!(ind_outliers, j)
    #             plot!([y_true[j]], [y_pred[j]], seriestype=:scatter, color=:red)
    #         end
    #     end
    # end
    maxVal = maximum(y_true) 
    minVal = minimum(y_true)
    # for j ∈ 1:N-Ntrain
    #     if abs(y_pred[j] - y_true[j]) > 0.2 * (maxVal - minVal)
    #         push!(ind_outliers, j)
    #         plot!([y_true[j]], [y_pred[j]], seriestype=:scatter, color=:red)
    #     end
    # end
    plot!(y_true[ind_outliers], y_pred[ind_outliers], seriestype=:scatter, color=:red,legend=false)
    plot!(y_true[1:10:end],y_pred[1:10:end], seriestype=:scatter, color=:blue, legend=false)

    display(plot!())

    # savefig("data/data_CliMA/regression_relevantdata/$(keys_goal[idx]).png")
end


rsq


# ## NEURAL NETWORK

# using Flux
# using Flux.Data: DataLoader

# X, Y = Xtrain', Ztrain[:,2]'

# data = DataLoader((X,Y); batchsize=2, shuffle=true)

# model = Chain(
#     Dense(99, 10, relu),   # Input layer: 2 features, 10 nodes, σ activation function
#     Dense(10, 1)       # Output layer: 1 node (for regression tasks)
# )

# # Defining a loss function and an optimizer
# loss(model, x, y) = mean(abs2.(model(x) .- y));
# opt = Descent()

# for epoch in 1:5
#     @info "Epoch $epoch and loss $(loss(model, X,Y))"
#     Flux.train!(loss, model, data, opt)
# end


# # Make predictions
# X_new = Xtest'
# prediction =model(X_new)
# for (i, idx) ∈ enumerate(selectQOI)
#     y_true = Ztest[:,i]
#     y_pred = prediction[i,:]
    
#     ss_res = sum((y_true - y_pred).^2)  # Residual sum of squares
#     ss_tot = sum((y_true .- mean(y_true)).^2)  # Total sum of squares
    
#     rsq[i] = 1 - ss_res / ss_tot
#     plot(title=keys_goal[idx],  xlabel="Truth", ylabel="Predicted", dpi=300)
#     display(plot!(y_true[1:1000],y_pred[1:1000], seriestype=:scatter, legend=false))
#     # savefig("data/data_CliMA/regression_relevantdata/$(keys_goal[idx]).png")
# end
# rsq

# plot(title=keys_goal[selectQOI[2]],  xlabel="Truth", ylabel="Predicted", dpi=300)
# display(plot!(Ztest[:,2], prediction', seriestype=:scatter, legend=false))






# PLOTS FOR VERIFICATION

λ_ranges = [400.0 1300.0; 1450.0 1780.0; 2051.0 2451.0]
priormodel, wls_326 = get_priormodel(:standard; λ_ranges) # PriorModel instance
rtmodel = AOE.get_radiative_transfer(:modtran; λ_ranges);

prjld = jldopen("./priordata/priors_standard.jld")
means = collect(eachcol(prjld["means"][:,:]'))
covs = [prjld["covs"][:,:,:][i, :, :] for i ∈ 1:length(means)]
pr_λs = load("./LUTdata/wls.jld")["wls"]

prInd = 8
prsamp = rand(MvNormal(priormodel[prInd].mean, priormodel[prInd].cov))

wls_98 = wls_sobol[idx_sobol]
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


plot(wls_98, data_refl[:,1:100], linewidth=1, alpha=0.4, color=:cyan, label="")
plot!(wls_98, srf_mat * means[prInd] * 3, alpha=1, linewidth=2, color=:red, label="")
# plot!(pr_λs, means[prInd] * 3, alpha=1, linewidth=2, color=:black, label="")
vline!(1300:1450, color=:white, label="",alpha=0.3)
vline!(1780:2050, color=:white, label="",alpha=0.3)


idx_326 = AOE.get_λ_idx(pr_λs, λ_ranges) #### ADD THIS INTO THE DATA GOAL SLICING

save("data/data_CliMA/goaloperator.jld","oper",linoper, "offset",offset, "srfmat", srf_matrix(pr_λs, SRFSpec(wls, σ_125))[idx_sobol,idx_326])