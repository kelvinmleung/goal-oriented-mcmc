
using Distributions
using Plots
using LinearAlgebra
using NPZ
using MCMCDiagnosticTools

# Define the mean and covariance
μ = [1.0, 1.0, 1.0]

# Skewed (non-symmetric) covariance matrix
Σ = [1.0 0.8 0.3;
     0.8 1.5 0.5;
     0.3 0.5 1.2]


# Sample points from the distribution
N = 10000
samples = rand(MvNormal(μ, Σ), N)

# Separate x, y, z coordinates
x, y, z = eachrow(samples)

# 3D scatter plot
scatter3d(x, y, z, markersize=1, alpha=0.2, legend=false)


# Define the mean and covariance
μ = [1.0, 1.0]

# Skewed (non-symmetric) covariance matrix
Σ = [1.0 0.8;
     0.8 1.5]

# Sample points from the distribution
N = 10000
samples = rand(MvNormal(μ, Σ), N)

x, y = eachrow(samples)

# 3D scatter plot
scatter(x, y, markersize=1, alpha=0.2, legend=false, size=(400,300))




    
function get_plot_data(runName)

    # ensDir = "/Users/kmleung/Documents/JPLspatial/AOE.jl/examples/transport/ensembles/" * runName * ".npy"
    mcmcChainDir = "/Users/kmleung/Documents/JPLproject/resultsGibbs/"  * runName * "_SNR50_RandWalkIsofitCovEps0_11_2M/mcmcchain.npy"
    # xtruthDir = "/Users/kmleung/Documents/Github/transport-retrieval/data/x_" * runName * ".npy"
    # xisofitMuDir = "/Users/kmleung/Documents/Github/transport-retrieval/data/x_iso_" * runName * ".npy"
    # xisofitGammaDir = "/Users/kmleung/Documents/Github/transport-retrieval/data/x_iso_gamma_" * runName * ".npy"
    # samp_pr = npzread(ensDir)
    samp_tran = npzread("/Users/kmleung/Documents/MIT Year 4/Spring 2023/MapTests_Max/transportResults/sampTransport_" * runName * ".npy")

    samp_mcmc = npzread(mcmcChainDir)[:,1:20:end]
    return samp_tran, samp_mcmc
end


# runNames = ["177","306","mars","dark"]
# year = "2014"

# bands = npzread("/Users/kmleung/Documents/Github/transport-retrieval/data/wl_ind_" * year * ".npy")
# wls = npzread("/Users/kmleung/Documents/Github/transport-retrieval/data/wls.npy")


# samp_tran, samp_mcmc = get_plot_data("306")
# mean_tran = mean(samp_tran, dims=1)[1,3:end]
# mean_mcmc = mean(samp_mcmc, dims=2)[bands,1]
# relerr = abs.(mean_tran .- mean_mcmc) ./ mean_mcmc
# plot(wls, relerr, label=false, size=(400,150), linewidth=2, color=:black, dpi=300, grid=false)
# savefig("plots/08042025_fordefense/306_relerr.png")
# plot(wls, mean_mcmc, label="MCMC", size=(400,150), linewidth=2, color=:blue4 , dpi=300, grid=false)
# plot!(wls, mean_tran, label="Transport", size=(400,150), linewidth=2, color=:red, dpi=300, grid=false)
# plot!(legend=:bottom)
# savefig("plots/08042025_fordefense/306_mean.png")


# samp_tran, samp_mcmc = get_plot_data("mars")
# mean_tran = mean(samp_tran, dims=1)[1,3:end]
# mean_mcmc = mean(samp_mcmc, dims=2)[bands,1]
# relerr = abs.(mean_tran .- mean_mcmc) ./ mean_mcmc
# plot(wls, relerr, label=false, size=(400,150), linewidth=2, color=:black, dpi=300, grid=false)
# savefig("plots/08042025_fordefense/mars_relerr.png")
# plot(wls, mean_mcmc, label="MCMC", size=(400,150), linewidth=2, color=:blue4, dpi=300, grid=false)
# plot!(wls, mean_tran, label="Transport", size=(400,150), linewidth=2, color=:red, dpi=300, grid=false)
# plot!(legend=:bottom)
# savefig("plots/08042025_fordefense/mars_mean.png")




### PLOTS FOR THESIS ###

runNames = ["177","306","mars","dark"]
year = "2014"

bands = npzread("/Users/kmleung/Documents/Github/transport-retrieval/data/wl_ind_" * year * ".npy")
wls = npzread("/Users/kmleung/Documents/Github/transport-retrieval/data/wls.npy")



# Create individual subplots
titles = ["Building 177", "Building 306", "Mars Yard", "Parking Lot"]


plots = []
for i in 1:4
     samp_tran, samp_mcmc = get_plot_data(runNames[i])
     p = plot(wls, mean(samp_mcmc, dims=2)[bands,1], label="MCMC", color = :black, linewidth = 2)
     plot!(wls, mean(samp_tran, dims=1)[1,3:end], label="Transport", color = :red, linewidth=2)
     # plot!(xlabel = (i == 4 ? "Wavelength [nm]" : ""),  ylabel = (i == 2 ? "Reflectance" : ""), title = titles[i], legend = (i == 1), grid = false)
     plot!(title = titles[i], grid = false, legend = (i == 1),)
     plot!(ylims=(0,maximum(vcat(mean(samp_mcmc, dims=2)[bands,1], mean(samp_tran, dims=1)[1,3:end]))))
     push!(plots, p)
end

# Combine into one 4x1 plot
plot(plots..., layout = (4, 1), size = (500, 700), dpi=300)
savefig("plots/08142025_forthesis/meanrefl.png")

plots = []
for i in 1:4
     samp_tran, samp_mcmc = get_plot_data(runNames[i])
     p = plot(wls, sqrt.(diag(cov(samp_mcmc[bands,:],dims=2))), label="MCMC", color = :black, linewidth = 2)
     plot!(wls, sqrt.(diag(cov(samp_tran[:,3:end],dims=1))), label="Transport", color = :red, linewidth=2)
     # plot!(xlabel = (i == 4 ? "Wavelength [nm]" : ""),  ylabel = (i == 2 ? "Reflectance" : ""), title = titles[i], legend = (i == 1), grid = false)
     plot!(title = titles[i], grid = false, legend = (i == 1),)
     plot!(ylims=(0,maximum(sqrt.(diag(cov(samp_tran[:,3:end],dims=1))))))
     push!(plots, p)
end

# Combine into one 4x1 plot
plot(plots..., layout = (4, 1), size = (500, 700), dpi=300)
savefig("plots/08142025_forthesis/meanstddev.png")





MCMCDiagnosticTools.ess(npzread("data/data_clima_may2025/z_naive_mcmc.npy")[4,:])

MCMCDiagnosticTools.rhat(npzread("data/data_clima_may2025/z_naive_mcmc.npy")[4,:])