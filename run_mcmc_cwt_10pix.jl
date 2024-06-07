include("inverseProblem.jl")
include("mcmc_1d.jl")
using StatsPlots, NPZ

Random.seed!(123)




n, p = 326, 1
m = 20000

prmean = npzread("data_refl/10pix/prmean_10pix.npy")
prcov = npzread("data_refl/10pix/prcov_10pix.npy")
prclass = npzread("data_refl/10pix/prclass_10pix.npy")
prscale = npzread("data_refl/10pix/prscale_10pix.npy")
s_all = npzread("data_refl/10pix/s_10pix.npy")
y_all = npzread("data_refl/10pix/y_10pix.npy")



for i in 1:10
    for j in 1:10

        display("Pixel (" * string(i) * ", " * string(j) * ")")
        filesuffix = "_" * string(i) * "_" * string(j) * ".npy"
        prcomp = Int(prclass[i,j])
        O = vcat(zeros(2), npzread("data_canopy/goal_op_" * string(prcomp) * "_unscaled.npy"))' / prscale[i,j]
        O_offset = npzread("data_canopy/goal_op_const_" * string(prcomp) * "_unscaled.npy")
        x_true = s_all[i,j,:] #x_true atm = 0.19, 1.31
        z_true = O[3:end]' * x_true + O_offset
        y = y_all[i,j,:]

        # Inference parameters
        μ_x = vcat([0.2; 1.3], prmean[i,j,:])
        Γ_x = zeros((328, 328))
        Γ_x[1:2,1:2] = [0.01 0; 0 0.04]
        Γ_x[3:end,3:end] = prcov[i,j,:,:]
        Γ_ϵ = diagm(y * 1e-4)
        μ_z = O * μ_x
        Γ_z = O * Γ_x * O'
        invΓ_x, invΓ_z, invΓ_ϵ = inv(Γ_x), inv(Γ_z), inv(Γ_ϵ)
        Q = Γ_x * O' * invΓ_z

        normDist = MvNormal(μ_x, Γ_x)
        x_prsamp = rand(normDist, m)
        z_prsamp = (O * x_prsamp)' .+ O_offset

        @time z_possamp_lowrank = mcmc_lis_1d(μ_z[1], μ_x, Γ_x, Γ_ϵ, Q, O, y; N=m) .+ O_offset

        npzwrite("data_out_10pix/z_pos" * filesuffix, z_possamp_lowrank)

    end
end


# ## SAVE THIS TO NPY!!!!!!!
# npzwrite("data_canopy/z_chain_1_1_jun29.npy", z_possamp_lowrank)



density(z_possamp_lowrank[2500:10:end], color=:blue, linewidth=2, label="Low Rank",  title="1D Goal Posterior - Marginal Density")#, xlim=[0.1,0.4])
density!(z_prsamp[2500:10:end], color=:black, linewidth=1, label="Prior")
# plot!([mean(z_possamp[1:100:end])], seriestype="vline", color=:red3, linewidth=3, label=false)
# plot!([mean(z_possamp_lowrank[1:100:end])], seriestype="vline", color=:blue3, linewidth=3, label=false)
# plot!([mean(z_prsamp[1:100:end])], seriestype="vline", color=:black, linewidth=2, label=false)
plot!([z_true], seriestype="vline", color=:black, linewidth=3, label="Truth")


# ## MCMC Chain plots
plot(z_possamp, xlabel="Sample number", ylabel="Z", title="Naive MCMC", label=false)
plot(z_possamp_lowrank[1:10000], xlabel="Sample number", ylabel="Z", title="Goal-oriented MCMC", label=false)


