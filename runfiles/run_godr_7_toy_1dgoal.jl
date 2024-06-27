include("inverseProblem.jl")
include("mcmc.jl")
include("mcmc_1d.jl")

Random.seed!(123)

## DIFFERENCE FROM RUN GODR 6: I am comparing Naive MCMC to the pseudomarginal MCMC (which is the same as what was implemented before, but see notebook notes or Youssef meeting 6/12)

using StatsPlots, NPZ


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
x_true = [-0.06457306721039767; -0.14632513788889215; -0.16236037455860808] #rand(normDist)
z_true = O * x_true
# Apply forward model
eps = sqrt(σ_ϵ²) .* randn(n)
y = fwdtoy(x_true) + eps


m = 100000
x_prsamp = rand(normDist, m)
z_prsamp = O * x_prsamp

x0 = x_true
@time x_possamp = mcmc_amm_simple(x0, μ_x, Γ_x, Γ_ϵ, y, m)
x_pos_mean = mean(x_possamp[:,Int(m/2):end], dims=2)
z_possamp = O * x_possamp

# low rank 1D

@time z_possamp_lowrank = mcmc_lis_1d(x0, μ_x, Γ_x, Γ_ϵ, Q, O, y; N=Int(m/10))


density(z_possamp[1000:10:end], color=:red, linewidth=2, label="Naive", title="1D Goal Posterior - Marginal Density", xlims=((-1,1)))
density!(z_possamp_lowrank[10000:10:end], color=:blue, linewidth=2, label="Low Rank")
density!(z_prsamp[1:10:end], color=:black, linewidth=1, label="Prior")
plot!([z_true], seriestype="vline", color=:black, linewidth=3, label="Truth")


# # @time z_possamp_unified, x_possamp_unified = mcmc_lis_unified(μ_z[1], μ_x, Γ_x, Γ_ϵ, Q, O, y; N=m)
# # z_possamp_unified = O * x_possamp_unified

# # x_perp = zeros((n,m))
# x_all = zeros((n,30000))
# x_all[1,:] = z_possamp_lowrank 
# y_prsamp_pred = zeros((n,m))
# y_possamp_pred = zeros((n,m))
# y_possamp_pred_unified = zeros((n,m))



plot(x_prsamp[1,Int(m/2):1000:end], x_prsamp[2,Int(m/2):1000:end], seriestype=:scatter, mc=:cyan, label="Prior ", title="Samples of X", alpha=0.3)
plot!(x_possamp[1,Int(m/2):1000:end], x_possamp[2,Int(m/2):1000:end], seriestype=:scatter, mc=:black, label="Posterior X")
# plot!(x_all[1,Int(m/2):end], x_all[2,Int(m/2):end], seriestype=:scatter, mc=:black, label="Posterior", title="Samples of Z")
# plot!((Q*vec(z_possamp_lowrank .- O*μ_x)' .+ μ_x )[1,1:10:end], (Q*vec(z_possamp_lowrank .- O*μ_x)' .+ μ_x)[2,1:10:end], seriestype=:scatter, mc=:blue, label="Posterior QZ Low Rank")
plot!([x_true[1]], [x_true[2]], seriestype=:scatter, mc=:red, label="Truth")




# # ## MCMC Chain plots
# plot(z_possamp[1:1:end], xlabel="Sample number", ylabel="Z", title="Goal-oriented MCMC", label=false)
# plot(z_possamp_lowrank[1:end], xlabel="Sample number", ylabel="Z", title="Goal-oriented MCMC", label=false)

# npzwrite("data_pseudomarg_tests/z_pos_cdr_1000prsamp_initTrue.npy", z_possamp_lowrank)
# npzwrite("data_pseudomarg_tests/z_pos_naive_initTrue.npy", z_possamp)



density(npzread("data_pseudomarg_tests/z_pos_naive_initTrue.npy")[100000:10:end], color=:red, linewidth=2, label="Naive", title="1D Goal Posterior - Marginal Density")#, xlims=(-0.15,0.03))
# density(z_possamp[100000:100:end], color=:red, linewidth=2, label="Naive", title="1D Goal Posterior - Marginal Density")#, xlims=(-0.15,0.03))

density!(npzread("data_pseudomarg_tests/z_pos_cdr_10prsamp_initTrue.npy"), color=:blue, linewidth=2, label="CDR – 10 PrSamp", linestyle=:solid)
density!(npzread("data_pseudomarg_tests/z_pos_cdr_100prsamp_initTrue.npy"), color=:blue, linewidth=2, label="CDR – 100 PrSamp", linestyle=:dash)
density!(npzread("data_pseudomarg_tests/z_pos_cdr_1000prsamp_initTrue.npy"), color=:blue, linewidth=2, label="CDR – 1000 PrSamp", linestyle=:dashdot)
display(plot!([z_true], seriestype="vline", color=:black, linewidth=3, label="Truth"))