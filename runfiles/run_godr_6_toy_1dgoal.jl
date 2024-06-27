include("inverseProblem.jl")
include("mcmc_1d.jl")

Random.seed!(123)

## DIFFERENCE FROM RUN GODR 5: here I'm implementing "Unified" MCMC to toy problem

using StatsPlots


O = [1 1 0]
p, n = size(O)
# Goal oriented 

# Inference parameters
μ_x = 0.05*ones(n)#zeros(n) #
μ_z = O * μ_x
σ_X², σ_Z² = 0.01,0.01
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


m = 300000
x_prsamp = rand(normDist, m)
z_prsamp = O * x_prsamp


@time x_possamp = mcmc_amm_simple(μ_x, μ_x, Γ_x, Γ_ϵ, y, m+20000)[:,20001:end]
x_pos_mean = mean(x_possamp[:,Int(m/2):end], dims=2)
z_possamp = O * x_possamp

# low rank 1D
x0 = [0;0;0]
@time z_possamp_lowrank = mcmc_lis_1d(x0, μ_x, Γ_x, Γ_ϵ, Q, O, y; N=Int(m/10)) 


density(z_possamp[10000:10:end], color=:red, linewidth=2, label="Naive", title="1D Goal Posterior - Marginal Density")
density!(z_possamp_lowrank[10000:10:end], color=:blue, linewidth=2, label="Low Rank")
# density!(z_possamp_unified[10000:100:end], color=:green, linewidth=2, label="Unified")
density!(z_prsamp[1:10:end], color=:black, linewidth=1, label="Prior")
plot!([z_true], seriestype="vline", color=:black, linewidth=3, label="Truth")


# @time z_possamp_unified, x_possamp_unified = mcmc_lis_unified(μ_z[1], μ_x, Γ_x, Γ_ϵ, Q, O, y; N=m)
# z_possamp_unified = O * x_possamp_unified

# x_perp = zeros((n,m))
x_all = zeros((n,m))
x_all[1,:] = z_possamp_lowrank 
y_prsamp_pred = zeros((n,m))
y_possamp_pred = zeros((n,m))
y_possamp_pred_unified = zeros((n,m))

for i = 1:m
    x_all[:,i] = x_all[:,i] + dfwdtoy(Q*z_possamp_lowrank[i]) * (I - Q*O) * x_prsamp[:,i]
    y_prsamp_pred[:,i] = fwdtoy(x_prsamp[:,i])
    y_possamp_pred[:,i] = fwdtoy(x_all[:,i])
    y_possamp_pred_unified[:,i] = fwdtoy(x_possamp_unified[:,i])
end




plot(x_prsamp[1,Int(m/2):10:end], x_prsamp[2,Int(m/2):10:end], seriestype=:scatter, mc=:cyan, label="Prior ", title="Samples of X", alpha=0.3)
plot!(x_possamp_unified[1,1:10:end], x_possamp_unified[2,1:10:end], seriestype=:scatter, mc=:green, label="Posterior QZ Unified", alpha=0.5)
plot!(x_possamp[1,Int(m/2):10:end], x_possamp[2,Int(m/2):10:end], seriestype=:scatter, mc=:black, label="Posterior X")
# plot!(x_all[1,Int(m/2):end], x_all[2,Int(m/2):end], seriestype=:scatter, mc=:black, label="Posterior", title="Samples of Z")
plot!((Q*vec(z_possamp_lowrank)')[1,1:10:end], (Q*vec(z_possamp_lowrank)')[2,1:10:end], seriestype=:scatter, mc=:blue, label="Posterior QZ Low Rank")
plot!([x_true[1]], [x_true[2]], seriestype=:scatter, mc=:red, label="Truth")

plot(y_prsamp_pred[1,Int(m/2):10:end], y_prsamp_pred[2,Int(m/2):10:end], seriestype=:scatter, mc=:cyan, label="Prior", title="Samples of Y")
plot!(y_possamp_pred_unified[1,Int(m/2):10:end], y_possamp_pred_unified[2,Int(m/2):10:end], seriestype=:scatter, mc=:green, label="Posterior - Unified")
plot!(y_possamp_pred[1,Int(m/2):10:end], y_possamp_pred[2,Int(m/2):10:end], seriestype=:scatter, mc=:black, label="Posterior")
plot!([fwdtoy(x_true)[1]], [fwdtoy(x_true)[2]], seriestype=:scatter, mc=:red, label="Truth")
plot!([y[1]], [y[2]], seriestype=:scatter, mc=:orange, label="Observation")



# ## MCMC Chain plots
plot(z_possamp[1:end], xlabel="Sample number", ylabel="Z", title="Goal-oriented MCMC", label=false)
plot(z_possamp_lowrank[1:end], xlabel="Sample number", ylabel="Z", title="Goal-oriented MCMC", label=false)