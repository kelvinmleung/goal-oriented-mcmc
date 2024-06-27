include("inverseProblem.jl")
include("mcmc_1d.jl")
using StatsPlots, GaussianMixtures, NPZ

Random.seed!(123)


λ_ranges = [400.0 1300.0; 1450.0 1780.0; 2051.0 2451.0]
priormodel, wls = get_priormodel(:standard; λ_ranges) # PriorModel instance
rtmodel = AOE.get_radiative_transfer_modtran(:LUTRT1; λ_ranges);
rdbufs = get_RetrievalData_bufs(nλ) 
xa, xs = AOE.invert(y, rdbufs[1], rtmodel, priormodel)
fx = AOE.fwdfun(xa, xs, rtmodel) 
dfx = AOE.gradfwd_accel(xa, xs, rtmodel, fx)[:,3:end]
x_map = vcat(xa,xs)

m = 10000
O_unscaled = npzread("data_canopy/goal_op_8_unscaled.npy") 
samp_y = zeros((326,m))
samp_z = zeros((1,m))
samp_pr_s = npzread("data_canopy/priorSamples_8_unscaled.npy")
samp_pr_a = rand(MvNormal([0.2,1.3],[0.01 0; 0 1]), m)
# samp_x = vcat(samp_pr_a, samp_pr_s)
for i in 1:m
    samp_y[:,i] = AOE.fwdfun(samp_pr_a[:,i], samp_pr_s[:,i], rtmodel) 
    samp_z[i] = dot(O_unscaled, samp_pr_s[:,i])
end
size(vcat(samp_z,samp_y))
GMM(10, vcat(samp_z,samp_y))
GMM(10, vcat(samp_z,samp_y); method=:em, kind=:diag, nInit=50, nIter=10, nFinal=10)


O = vcat(zeros(2), npzread("data_canopy/goal_op_8_unscaled.npy"))' / npzread("data_canopy/prscale_1_1.npy")
O_offset = npzread("data_canopy/goal_op_const_8_unscaled.npy")
x_true = npzread("data_canopy/s_true.npy")[1,1,:] #x_true atm = 0.19, 1.31
z_true = O[3:end]' * x_true + O_offset
y = npzread("data_canopy/y.npy")[1,1,:]

# Inference parameters
μ_x = vcat([0.2; 1.3], npzread("data_canopy/prmean_1_1.npy"))
Γ_x = zeros((328, 328))
Γ_x[1:2,1:2] = [0.01 0; 0 0.04]
Γ_x[3:end,3:end] = npzread("data_canopy/prcov_1_1.npy")
Γ_ϵ = diagm(y * 1e-4)
μ_z = O * μ_x
Γ_z = O * Γ_x * O'
invΓ_x, invΓ_z, invΓ_ϵ = inv(Γ_x), inv(Γ_z), inv(Γ_ϵ)
Q = Γ_x * O' * invΓ_z
