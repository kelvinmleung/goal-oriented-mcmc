include("inverseProblem.jl")
include("mcmc.jl")
include("mcmc_1d.jl")

Random.seed!(123)
using StatsPlots, NPZ

### Meeting with Mathieu

n, p = 326, 1

λ_ranges = [400.0 1300.0; 1450.0 1780.0; 2051.0 2451.0]
priormodel, wls = get_priormodel(:standard; λ_ranges) # PriorModel instance
rtmodel = AOE.get_radiative_transfer_modtran(:LUTRT1; λ_ranges);
λ_idx = AOE.get_λ_idx(wls, λ_ranges)


site_label = ["177", "306", "mars", "dark"]

ind = 1
y = npzread("/Users/kmleung/Documents/Github/transport-retrieval/data/y_177.npy")
x_true = npzread("/Users/kmleung/Documents/Github/transport-retrieval/data/x_177.npy")
rdbufs = get_RetrievalData_bufs(n) 

xa, xs = AOE.invert(y, rdbufs[1], rtmodel, priormodel)

invΓ_ϵ = diagm(1 ./AOE.dummy_noisemodel(y))

plot(AOE.dummy_noisemodel(y))

# ForwardDiff.jacobian(aoe_fwdfun, vcat([0.1,1.5], x_true))


x_concat = vcat([0.1,1.5], x_true)
fx = aoe_fwdfun(x_concat)

finitediff = FiniteDiff.finite_difference_jacobian(aoe_fwdfun, x_concat)
analyticgrad = aoe_gradfwdfun(x_concat, fx)
finitediff - analyticgrad ### CHECK THIS



# finite difference



minimum(AOE.dummy_noisemodel(y))
###AUTODIFF CHECK
### using Test
# create test folder
# using ForwardDiff



O = vcat(zeros(2), npzread("data_canopy/goal_op_8_unscaled.npy"))' / npzread("data_canopy/prscale_1_1.npy")
O_offset = npzread("data_canopy/goal_op_const_8_unscaled.npy")
x_true = npzread("data_canopy/s_true.npy")[1,1,:] #x_true atm = 0.19, 1.31
z_true = O[3:end]' * x_true + O_offset
y = npzread("data_canopy/y.npy")[1,1,:]

# xa, xs = AOE.invert(y, rdbufs[1], rtmodel, priormodel)
fx = AOE.fwdfun(xa, xs, rtmodel) 
dfx = AOE.gradfwd_accel(xa, xs, rtmodel, fx)[:,3:end]
# x_map = vcat(xa,xs)

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


N = 1000
H = zeros((326,326))

for i = 1:N
    randx = rand(MvNormal(μ_x, Γ_x))
    fx = aoe_fwdfun(randx)
    dfx = aoe_gradfwdfun(randx, fx)

    H = H + 1/N * dfx * reshape(O', 328,1) * O * dfx' 
end
# print(size(dfx), size(reshape(O', 328,1)), size(O), size(dfx'))
size(dfx)
size(O)
size(invΓ_ϵ)
F = svd(H)
S = F.S
plot(S, yaxis=:log)


plot(cumsum(S)/sum(S))

fx = aoe_fwdfun(vcat([0.1,1.5], x_true))
dfx = aoe_gradfwdfun(vcat([0.1,1.5], x_true),fx)





F = svd(sqrt.(invΓ_ϵ) * dfx * dfx' * sqrt.(invΓ_ϵ));

F = svd(sqrt.(invΓ) * dfx' * dfx * sqrt.(invΓ_ϵ));
plot(F.S, yaxis=:log)
S = F.S
plot(cumsum(S)/sum(S) )






G = [ 1 10; 1 -1]
svd(G)
plot()