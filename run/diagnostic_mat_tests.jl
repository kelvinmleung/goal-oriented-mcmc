include("../src/forward.jl")
# include("../src/mcmc.jl")
# include("../src/mcmc_1d.jl")

Random.seed!(123)
using StatsPlots, NPZ, FiniteDiff


data_dir = "data/data_canopy/"
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

# analyticgrad = aoe_gradfwdfun(x_concat, fx)
# finitediff = FiniteDiff.finite_difference_jacobian(aoe_fwdfun, x_concat)
@time analyticgrad = AOE.gradfwd_accel(x_concat[1:2], x_concat[3:end], rtmodel, fx);

@time finitediff = FiniteDiff.finite_difference_jacobian(aoe_fwdfun, x_concat)
diffgrad = finitediff - analyticgrad ### CHECK THIS

finitediff - analyticgrad ### CHECK THIS

plot(diag(diffgrad[:,3:end]))


# finite difference


###AUTODIFF CHECK
### using Test
# create test folder
# using ForwardDiff



O = vcat(zeros(2), npzread(data_dir * "goal_op_8_unscaled.npy"))' / npzread(data_dir * "prscale_1_1.npy")
O_offset = npzread(data_dir * "goal_op_const_8_unscaled.npy")
x_true = npzread(data_dir * "s_true.npy")[1,1,:] #x_true atm = 0.19, 1.31
z_true = O[3:end]' * x_true + O_offset
y = npzread(data_dir * "y.npy")[1,1,:]

# xa, xs = AOE.invert(y, rdbufs[1], rtmodel, priormodel)
fx = AOE.fwdfun(xa, xs, rtmodel) 
dfx = AOE.gradfwd_accel(xa, xs, rtmodel, fx)[:,3:end]
# x_map = vcat(xa,xs)

# Inference parameters
μ_x = vcat([0.2; 1.3], npzread(data_dir * "prmean_1_1.npy"))
Γ_x = zeros((328, 328))
Γ_x[1:2,1:2] = [0.01 0; 0 0.04]
Γ_x[3:end,3:end] = npzread(data_dir * "prcov_1_1.npy")
Γ_ϵ = diagm(y * 1e-4)
μ_z = O * μ_x
Γ_z = O * Γ_x * O'
invΓ_x, invΓ_z, invΓ_ϵ = inv(Γ_x), inv(Γ_z), inv(Γ_ϵ)
Q = Γ_x * O' * invΓ_z


N = 10000

# NON WHITENED
H = zeros((326,326))
@time for i = 1:N
    randx = rand(MvNormal(μ_x, Γ_x))
    fx = aoe_fwdfun(randx)
    dfx = aoe_gradfwdfun(randx, fx)
    # dfx = FiniteDiff.finite_difference_jacobian(aoe_fwdfun, randx)

    H = H + 1/N * dfx * reshape(O', 328,1) * O * dfx' 
end
F = svd(H)
S = F.S
display(plot(S, yaxis=:log, label="", title="Diagnostic Matrix - Non Whitened"))


# WHITENED
H = zeros((326,326))
invsqrtΓ_ϵ = inv(sqrt(Γ_ϵ))
for i = 1:N
    randx = rand(MvNormal(μ_x, Γ_x))
    fx = aoe_fwdfun(randx)
    dfx = aoe_gradfwdfun(randx, fx)
    # dfx = FiniteDiff.finite_difference_jacobian(aoe_fwdfun, randx)

    H = H + 1/N * invsqrtΓ_ϵ * dfx * Γ_x *  reshape(O', 328,1) * O * Γ_x * dfx' * invsqrtΓ_ϵ
end
F = svd(H)
S = F.S
display(plot(S, yaxis=:log, label="", title="Diagnostic Matrix - Whitened"))






# G = [ 1 10; 1 -1]
# svd(G)
# plot()