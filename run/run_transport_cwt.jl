include("../src/forward.jl")
include("../src/mcmc_simple.jl")
include("../src/mcmc_1d.jl")

Random.seed!(123)
using TransportBasedInference, StatsPlots, NPZ, AOE

λ_ranges = [400.0 1300.0; 1450.0 1780.0; 2051.0 2451.0]
priormodel, wls = get_priormodel(:standard; λ_ranges) # PriorModel instance
rtmodel = AOE.get_radiative_transfer_modtran(:LUTRT1; λ_ranges);
rdbufs = get_RetrievalData_bufs(326) 
n, p = 326, 1
m = 50000
m_naive = 3000000

prmean = npzread("data/data_refl/10pix/prmean_10pix.npy")
prcov = npzread("data/data_refl/10pix/prcov_10pix.npy")
prclass = npzread("data/data_refl/10pix/prclass_10pix.npy")
prscale = npzread("data/data_refl/10pix/prscale_10pix.npy")
s_all = npzread("data/data_refl/10pix/s_10pix.npy")
y_all = npzread("data/data_refl/10pix/y_10pix.npy")

##### INDEX 1,1 ######
indx, indy = 1,1

O = vcat(zeros(2), npzread("data/data_canopy/goal_op_"* string(Int(prclass[indx,indy])) *"_unscaled.npy"))' / prscale[indx,indy]
O_offset = npzread("data/data_canopy/goal_op_const_"* string(Int(prclass[indx,indy])) *"_unscaled.npy")
x_true = s_all[indx,indy,:] 
z_true = O[3:end]' * x_true + O_offset
y = y_all[indx,indy,:] + rand(MvNormal(zeros(n), diagm(AOE.dummy_noisemodel(y_all[indx,indy,:]))))

xa, xs = AOE.invert(y, rdbufs[1], rtmodel, priormodel)
fx = AOE.fwdfun(xa, xs, rtmodel) 
dfx = AOE.gradfwd_accel(xa, xs, rtmodel, fx)[:,3:end]
x_map = vcat(xa,xs)

# Inference parameters
μ_x = vcat([0.2; 1.3], prmean[indx,indy,:])
Γ_x = zeros((328, 328))
Γ_x[1:2,1:2] = [0.01 0; 0 0.04]
Γ_x[3:end,3:end] = prcov[indx,indy,:,:]

Γ_ϵ = diagm(AOE.dummy_noisemodel(y))# diagm(y * 1e-4)

μ_z = O * μ_x #+ O_offset
Γ_z = O * Γ_x * O'
invΓ_x, invΓ_z, invΓ_ϵ = inv(Γ_x), inv(Γ_z), inv(Γ_ϵ)
Q = Γ_x * O' * invΓ_z

normDist = MvNormal(μ_x, Γ_x)
x_prsamp = rand(normDist, m)
z_prsamp = O * x_prsamp
noiseDist = MvNormal(zeros(n), Γ_ϵ)
y_prsamp_pred = rand(noiseDist, m)

for i = 1:m
    y_prsamp_pred[:,i] = y_prsamp_pred[:,i] + aoe_fwdfun(x_prsamp[:,i])
end

# yz_prsamp = hcat(y_prsamp_pred', (z_prsamp .- O * μ_x)')


## DIMENSION REDUCTION


N = 10000

# WHITENED
H = zeros((n,n))
invsqrtΓ_ϵ = inv(sqrt(Γ_ϵ))
for i = 1:N
    randx = rand(MvNormal(μ_x, Γ_x))
    fx = aoe_fwdfun(randx)
    dfx = aoe_gradfwdfun(randx, fx)
    H = H + 1/N * invsqrtΓ_ϵ * dfx * Γ_x *  reshape(O', n+2,1) * O * Γ_x * dfx' * invsqrtΓ_ϵ
end
plot(eigvals(H)[end:-1:1],yaxis=:log, label="")
plot!([18], seriestype="vline", color=:black, linewidth=1)

V = eigvecs(H)[:,end:-1:1]
plot(V[:,1:5], linewidth=2,title="Leading eigenvectors")






X = vcat(y_prsamp_pred, (z_prsamp .- O * μ_x))




Nx, Ny = p, n
S = HermiteMap(10, X; diag = true, factor = 0.5, α = 1e-6, b = "ProHermiteBasis");
@time S = optimize(S, X, "split"; maxterms = 30, withconstant = true, withqr = true, verbose = true, 
                                  maxpatience = 30, start = 1, hessprecond = true)

F = evaluate(S, X; start = Ny+1)

# Let's generate the posterior samples by partially inverting the map $\boldsymbol{S}^{\boldsymbol{\mathcal{X}}}$, for $\boldsymbol{y}^\star = 0.25$
Ystar = y

Xa = deepcopy(X)
@time hybridinverse!(Xa, F, S, Ystar; apply_rescaling = true, start = 2)
z_possamp_transport = Xa[4,:]                           


# nComp = 10

# gmm = GMM(nComp, yz_prsamp, method=:kmeans, kind=:full, nInit=100, nIter=50, nFinal=50)

# z_possamp_gmmpos = gmm_pos_samp(gmm, y, 100000)
# density(z_possamp_gmmpos)

# naive
# @time x_possamp = mcmc_bm_3block(μ_x, Γ_x, Γ_ϵ, y, m_naive)
# z_possamp_naive = (O * x_possamp)' .+ O_offset
# npzwrite("data/data_canopy/june28/10pix_ind("*string(indx)*","*string(indy)*")/z_naive.npy", z_possamp_naive)
z_possamp_naive = npzread("data/data_canopy/june28/10pix_ind("*string(indx)*","*string(indy)*")/z_naive.npy")


# low rank 1D
# @time z_possamp_lowrank_covexpand = mcmc_lis_1d(vcat(xa,xs), μ_x, Γ_x, Γ_ϵ, Q, O, y; N=m, logposmethod="covexpand") .+ O_offset
# npzwrite("data/data_canopy/june28/10pix_ind("*string(indx)*","*string(indy)*")/z_covexpand.npy", z_possamp_lowrank_covexpand)
z_possamp_covexpand = npzread("data/data_canopy/june28/10pix_ind("*string(indx)*","*string(indy)*")/z_covexpand.npy")

# @time z_possamp_lowrank_gmm = mcmc_lis_1d(vcat(xa,xs), μ_x, Γ_x, Γ_ϵ, Q, O, y; N=m, logposmethod="gmm") .+ O_offset
# npzwrite("data/data_canopy/june28/10pix_ind("*string(indx)*","*string(indy)*")/z_gmm.npy", z_possamp_lowrank_gmm)
z_possamp_gmm = npzread("data/data_canopy/june28/10pix_ind("*string(indx)*","*string(indy)*")/z_gmm.npy")

# @time z_possamp_lowrank_pseudomarg = mcmc_lis_1d(vcat(xa,xs), μ_x, Γ_x, Γ_ϵ, Q, O, y; N=m, logposmethod="pseudomarg") .+ O_offset
# npzwrite("data/data_canopy/june28/10pix_ind(1,1)/z_pseudomarg.npy", z_possamp_lowrank_pseudomarg)


density(z_possamp_gmm[2000:1:end], color=:blue, linewidth=2, label="Low Rank - GMM",  title="1D Goal Posterior - Marginal Density")
density!(z_possamp_covexpand[2000:1:end], color=:red, linewidth=2, label="Low Rank - CovExpand")
density!(z_possamp_naive[1000000:10:end], color=:black, linewidth=2, label="Naive")#, xlim=[0.1,0.4])
display(plot!([z_true], seriestype="vline", color=:black, linewidth=3, label="Truth"))

plot(z_possamp_gmm[1:1:50000], color=:blue, linewidth=0.5, label="Low Rank - GMM",  title="1D Goal Posterior - Marginal Density")



