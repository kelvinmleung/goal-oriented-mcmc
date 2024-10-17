include("../src/goalorientedtransport.jl")



λ_ranges = [400.0 1300.0; 1450.0 1780.0; 2051.0 2451.0]
priormodel, wls = get_priormodel(:standard; λ_ranges) # PriorModel instance
rtmodel = AOE.get_radiative_transfer(:modtran; λ_ranges);
n, p = 326, 6

prjld = jldopen("./priordata/priors_standard.jld")
means = collect(eachcol(prjld["means"][:,:]'))
covs = [prjld["covs"][:,:,:][i, :, :] for i ∈ 1:length(means)]
pr_λs = load("./LUTdata/wls.jld")["wls"]
idx_326 = AOE.get_λ_idx(pr_λs, [400.0 1300.0; 1450.0 1780.0; 2051.0 2451.0])
wls = pr_λs[idx_326]

oper, offset, wls_clima, idx_clima, refl_clima, goal_clima, selectQOI = load("data/data_CliMA/goaloperator_pr8.jld", "goaloperator", "offset", "wls_clima", "idx_clima", "sampRefl", "sampQOI", "selectQOI")

O = oper[:,selectQOI]
O_offset = offset[selectQOI]

indPr = 8
prmean, prcov = means[indPr][idx_326], covs[indPr][idx_326,idx_326] / 4



srfmat = get_srfmat(wls_clima, pr_λs)[idx_clima, idx_326]



N = 1000
Random.seed!(10032024)
samp_a = rand(MvNormal([0.2; 1.45], [0.01 0; 0 0.004]), N)
samp_s = rand(MvNormal(prmean, prcov), N) 
samp_γ = rand(Uniform(1,5), N) 
plot(wls_clima[idx_clima], srfmat * samp_s, alpha=1, color=:red, label="")
# plot(wls, samp_x, alpha=1, color=:red, label="")
plot!(wls_clima[idx_clima], refl_clima[:,1:N], alpha=1, color=:blue, label="")


## Apply goal operator and srfmat to prior samples to see if we get sensible goals
samp_y = zeros(n, N)
for i in 1:N
    samp_y[:,i] = AOE.fwdfun(samp_a[:,i], samp_s[:,i] * samp_γ[i], rtmodel)
end
samp_z = O' * srfmat * samp_s .+ O_offset

# Verify values to make sure most are within in the limits
keys = ["BROWN","CHL","CBC", "LMA", "PRO", "LWC"]
xlims = [[0,1], [0,90], [0,0.05], [0.01,0.05], [0,0.005], [0,20]]
for i in 1:p
    display(histogram(samp_z[i,:], title=keys[i], label="", xlim=xlims[i]))
end


a_true, s_true, γ_true = rand(MvNormal([0.2; 1.45], [0.01 0; 0 0.004])), rand(MvNormal(prmean, prcov)),  rand(Uniform(1,5)) 
y_true, z_true = AOE.fwdfun(a_true, s_true * γ_true, rtmodel), O' * srfmat * s_true .+ O_offset


samp_z[:,1:10]

############ TEST WITH goalorientedtransport.jl ###############

using JLD, SRFTools
m, n, p = 100000, 326, 6
include("../src/forward.jl")
include("../src/goalorientedtransport.jl")
setup = initialize_GODRdata(n, p)
GODRdata_pr8!(setup; n=n, p=p);
# prsamp = gen_pr_samp_with_scale(setup, m; n=n, p=p);
prsamp = gen_pr_samp(setup, m; n=n, p=p);

invsqrtcovy = inv(sqrt(cov(prsamp.y, dims=2)))
meany = mean(prsamp.y, dims=2)
H_mat = diagnostic_matrix(10000, invsqrtcovy)
eigs, V = diagnostic_eigendecomp(H_mat; showplot=true, setup=setup)
r = energy_cutoff(eigs, 0.99)

V_r = invsqrtcovy * V[:,1:r]

X = vcat(V_r' * prsamp.y, sqrt(setup.invΓ_z) * (prsamp.z .- setup.μ_z))[:,1:Int(m/10)]
yobs = repeat(V_r' * setup.y, 1, Int(m/10))
# z_possamp, S, F = apply_cond_transport(X, yobs, r; order=10)
# z_possamp_transport =  sqrt(setup.Γ_z) * z_possamp .+ setup.μ_z .+ setup.O_offset 

μ_zgy, Σ_zgy = apply_cond_gaussian(X, vec(V_r' * setup.y), r)
zgy_samp =  sqrt(setup.Γ_z) * rand(MvNormal(μ_zgy, tril(Σ_zgy, 0) + tril(Σ_zgy, -1)'), m) .+ setup.O_offset .+ setup.μ_z
# 

keys = ["BROWN","CHL","CBC", "LMA", "PRO", "LWC"]
keys = ["Senescent material (brown pigments) fraction","Chlorophyll content", "Carbon-based constituents", "Dry leaf mass per unit area", "Protein content", "Leaf water content"]
xlims = [[0,1], [0,90], [0,0.05], [0.01,0.05], [0,0.005], [0,20]]
for i in 1:p
    # display(histogram(samp_z[i,:], title=keys[i], label="", xlim=xlims[i]))
    # density(z_possamp_transport[i, 1:10:end], linewidth=2, label="Goal-Oriented SBI", dpi=300)
    density(zgy_samp[i, 1:10:end], label="EnKF Update")
    density!(prsamp.z[i, 1:10:end] .+ setup.O_offset[i], label="Prior")
    display(plot!([setup.z_true[i]], seriestype="vline", color=:black, linestyle=:dash,linewidth=3, label="Truth", title=keys[i],  legend=:bottomright)) #xlim=xlims[i],
    # savefig("plots/10032024/qoi_$i.png")
end


# # PLOT VERIFICATION
# plot(wls,setup.μ_x[3:end])

# plot(wls,prsamp.y[:,1:100], linecolor=:red, alpha=0.1, label="")
# plot!(wls,mean(prsamp.y,dims=2), linecolor=:blue, linewidth=2, label="Mean")
# plot!(wls, setup.y, linecolor=:black, linewidth=2,label="Observed")

# plot(wls, setup.x_true)
# plot!(wls, setup.μ_x[3:end])

# histogram(prsamp.z[2,:] .+ setup.O_offset[2])

# plot(wls,  V[:,1:r])


