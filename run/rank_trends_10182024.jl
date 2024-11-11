
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
eigs, V = diagnostic_eigendecomp(H_mat; showplot=false, setup=setup)
# plot!(title="Eigenvalue Decay")
# savefig("plots/10172024/eigvals.png")


r = energy_cutoff(eigs, 0.99)
# r = 326



keys = ["BROWN","CHL","CBC", "LMA", "PRO", "LWC"]
keys = ["Senescent material (brown pigments) fraction","Chlorophyll content", "Carbon-based constituents", "Dry leaf mass per unit area", "Protein content", "Leaf water content"]
xlims = [[0,1], [0,90], [0,0.05], [0.01,0.05], [0,0.005], [0,20]]

indQOI = 3

density(prsamp.z[indQOI, 1:10:end] .+ setup.O_offset[indQOI], label="Prior", color=:black)

for r ∈ [1,7,10,50,200]
    V_r = invsqrtcovy * V[:,1:r]



    X = vcat(V_r' * prsamp.y, sqrt(setup.invΓ_z) * (prsamp.z .- setup.μ_z))[:,1:Int(m/10)]
    yobs = repeat(V_r' * setup.y, 1, Int(m/10))

    μ_zgy, Σ_zgy = apply_cond_gaussian(X, vec(V_r' * setup.y), r)
    zgy_samp =  sqrt(setup.Γ_z) * rand(MvNormal(μ_zgy, tril(Σ_zgy, 0) + tril(Σ_zgy, -1)'), m) .+ setup.O_offset .+ setup.μ_z


    X_pca = vcat(V_pca[:,1:r]' * prsamp.y, sqrt(setup.invΓ_z) * (prsamp.z .- setup.μ_z))[:,1:Int(m/10)]
    μ_zgy_pca, Σ_zgy_pca = apply_cond_gaussian(X_pca, vec(V_pca[:,1:r]' * setup.y), r)
    zgy_samp_pca =  sqrt(setup.Γ_z) * rand(MvNormal(μ_zgy_pca, tril(Σ_zgy_pca, 0) + tril(Σ_zgy_pca, -1)'), m) .+ setup.O_offset .+ setup.μ_z

    density!(zgy_samp[indQOI, 1:10:end], label="Rank $r (GODR)", color=:red, alpha=0.5)
    density!(zgy_samp_pca[indQOI, 1:10:end], label="Rank $r (PCA)", color=:blue, alpha=0.5)

end

display(plot!([setup.z_true[indQOI]], seriestype="vline", color=:black, linestyle=:dash,linewidth=3, label="Truth", title=keys[indQOI],  legend=:bottomright)) #xlim=xlims[i],

