include("../src/forward.jl")
include("../src/goalorientedtransport.jl")

Random.seed!(123)

λ_ranges = [400.0 1300.0; 1450.0 1780.0; 2051.0 2451.0]
# λ_ranges = [300.0 2550.0]
priormodel, wls = get_priormodel(:standard; λ_ranges) # PriorModel instance

rtmodel = AOE.get_radiative_transfer(:modtran; λ_ranges);
n, p = 326, 7
m = 100000
# m_naive = 3000000


setup = initialize_GODRdata(n, p)
size(setup.O)

GODRdata_prmix!(setup; n=n, p=p)

plot(wls,setup.μ_x[3:end])


prsamp = gen_pr_samp(setup, m; n=n, p=p);

plot(wls,prsamp.y[:,1:100], linecolor=:red, alpha=0.1, label="")
plot!(wls,mean(prsamp.y,dims=2), linecolor=:blue, linewidth=2, label="Mean")
plot!(wls, setup.y, linecolor=:black, linewidth=2,label="Observed")

plot(wls, setup.x_true)
plot!(wls, setup.μ_x[3:end])


oper, srfmat = load("data/data_CliMA/goaloperator.jld","oper","srfmat")
plot(oper)


setup.O .= hcat(zeros(p,2), oper' * srfmat)
setup.O_offset .= load("data/data_CliMA/goaloperator.jld","offset")


histogram(prsamp.z[1,:])

invsqrtcovy = inv(sqrt(cov(prsamp.y, dims=2)))
meany = mean(prsamp.y, dims=2)

H_mat = diagnostic_matrix(10000, invsqrtcovy)

eigs, V = diagnostic_eigendecomp(H_mat; showplot=true, setup=setup)

prsamp.z



r = energy_cutoff(eigs, 0.99)



V_r = invsqrtcovy * V[:,1:r]

plot(wls,  V[:,1:r] )
# O_tilde = inv(sqrt(setup.Γ_z)) *  setup.O * sqrt(setup.Γ_x)
# plot!(wls[O_tilde[3:end] .> 0.02], seriestype=:vline, alpha=0.2, linewidth=2, color=:red, label="Important to goal")



X = vcat(V_r' * prsamp.y, sqrt(setup.invΓ_z) * (prsamp.z .- setup.μ_z))[:,1:Int(m/10)]
yobs = repeat(V_r' * setup.y, 1, Int(m/10))

# scatter(X[3,:], X[1,:], alpha=0.2)
# plot!([(V_r' * (setup.y - meany))[1]], seriestype=:hline)

# # scatter(X[3,:], X[2,:])
# plot!([(V_r' * setup.y)[2]], seriestype=:hline)

z_possamp, S, F = apply_cond_transport(X, yobs, r; order=10)

# histogram(z_possamp_transport)
z_possamp_transport =  sqrt(setup.Γ_z) * z_possamp .+ setup.μ_z .+ setup.O_offset 
# plotrange = 0.:0.001:0.3
# kde_transport = kde(vec(z_possamp_transport), plotrange)
# z_truepos = rand(MvNormal(setup.O * μ_xgy .+ setup.O_offset, setup.O * Γ_xgy * setup.O'), m);

μ_zgy, Σ_zgy = apply_cond_gaussian(X, vec(V_r' * setup.y), r; whitened=false)
zgy_samp =  sqrt(setup.Γ_z) * rand(MvNormal(μ_zgy, tril(Σ_zgy, 0) + tril(Σ_zgy, -1)'), m) .+ setup.O_offset .+ setup.μ_z


# density(z_truepos[1:10:end], color=:black, linewidth=2, label="True Posterior", title="1D Goal Posterior - Marginal Density", legend=:topright, dpi=800, xlim=[0,0.3])#xlim=[0.15,0.3])
# density(z_possamp_gmm[2000:1:end], color=:red, linewidth=2, label="Low Rank MCMC")#, xlim=[0.15,0.3])
# plot!(kde_transport.x, kde_transport.density, color=:green, linewidth=2, label="Transport")
# density(z_possamp_naive[2000000:100:end], linewidth=2, label="MCMC")

indQOI = 4
density(z_possamp_transport[indQOI, 1:10:end], linewidth=2, label="Goal-Oriented SBI", dpi=300)
density!(zgy_samp[indQOI, 1:10:end], label="EnKF Update")
display(plot!([setup.z_true[indQOI]], seriestype="vline", color=:black, linestyle=:dash,linewidth=3, label="Truth"))

