include("../src/forward.jl")
include("../src/mcmc_simple.jl")
include("../src/mcmc_1d.jl")
include("../src/goalorientedtransport.jl")

Random.seed!(123)

λ_ranges = [400.0 1300.0; 1450.0 1780.0; 2051.0 2451.0]
priormodel, wls = get_priormodel(:standard; λ_ranges) # PriorModel instance
rtmodel = AOE.get_radiative_transfer(:modtran; λ_ranges);
n, p = 326, 1
m = 100000
m_naive = 3000000


setup = initialize_GODRdata(n, p)
GODRdata_from_10pix!(setup, 1,1);


# prsamp = gen_pr_samp(setup, m);
# H = diagnostic_matrix(10000);

# r = energy_cutoff(eigs, 0.9999999);
# eigs, V = diagnostic_eigendecomp(H)
# V_r = V[:,1:r]

# μ_y = vec(mean(prsamp.y, dims=2));
# y_whiten, yobs_whiten, z_whiten = whiten_samples(setup, prsamp, μ_y, V_r);
# X = vcat(y_whiten, z_whiten)
# # scatter(y_whiten[1,1:100:end], z_whiten[1:100:end])

# z_possamp_whiten = apply_cond_transport(X, yobs_whiten, r)
# z_possamp_transport = sqrt(setup.Γ_z) .* z_possamp_whiten .+ setup.μ_z .+ setup.O_offset 

# plotrange = 0.:0.001:0.5
# kde_transport = kde(vec(z_possamp_transport), plotrange)

# z_possamp_naive = npzread("data/data_canopy/june28/10pix_ind(1,1)/z_naive.npy")
# z_possamp_covexpand = npzread("data/data_canopy/june28/10pix_ind(1,1)/z_covexpand.npy")
# z_possamp_gmm = npzread("data/data_canopy/june28/10pix_ind(1,1)/z_gmm.npy")

# density(z_possamp_naive[1000000:10:end], color=:black, linewidth=2, label="Full Rank MCMC", title="1D Goal Posterior - Marginal Density", legend=:topright, dpi=800, xlim=[0.1,0.4])
# density!(z_possamp_gmm[2000:1:end], color=:red, linewidth=2, label="Low Rank MCMC")#, xlim=[0.15,0.3])
# plot!(kde_transport.x, kde_transport.density, color=:green, linewidth=2, label="Transport")
# display(plot!([setup.z_true], seriestype="vline", color=:black, linewidth=3, label="Truth"))



## Linear version to test
fx, gradG, μ_xgy, Γ_xgy = replace_with_linearized_model!(setup);

# make the samples adhere to the new linear model
# prsamp = gen_pr_samp(setup, m);
# plot(prsamp.y[:,1:100], linecolor=:black, alpha=0.1)
prsamp = initialize_EnsembleSamples(n, p, m)
x_samp = rand(MvNormal(setup.μ_x, setup.Γ_x), m)
prsamp.z .= setup.O * x_samp
prsamp.y .= rand(MvNormal(zeros(n), setup.Γ_ϵ), m)
println("Applying forward model to prior samples...")
@time for i = 1:m
    prsamp.y[:,i] .= prsamp.y[:,i] + gradG * (x_samp[:,i] - setup.μ_x) + fx
end
plot(wls, setup.y, linecolor=:black, linewidth=2,label="Observed")
plot!(wls,prsamp.y[:,1:100], linecolor=:red, alpha=0.1, label="Prior Samp")


setup.x_true .= x_samp[3:end,1];
setup.z_true .= setup.O[3:end]' * setup.x_true .+ setup.O_offset;
setup.y .= fx + gradG * (vcat([1.2,1.5], setup.x_true) - setup.μ_x) 



# avgdiff_aod = zeros(n)
# avgdiff_h2o = zeros(n)
# avgdiff_refl = zeros(n)
# ntest = 100
# for i = 1:ntest
#     diff = abs.(aoe_gradfwdfun(x_samp[:,i], aoe_fwdfun(x_samp[:,i])) - FiniteDiff.finite_difference_jacobian(aoe_fwdfun, x_samp[:,i]))
#     avgdiff_aod = avgdiff_aod + 1/ntest * diff[:,1]
#     avgdiff_h2o = avgdiff_h2o + 1/ntest * diff[:,2]
#     avgdiff_refl = avgdiff_refl + 1/ntest * diag(diff[:,3:end])
# end
# plot(title="Difference in Gradient by component")
# plot!(wls, avgdiff_aod, label="AOD")
# plot!(wls, avgdiff_h2o, label="H2O")
# plot!(wls, avgdiff_refl, label="Refl")


H = diagnostic_matrix(10000);


eigs, V = diagnostic_eigendecomp(H; showplot=true, setup=setup)

r = energy_cutoff(eigs, 0.999999);
V_r = V[:,1:r]

μ_y = vec(mean(prsamp.y, dims=2));
y_whiten, yobs_whiten, z_whiten = whiten_samples(setup, prsamp, μ_y, V_r);
X = vcat(y_whiten, z_whiten)
scatter( z_whiten[1:100:end], y_whiten[1,1:100:end], xlabel="Z tilde", ylabel="Y1 tilde")
plot!([yobs_whiten[1]], seriestype="hline", color=:black, linewidth=3, label="Observation")
# kde_ref = kde(vec(F[end,:]), -3:0.01:4)
# plot!(kde_ref.x, kde_ref.density ./ maximum(kde_ref.density) .* 250 .+ yobs_whiten[1], linewidth=2, label="Conditional Density")


y_tilde = V[:,1]' * (sqrt(inv(cov(prsamp.y,dims=2))) * (prsamp.y .- μ_y));
scatter( z_whiten[1:10:end], y_tilde[1,1:10:end], xlabel="Z", ylabel="Ytilde", title="Whitened Samples", alpha=0.1, label="")
plot!( [V[:,1]' * (sqrt(inv(cov(prsamp.y,dims=2))) * (setup.y .- μ_y))], seriestype=:hline,linewidth=2, label="Observation")



# scatter(X[1,:]./sqrt(var(X[1,:])),X[2,:])
# plot!([ystar], seriestype="hline", color=:black, linewidth=3, label="Observation")
# kde_ref = kde(vec(F[end,:]), -3:0.01:3)
# plot!(kde_ref.x, kde_ref.density ./ maximum(kde_ref.density) .+ ystar, linewidth=2, label="Conditional Density")


# H_unwhite = diagnostic_matrix_nonwhiten(100000)
# r_unwhite = energy_cutoff(eigs, 0.999999);
# eigs_unwhite, V_unwhite = diagnostic_eigendecomp(H_unwhite; showplot=true, setup=setup)
# V_r_unwhite = V_unwhite[:,1:r_unwhite]
# X_unwhite = vcat(V_r_unwhite' * prsamp.y, prsamp.z)
# scatter( prsamp.z[1:100:end], (V_r_unwhite' * prsamp.y)[1,1:100:end], xlabel="Z tilde", ylabel="Y1 tilde")
# plot!([(V_r_unwhite' * setup.y)[1]], seriestype="hline", color=:black, linewidth=3, label="Observation")

# z_possamp_unwhite, S_unwhite, F_unwhite = apply_cond_transport(X_unwhite, repeat(V_r_unwhite' * setup.y, 1, m), r; order=10)
# z_possamp_transport_unwhite = z_possamp_unwhite.+ setup.O_offset 
# density(z_possamp_transport_unwhite)


z_possamp_whiten, S, F = apply_cond_transport(X, repeat(yobs_whiten, 1, m), r; order=10)



z_possamp_transport =  sqrt(setup.Γ_z) .* z_possamp_whiten .+ setup.μ_z .+ setup.O_offset 
plotrange = 0.:0.001:0.3
kde_transport = kde(vec(z_possamp_transport), plotrange)
z_truepos = rand(MvNormal(setup.O * μ_xgy .+ setup.O_offset, setup.O * Γ_xgy * setup.O'), m);

density(z_truepos[1:10:end], color=:black, linewidth=2, label="True Posterior", title="1D Goal Posterior - Marginal Density", legend=:topright, dpi=800, xlim=[0,0.3])#xlim=[0.15,0.3])
# density!(z_possamp_gmm[2000:1:end], color=:red, linewidth=2, label="Low Rank MCMC")#, xlim=[0.15,0.3])
plot!(kde_transport.x, kde_transport.density, color=:green, linewidth=2, label="Transport")
display(plot!([setup.z_true], seriestype="vline", color=:black, linestyle=:dash,linewidth=3, label="Truth"))

density!(prsamp.z[1:10:end] .+ setup.O_offset , color=:blue, label="Prior")


bar!(wls, abs.(setup.O[3:end]), title="Influence on bands on CWT")