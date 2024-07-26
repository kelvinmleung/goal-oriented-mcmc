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


## Linear version to test
fx, gradG, μ_xgy, Γ_xgy = replace_with_linearized_model!(setup);

# make the samples adhere to the new linear model
# prsamp = gen_pr_samp(setup, m);
# plot(prsamp.y[:,1:100], linecolor=:black, alpha=0.1)
prsamp = initialize_EnsembleSamples(n, p, m)
x_samp = rand(MvNormal(setup.μ_x, setup.Γ_x), m)
prsamp.z .= setup.O * x_samp
println("Applying forward model to prior samples...")
@time for i = 1:m
    prsamp.fx[:,i] .= gradG * (x_samp[:,i])# - setup.μ_x) + fx
end
prsamp.y .= prsamp.fx + rand(MvNormal(zeros(n), setup.Γ_ϵ), m)

plot(wls, setup.y, linecolor=:black, linewidth=2,label="Observed")
# plot!(wls, fx, linecolor=:black, linewidth=2,label="Observed")
plot!(wls,prsamp.y[:,1:100], linecolor=:red, alpha=0.1, label="")

plot(wls, setup.x_true)
plot!(wls, setup.μ_x[3:end])



# # npzwrite("x_true_prsamp.npy", x_samp[3:end,1])
# setup.x_true .= npzread("x_true_prsamp.npy")#x_samp[3:end,1]
# setup.z_true .= setup.O[3:end]' * setup.x_true .+ setup.O_offset;
# fx, gradG, μ_xgy, Γ_xgy = replace_with_linearized_model!(setup);
# # setup.y .= fx + gradG * (vcat([0.2; 1.45], setup.x_true) - setup.μ_x) ;
# setup.y .= gradG * vcat([0.2; 1.45], setup.x_true);

# H_mat = diagnostic_matrix(10000)



# display(H_mat[1:5,1:5])
invsqrtΓ_ϵ = inv(sqrt(setup.Γ_ϵ))
H_mat = invsqrtΓ_ϵ * gradG * setup.Γ_x * setup.O' * setup.invΓ_z * setup.O * setup.Γ_x * gradG' * invsqrtΓ_ϵ
# display(H_mat[1:5,1:5])

# H_mat = invsqrtΓ_ϵ * gradG * setup.Γ_x * gradG' * invsqrtΓ_ϵ #* reshape(setup.O', n+2,1) * setup.invΓ_z * setup.O * setup.Γ_x 


# H_mat = inv(sqrt(cov(prsamp.y, dims=2))) * gradG * setup.Γ_x * gradG' * inv(sqrt(cov(prsamp.y, dims=2)))

# H_mat = gradG * setup.O' * setup.O * gradG'

eigs, V = diagnostic_eigendecomp(H_mat; showplot=false, setup=setup)
eigs = real(eigs)
V = real(V)
# plot(V[:,1:5])
r = energy_cutoff(eigs, 0.999)
r = 326
V_r = V[:,1:r]


μ_y = vec(mean(prsamp.y, dims=2));
y_whiten, yobs_whiten, z_whiten = whiten_samples(setup, prsamp, μ_y, V_r);
# y_whiten, yobs_whiten, z_whiten = whiten_samples(setup, prsamp, μ_y, diagm(ones(326)));

X = vcat(y_whiten, z_whiten)
scatter( z_whiten[1:10:end], y_whiten[1,1:10:end], xlabel="Z tilde", ylabel="Y1 tilde", alpha=0.2)
plot!([yobs_whiten[1]], seriestype="hline", color=:black, linewidth=3, label="Observation")
# kde_ref = kde(vec(F[end,:]), -3:0.01:4)
# # plot!(kde_ref.x, kde_ref.density ./ maximum(kde_ref.density) .* 250 .+ yobs_whiten[1], linewidth=2, label="Conditional Density")

y_tilde = V[:,1:r]' * sqrt(inv(cov(prsamp.y,dims=2))) * (prsamp.y .- μ_y);
scatter( z_whiten[1:10:end], y_tilde[1,1:10:end], xlabel="Z", ylabel="Ytilde", title="Whitened Samples", alpha=0.1, label="")
plot!( [V[:,1]' * sqrt(inv(cov(prsamp.y,dims=2))) * (setup.y .- μ_y)], seriestype=:hline,linewidth=2, label="Observation")

# plot(V[:,end-2])


μ_zgy_whiten, var_zgy_whiten = apply_cond_gaussian(X, yobs_whiten)
# μ_zgy_whiten = apply_cond_gaussian(X, yobs_whiten,V_r, setup, μ_y)

# r = 326
# cov_zy_whiten = cov(vcat(sqrt(inv(cov(prsamp.y,dims=2))) * (prsamp.y .- μ_y), z_whiten), dims=2)
# μ_zgy_whiten = -cov_zy_whiten[1:r,end]' * inv(cov_zy_whiten[1:r,1:r]) * sqrt(inv(cov(prsamp.y,dims=2))) * (setup.y .- μ_y)
# var_zgy_whiten = (cov_zy_whiten[end,end] - cov_zy_whiten[1:r,end]'* cov_zy_whiten[1:r,end])[1,1]
z_possamp_schur_whiten = rand(Normal(μ_zgy_whiten, sqrt(var_zgy_whiten)), m)
z_possamp_schur = sqrt(setup.Γ_z) .* z_possamp_schur_whiten .+ setup.μ_z .+ setup.O_offset 
# z_possamp_schur = z_possamp_schur_whiten .+ setup.μ_z .+ setup.O_offset 


z_truepos = rand(MvNormal(setup.O * μ_xgy .+ setup.O_offset, setup.O * Γ_xgy * setup.O'), m);

density(z_truepos[1:2:end], color=:black, linewidth=2, label="True Posterior", title="1D Goal Posterior - Marginal Density", legend=:topright, dpi=800)#xlim=[0.15,0.3])

density!(z_possamp_schur, label="EnKF Method, r = $(r)")
display(plot!([setup.z_true], seriestype="vline", color=:black, linestyle=:dash,linewidth=3, label="Truth"))




# density!(prsamp.z[1:10:end] .+ setup.O_offset , color=:blue, label="Prior")
# bar!(wls, abs.(setup.O[3:end]), title="Influence on bands on CWT")

