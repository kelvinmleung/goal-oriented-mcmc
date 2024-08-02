include("../src/forward.jl")
include("../src/mcmc_simple.jl")
include("../src/mcmc_1d.jl")
include("../src/goalorientedtransport.jl")

Random.seed!(123)

λ_ranges = [400.0 1300.0; 1450.0 1780.0; 2051.0 2451.0]
priormodel, wls = get_priormodel(:standard; λ_ranges) # PriorModel instance
rtmodel = AOE.get_radiative_transfer(:modtran; λ_ranges);
n, p = 326, 1
m = 1000000
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


# plot!(wls, fx, linecolor=:black, linewidth=2,label="Observed")
plot(wls,prsamp.y[:,1:100], linecolor=:red, alpha=0.1, label="")
plot!(wls,mean(prsamp.y,dims=2), linecolor=:blue, linewidth=2, label="Mean")
plot!(wls, setup.y, linecolor=:black, linewidth=2,label="Observed")

plot(wls, setup.x_true)
plot!(wls, setup.μ_x[3:end])


# invsqrtcovy = inv(sqrt(gradG * setup.Γ_x * gradG' + setup.Γ_ϵ )) #inv(sqrt(cov(prsamp.y, dims=2)))
invsqrtcovy = inv(sqrt(cov(prsamp.y, dims=2)))

# y_tilde = inv(sqrt(cov(prsamp.y, dims=2))) * (prsamp.y .- mean(prsamp.y,dims=2))


# invsqrtΓ_ϵ = sqrt(setup.invΓ_ϵ)
# sqrtΓ_x = sqrt(setup.Γ_x)
# invsqrtΓ_z = sqrt(setup.invΓ_z)
# gradG_tilde = invsqrtΓ_ϵ * gradG * sqrtΓ_x
# # O_tilde = invsqrtΓ_z * setup.O * sqrtΓ_x
# O_tilde = setup.O * sqrtΓ_x



# function test_nonwhiten()

# H_mat = gradG * setup.O' * setup.O * gradG'
# H_mat = gradG_tilde * O_tilde' * O_tilde * gradG_tilde'
# H_mat = invsqrtcovy * gradG * setup.Γ_x * setup.O' * setup.invΓ_z * setup.O * setup.Γ_x * gradG' * invsqrtcovy
H = diagnostic_matrix(10000);


eigs, V = diagnostic_eigendecomp(H_mat; showplot=false, setup=setup)
eigs = real(eigs)
V = real(V)

# plot(wls,invsqrtΓ_z .* (V[:,1]' * sqrtΓ_x[3:end,3:end])', linewidth=2, label="Diagnostic with Goal", title="Leading Eigenvector")
# plot!(wls,invsqrtΓ_z .* (V[:,1]' * inv(sqrt(cov(prsamp.y,dims=2))))', linewidth=2, label="Diagnostic with Goal", title="Leading Eigenvector")
# plot!(wls, O_tilde[3:end] ./2, label="O_tilde")
# plot!(wls[O_tilde[3:end] .> 0.02], seriestype=:vline, alpha=0.2, linewidth=2, color=:red, label="Important to goal")

# plot(wls,V[:,1], linewidth=2, label="Diagnostic with Goal", title="Leading Eigenvector")
# plot!(wls, setup.O[3:end] ./2, label="O")



# histogram((invsqrtΓ_z .* V[:,1]' * sqrt(setup.Γ_ϵ)* (prsamp.y .- μ_y))[1:10:end])


r = energy_cutoff(eigs, 0.99)
V_r = invsqrtcovy * V[:,1:r]

# plot(wls, V[:,1:r] )
# O_tilde = inv(sqrt(setup.Γ_z)) *  setup.O * sqrt(setup.Γ_x)
# plot!(wls[O_tilde[3:end] .> 0.02], seriestype=:vline, alpha=0.2, linewidth=2, color=:red, label="Important to goal")



μ_zgy_unwhiten, var_zgy_unwhiten = apply_cond_gaussian(vcat(V_r' * prsamp.y, prsamp.z), V_r' * setup.y; whitened=false) 
# μ_zgy_unwhiten = (setup.μ_z .- setup.O * setup.Γ_x * gradG' * V_r * V_r' * (gradG * setup.μ_x - setup.y))[1]
# 
# μ_zgy_unwhiten, var_zgy_unwhiten = apply_cond_gaussian(vcat(y_whiten, prsamp.z), yobs_whiten; whitened=false) 
z_possamp_schur = rand(Normal(μ_zgy_unwhiten, sqrt(var_zgy_unwhiten)), m) .+ setup.O_offset 
z_truepos = rand(MvNormal(setup.O * μ_xgy .+ setup.O_offset, setup.O * Γ_xgy * setup.O'), m);

density(z_truepos[1:2:end], color=:black, linewidth=2, label="True Posterior", title="1D Goal Posterior - Marginal Density", legend=:topright, dpi=800)#xlim=[0.15,0.3])
density!(z_possamp_schur[1:2:end], label="EnKF, r = $(r)")
display(plot!([setup.z_true], seriestype="vline", color=:black, linestyle=:dash,linewidth=3, label="Truth"))

# end


# cov_vry_z = V_r' * gradG * setup.Γ_x *setup.O' 
# cov_vry = V_r' * (gradG * setup.Γ_x * gradG' + setup.Γ_ϵ) * V_r
# μ_z_g_vry = setup.μ_z .- cov_vry_z' * inv(cov_vry) * (V_r' * gradG * setup.μ_x - V_r' * setup.y)
# Γ_z_g_vry = setup.Γ_z - cov_vry_z' * inv(cov_vry) * cov_vry_z

# cov_y_z = gradG * setup.Γ_x *setup.O' 
# cov_y = (gradG * setup.Γ_x * gradG' + setup.Γ_ϵ) 
# μ_z_g_y = setup.μ_z .- cov_y_z' * inv(cov_y) * (gradG * setup.μ_x - setup.y)
# Γ_z_g_y = setup.Γ_z - cov_y_z' * inv(cov_y) * cov_y_z




