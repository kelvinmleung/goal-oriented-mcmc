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


# plot!(wls, fx, linecolor=:black, linewidth=2,label="Observed")
plot(wls,prsamp.y[:,1:100], linecolor=:red, alpha=0.1, label="")
plot!(wls,mean(prsamp.y,dims=2), linecolor=:blue, linewidth=2, label="Mean")
plot!(wls, setup.y, linecolor=:black, linewidth=2,label="Observed")

plot(wls, setup.x_true)
plot!(wls, setup.μ_x[3:end])


invsqrtcovy = inv(sqrt(cov(prsamp.y, dims=2)))
y_tilde = inv(sqrt(cov(prsamp.y, dims=2))) * (prsamp.y .- mean(prsamp.y,dims=2))
# # gradG_tilde * (sqrt(setup.invΓ_x) * (x_samp .- setup.μ_x))

# plot(wls, y_tilde[:,1:5], linecolor=:red, alpha=0.5, label="")
# plot!(wls, (gradG_tilde * (sqrt(setup.invΓ_x) * (x_samp .- setup.μ_x)))[:,1:5], linecolor=:blue, alpha=0.5, label="")

# plot(wls, y_tilde[:,1:100])


# plot((prsamp.y .- mean(prsamp.y,dims=2))[:,1:10])
# plot!(mean(prsamp.y,dims=2))

# # npzwrite("x_true_prsamp.npy", x_samp[3:end,1])
# setup.x_true .= npzread("x_true_prsamp.npy")#x_samp[3:end,1]
# setup.z_true .= setup.O[3:end]' * setup.x_true .+ setup.O_offset;
# fx, gradG, μ_xgy, Γ_xgy = replace_with_linearized_model!(setup);
# # setup.y .= fx + gradG * (vcat([0.2; 1.45], setup.x_true) - setup.μ_x) ;
# setup.y .= gradG * vcat([0.2; 1.45], setup.x_true);

# H_mat = diagnostic_matrix(10000)



# display(H_mat[1:5,1:5])
invsqrtΓ_ϵ = sqrt(setup.invΓ_ϵ)
sqrtΓ_x = sqrt(setup.Γ_x)
invsqrtΓ_z = sqrt(setup.invΓ_z)
gradG_tilde = invsqrtΓ_ϵ * gradG * sqrtΓ_x
# O_tilde = invsqrtΓ_z * setup.O * sqrtΓ_x
O_tilde = setup.O * sqrtΓ_x


# function test_nonwhiten()

# H_mat = gradG * setup.O' * setup.O * gradG'
# H_mat = gradG_tilde * O_tilde' * O_tilde * gradG_tilde'
H_mat = invsqrtcovy * gradG * setup.Γ_x * setup.O' * setup.invΓ_z * setup.O * setup.Γ_x * gradG' * invsqrtcovy


eigs, V = diagnostic_eigendecomp(H_mat; showplot=false, setup=setup)
eigs = real(eigs)
V = real(V)

plot(wls,invsqrtΓ_z .* (V[:,1]' * sqrtΓ_x[3:end,3:end])', linewidth=2, label="Diagnostic with Goal", title="Leading Eigenvector")
plot!(wls,invsqrtΓ_z .* (V[:,1]' * inv(sqrt(cov(prsamp.y,dims=2))))', linewidth=2, label="Diagnostic with Goal", title="Leading Eigenvector")
plot!(wls, O_tilde[3:end] ./2, label="O_tilde")
plot!(wls[O_tilde[3:end] .> 0.02], seriestype=:vline, alpha=0.2, linewidth=2, color=:red, label="Important to goal")

plot(wls,V[:,1], linewidth=2, label="Diagnostic with Goal", title="Leading Eigenvector")
plot!(wls, setup.O[3:end] ./2, label="O")



# histogram((invsqrtΓ_z .* V[:,1]' * sqrt(setup.Γ_ϵ)* (prsamp.y .- μ_y))[1:10:end])


r = energy_cutoff(eigs, 0.99)
V_r = invsqrtcovy * V[:,1:r]

# μ_y = vec(mean(prsamp.y, dims=2));
# plot(wls, μ_y, label="mu_y")
# plot!(wls, gradG * setup.μ_x, label="linmodel")

# y_whiten, yobs_whiten, z_whiten = whiten_samples(setup, prsamp, μ_y, V_r);
# # y_whiten, yobs_whiten, z_whiten = whiten_samples(setup, prsamp, μ_y, diagm(ones(326)));

# plot(wls, invsqrtΓ_ϵ* (prsamp.y .- μ_y)[:,1:10], color=:red, alpha=0.2 )

# plot(wls,(x_samp .- setup.μ_x)[3:end,1:100], color=:red, alpha=0.2 )

# histogram((V_r' * (prsamp.y .- μ_y))')



# plot(wls, (V_r * V_r' * (prsamp.y .- μ_y))[:,1:10])
# plot!(wls[O_tilde[3:end] .> 0.02], seriestype=:vline, alpha=0.2, linewidth=2, color=:red, label="Important to goal")


# # y_whiten = V_r' * sqrt(setup.Γ_ϵ) * (prsamp.y .- μ_y)
# # yobs_whiten = V_r' * sqrt(setup.Γ_ϵ) * (setup.y .- μ_y)

# # y_whiten = invsqrtΓ_z .* (V[:,1]' * sqrtΓ_x[3:end,3:end]) * (prsamp.y .- μ_y)
# # yobs_whiten = invsqrtΓ_z .* (V[:,1]' * sqrtΓ_x[3:end,3:end]) * (setup.y .- μ_y)


# scatter( prsamp.z[1:100:end], y_whiten[1:100:end], xlabel="Z", ylabel="Vᵣy_tilde", label="Joint Samples")
# plot!([yobs_whiten], seriestype=:hline, label="Observation")

μ_zgy_unwhiten, var_zgy_unwhiten = apply_cond_gaussian(vcat(V_r' * prsamp.y, prsamp.z), V_r' * setup.y; whitened=false) 


# μ_zgy_unwhiten, var_zgy_unwhiten = apply_cond_gaussian(vcat(y_whiten, prsamp.z), yobs_whiten; whitened=false) 
z_possamp_schur = rand(Normal(μ_zgy_unwhiten, sqrt(var_zgy_unwhiten)), m) .+ setup.O_offset 
z_truepos = rand(MvNormal(setup.O * μ_xgy .+ setup.O_offset, setup.O * Γ_xgy * setup.O'), m);

density(z_truepos[1:2:end], color=:black, linewidth=2, label="True Posterior", title="1D Goal Posterior - Marginal Density", legend=:topright, dpi=800)#xlim=[0.15,0.3])
density!(z_possamp_schur[1:2:end], label="EnKF, r = $(r)")
display(plot!([setup.z_true], seriestype="vline", color=:black, linestyle=:dash,linewidth=3, label="Truth"))

# end












H_mat = gradG_tilde * O_tilde' * O_tilde * gradG_tilde'
H_mat = gradG * setup.O' * setup.O * gradG'
# H_mat = gradG_tilde * gradG_tilde'

# H_mat = gradG * setup.O' * setup.O * gradG'

# H_mat = invsqrtΓ_ϵ * gradG * setup.Γ_x * setup.O' * setup.invΓ_z * setup.O * setup.Γ_x * gradG' * invsqrtΓ_ϵ
# display(H_mat[1:5,1:5])

# H_mat = invsqrtΓ_ϵ * gradG * setup.Γ_x * gradG' * invsqrtΓ_ϵ #* reshape(setup.O', n+2,1) * setup.invΓ_z * setup.O * setup.Γ_x 


# H_mat = inv(sqrt(cov(prsamp.y, dims=2))) * gradG * setup.Γ_x * gradG' * inv(sqrt(cov(prsamp.y, dims=2)))


eigs, V = diagnostic_eigendecomp(H_mat; showplot=false, setup=setup)
eigs = real(eigs)
V = real(V)

plot(wls,invsqrtΓ_z .* (V[:,1]' * sqrtΓ_x[3:end,3:end])', linewidth=2, label="Diagnostic with Goal", title="Leading Eigenvector")
plot!(wls, eigvecs(gradG_tilde * gradG_tilde')[:,end], linewidth=2, label="Diagnostic without Goal")

plot!(wls[O_tilde[3:end] .> 0.02], seriestype=:vline, alpha=0.2, linewidth=2, color=:red, label="Important to goal")

plot(wls, O_tilde[3:end])

plot!(setup.O' / norm(setup.O))
r = energy_cutoff(eigs, 0.99)


r = 100
V_r = V[:,1:r]


μ_y = vec(mean(prsamp.y, dims=2));
plot(wls, μ_y, label="mu_y")
plot!(wls, gradG * setup.μ_x, label="linmodel")

y_whiten, yobs_whiten, z_whiten = whiten_samples(setup, prsamp, μ_y, V_r);
# y_whiten, yobs_whiten, z_whiten = whiten_samples(setup, prsamp, μ_y, diagm(ones(326)));

x_whiten = sqrt(setup.invΓ_x) * (x_samp .- setup.μ_x)
YX = vcat(y_whiten, x_whiten)
jointmeanyx = mean(YX, dims=2)
jointcovyx = cov(YX, dims=2)
μ_y = @view jointmeanyx[1:r]
μ_x = @view jointmeanyx[r+1:end]
Σ_xx = @view jointcovyx[r+1:end, r+1:end]
Σ_yx = @view jointcovyx[1:r, r+1:end]

invΣ_yy = inv(jointcovyx[1:r, 1:r])
μ_xgy_whiten = μ_x .- Σ_yx' * invΣ_yy * (μ_y - yobs_whiten)
Γ_xgy_whiten = Σ_xx .- Σ_yx' * invΣ_yy * Σ_yx

μ_zgy_from_xgy = setup.O * (sqrt(setup.Γ_x) * μ_xgy_whiten .+ setup.μ_x) .+ setup.O_offset
Γ_zgy_from_xgy = setup.O * (sqrt(setup.Γ_x) * Γ_xgy_whiten * sqrt(setup.Γ_x))* setup.O'

z_via_xgy = rand(MvNormal(μ_zgy_from_xgy, Γ_zgy_from_xgy), m);


plot(wls, μ_xgy[3:end], label="True Posterior", title="Posterior X|Y Comparison")
plot!(wls, (sqrt(setup.Γ_x) * μ_xgy_whiten .+ setup.μ_x)[3:end], label="EnKF, r=$r")

plot(wls, 1 .-(abs.(sqrt(setup.Γ_x) * μ_xgy_whiten .+ setup.μ_x -μ_xgy) ./ μ_xgy)[3:end], label="EnKF, r=$r", title="Accuracy in Posterior Mean X|Y", xlabel="Wavelength", ylabel="Relative Accuracy", ylims=[0.8, 1.01])
plot!(wls[O_tilde[3:end] .> 0.02], seriestype=:vline, alpha=0.2, linewidth=2, color=:red, label="Important to goal")

# wls[setup.O[3:end] .> 0.1]
# histogram(sort(abs.(setup.O),dims=2)')
# histogram(sort(abs.(O_tilde),dims=2)')

# μ_zgy_unwhiten, var_zgy_unwhiten = apply_cond_gaussian(vcat(V_r' * (prsamp.y .- μ_y), prsamp.z), V_r' * (setup.y - μ_y); whitened=false)
# μ_zgy_whiten, var_zgy_whiten = apply_cond_gaussian(vcat(y_whiten, z_whiten), yobs_whiten; whitened=false)
# z_possamp_schur_whiten = rand(Normal(μ_zgy_whiten, sqrt(var_zgy_whiten)), m)
# z_possamp_schur = sqrt(setup.Γ_z) .* z_possamp_schur_whiten .+ setup.μ_z .+ setup.O_offset 

z_truepos = rand(MvNormal(setup.O * μ_xgy .+ setup.O_offset, setup.O * Γ_xgy * setup.O'), m);

density(z_truepos[1:2:end], color=:black, linewidth=2, label="True Posterior", title="1D Goal Posterior - Marginal Density", legend=:topright, dpi=800)#xlim=[0.15,0.3])
# density(z_truepos[1:10:end], color=:black, linewidth=2, label="True Posterior", title="a")
density!(z_via_xgy[1:2:end], label="EnKF via X|Y, r = $(r)")
# density!(z_possamp_schur, label="EnKF Method Whitened, r = $(r)")
# density!(rand(Normal(μ_zgy_unwhiten, sqrt(var_zgy_unwhiten)), Int(m/10)) .+ setup.O_offset , label="EnKF Unwhitened, r=$r")
display(plot!([setup.z_true], seriestype="vline", color=:black, linestyle=:dash,linewidth=3, label="Truth"))



function test_H_for_Xgy()
    H_mat = gradG_tilde * O_tilde' * O_tilde * gradG_tilde'
    # H_mat = gradG_tilde * gradG_tilde'

    eigs, V = diagnostic_eigendecomp(H_mat; showplot=false, setup=setup)
    eigs = real(eigs)
    V = real(V)

    plot(wls,invsqrtΓ_z .* (V[:,1]' * sqrtΓ_x[3:end,3:end])', linewidth=2, label="Diagnostic with Goal", title="Leading Eigenvector")
    plot!(wls, eigvecs(gradG_tilde * gradG_tilde')[:,end], linewidth=2, label="Diagnostic without Goal")
    plot!(wls[O_tilde[3:end] .> 0.02], seriestype=:vline, alpha=0.2, linewidth=2, color=:red, label="Important to goal")

    r = energy_cutoff(eigs, 0.9999)
    r = 100
    V_r = V[:,1:r]

    μ_y = vec(mean(prsamp.y, dims=2));
    plot(wls, μ_y, label="mu_y")
    plot!(wls, gradG * setup.μ_x, label="linmodel")

    y_whiten, yobs_whiten, z_whiten = whiten_samples(setup, prsamp, μ_y, V_r);

    x_whiten = sqrt(setup.invΓ_x) * (x_samp .- setup.μ_x)
    YX = vcat(y_whiten, x_whiten)
    jointmeanyx = mean(YX, dims=2)
    jointcovyx = cov(YX, dims=2)
    μ_y = @view jointmeanyx[1:r]
    μ_x = @view jointmeanyx[r+1:end]
    Σ_xx = @view jointcovyx[r+1:end, r+1:end]
    Σ_yx = @view jointcovyx[1:r, r+1:end]

    invΣ_yy = inv(jointcovyx[1:r, 1:r])
    μ_xgy_whiten = μ_x .- Σ_yx' * invΣ_yy * (μ_y - yobs_whiten)
    Γ_xgy_whiten = Σ_xx .- Σ_yx' * invΣ_yy * Σ_yx

    μ_zgy_from_xgy = setup.O * (sqrt(setup.Γ_x) * μ_xgy_whiten .+ setup.μ_x) .+ setup.O_offset
    Γ_zgy_from_xgy = setup.O * (sqrt(setup.Γ_x) * Γ_xgy_whiten * sqrt(setup.Γ_x))* setup.O'

    z_via_xgy = rand(MvNormal(μ_zgy_from_xgy, Γ_zgy_from_xgy), m);

    plot(wls, μ_xgy[3:end], label="True Posterior", title="Posterior X|Y Comparison")
    plot!(wls, (sqrt(setup.Γ_x) * μ_xgy_whiten .+ setup.μ_x)[3:end], label="EnKF, r=$r")

    plot(wls, 1 .-(abs.(sqrt(setup.Γ_x) * μ_xgy_whiten .+ setup.μ_x -μ_xgy) ./ μ_xgy)[3:end], label="EnKF, r=$r", title="Accuracy in Posterior Mean X|Y", xlabel="Wavelength", ylabel="Relative Accuracy", ylims=[0.8, 1.01])
    plot!(wls[O_tilde[3:end] .> 0.02], seriestype=:vline, alpha=0.2, linewidth=2, color=:red, label="Important to goal")

    z_truepos = rand(MvNormal(setup.O * μ_xgy .+ setup.O_offset, setup.O * Γ_xgy * setup.O'), m);

    density(z_truepos[1:2:end], color=:black, linewidth=2, label="True Posterior", title="1D Goal Posterior - Marginal Density", legend=:topright, dpi=800)#xlim=[0.15,0.3])
    density!(z_via_xgy[1:2:end], label="EnKF via X|Y, r = $(r)")# density!(rand(Normal(μ_zgy_unwhiten, sqrt(var_zgy_unwhiten)), Int(m/10)) .+ setup.O_offset , label="EnKF Unwhitened, r=$r")
    display(plot!([setup.z_true], seriestype="vline", color=:black, linestyle=:dash,linewidth=3, label="Truth"))

end
