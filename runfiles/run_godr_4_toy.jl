include("inverseProblem.jl")

## in this version we investigate the local and global eigenvalue problems using nonlinear forward model
Random.seed!(123)


n, p = 3, 2
O = [1 0 0; 0 1 0]
# Goal oriented 

# Inference parameters
μ_x = zeros(n) #0.5*ones(n)
μ_z = O * μ_x
σ_X², σ_Z² = 0.01, 0.01
σ_k² = 4.0
σ_ϵ² = 1e-4
Γ_ϵ = diagm(σ_ϵ² * ones(n))
Γ_x = diagm(σ_X² * ones(n))
Γ_z = O * Γ_x * O'
invΓ_x, invΓ_z, invΓ_ϵ = inv(Γ_x), inv(Γ_z), inv(Γ_ϵ)
Q = Γ_x * O' * invΓ_z

normDist = MvNormal(μ_x, Γ_x)
x_true = rand(normDist)
z_true = O*x_true
# Apply forward model
eps = sqrt(σ_ϵ²) .* randn(n)
y = fwdtoy(x_true) + eps


m = 100000
x_prsamp = rand(normDist, m)
z_prsamp = O * x_prsamp


x_possamp = mcmc_amm_simple(μ_x, μ_x, Γ_x, Γ_ϵ, y, m+20000)[:,20001:end]
x_pos_mean = mean(x_possamp[:,5*m:end], dims=2)
z_possamp = O * x_possamp

y_prsamp_pred = zeros(n,m)
y_possamp_pred = zeros(n,m)
y_true = fwdtoy(x_true)
for i in 1:m
    y_prsamp_pred[:,i] = fwdtoy(x_prsamp[:,i])
    y_possamp_pred[:,i] = fwdtoy(x_possamp[:,i])

end

plot(z_prsamp[1,:], z_prsamp[2,:], seriestype=:scatter, mc=:cyan, label="Prior", title="Samples of Z")
plot!(z_possamp[1,:], z_possamp[2,:], seriestype=:scatter, mc=:black, label="Posterior")
plot!([z_true[1]], [z_true[2]], seriestype=:scatter, mc=:red, label="Truth")

plot(y_prsamp_pred[1,:], y_prsamp_pred[2,:], seriestype=:scatter, mc=:cyan, label="Prior", title="Samples of Y")
plot!(y_possamp_pred[1,:], y_possamp_pred[2,:], seriestype=:scatter, mc=:black, label="Posterior")
plot!([y_true[1]], [y_true[2]], seriestype=:scatter, mc=:red, label="Truth")
plot!([y[1]], [y[2]], seriestype=:scatter, mc=:orange, label="Observation")


# Compare Hessian computed using prior and posterior
# also look at the trend from local to global Hessian

globalH_pr = zeros((p,p,m))
eigH_pr = zeros((p,m))
fZ_mean = zeros((n,m))
DeltaXZ_mean = zeros((n,m))
expected = zeros((n,n,m))
err_norm = zeros((n,m))

for j = 1:m
    gQzj = fwdtoy(Q * z_prsamp[:,j])
    dgQzj = dfwdtoy(Q * z_prsamp[:,j])
    deltaXZj = dgQzj * (I - Q * O) * x_prsamp[:,j]
    if j == 1
        globalH_pr[:,:,j] = Q' * dgQzj' * inv(dgQzj * (Γ_x - Q * O * Γ_x) * dgQzj' + Γ_ϵ) * dgQzj * Q
        fZ_mean[:,j] = gQzj
        DeltaXZ_mean[:,j] = deltaXZj
        expected[:,:,j] .= 0 #gQzj * deltaXZj'
    else
        globalH_pr[:,:,j] = (m-1) / m * globalH_pr[:,:,j-1] + 1/m * Q' * dgQzj' * inv(dgQzj * (Γ_x - Q * O * Γ_x) * dgQzj' + Γ_ϵ) * dgQzj * Q
        fZ_mean[:,j] = (m-1)/m * fZ_mean[:,j-1] + 1/m * gQzj
        DeltaXZ_mean[:,j] = (m-1)/m * DeltaXZ_mean[:,j-1] + 1/m * deltaXZj
        # display((m-1) / m * expected[:,:,j-1]  + 1/m * (gQzj - fZ_mean[:,j]) * (deltaXZj - DeltaXZ_mean[:,j])')
        expected[:,:,j] = (m-1) / m * expected[:,:,j-1]  + 1/m * (gQzj - fZ_mean[:,j]) * (deltaXZj - DeltaXZ_mean[:,j])'
    end
    err_norm[:,j] = y_prsamp_pred[:,m] - gQzj - dgQzj * (I - Q * O) * x_prsamp[:,j]
    eigH_pr[:,j] = eigvals(globalH_pr[:,:,j])[end:-1:1]
    # fZ[:,j] = gQzj 
    # DeltaXZ[:,j] = deltaXZj
    # display(fZ_mean[:,j])
end

m = 5000
globalH_pr_gradfree = zeros((p,p))
gQz_all = zeros(n,m)
L_Δ = cholesky(Γ_Δ)

for i = 1:m
    zi = z_prsamp[:,i]
    cholesky(dgQzj * (Γ_x - Q * O * Γ_x) * dgQzj' + Γ_ϵ)
    gQz_all[:,i] = fwdtoy(Q * zi)
    
    for j = 1:i-1
        zj = z_prsamp[:,j]
        gQzi = gQz_all[:,i]
        gQzj = gQz_all[:,j]
        # display((gQzi - gQzj)' * (gQzi - gQzj))
        # display((zi-zj) * (zi-zj)' / norm(zi-zj)^2)
        globalH_pr_gradfree = globalH_pr_gradfree + 2 / (m * (m-1)) * (gQzi - gQzj)' * (gQzi - gQzj) * (zi-zj) * (zi-zj)' / norm(zi-zj)^2
    end
end


eigvals(globalH_pr_gradfree)



globalH = zeros((p,p,m))
eigH = zeros((p,m))
for j = 1:m
    # gQz[:,j] = fwdtoy(Q * z_prsamp[:,j])
    dgQzj = dfwdtoy(Q * z_possamp[:,j])
    if j == 1
        globalH[:,:,j] = Q' * dgQzj' * inv(dgQzj * (Γ_x - Q * O * Γ_x) * dgQzj' + Γ_ϵ) * dgQzj * Q
    else
        globalH[:,:,j] = (m-1) / m * globalH[:,:,j-1] + 1/m * Q' * dgQzj' * inv(dgQzj * (Γ_x - Q * O * Γ_x) * dgQzj' + Γ_ϵ) * dgQzj * Q
    end
    eigH[:,j] = eigvals(globalH[:,:,j])[end:-1:1]
end

plot(eigH_pr[1,:], yaxis=:log, title="First Eigenvalue", label="Prior", xlabel="Ensemble Size")
display(plot!(eigH[1,:], yaxis=:log, label="Posterior"))
plot(eigH_pr[2,:], yaxis=:log, title="Second Eigenvalue", label="Prior", xlabel="Ensemble Size")
display(plot!(eigH[2,:], yaxis=:log, label="Posterior"))

plot(mapslices(norm, expected, dims=(1,2))[1,1,:], title="Monte Carlo Est. E[f(z) Δ(z,x)^⊤]", xlabel="Ensemble Size", label=false)

scatter( fZ[1,:], fZ[2,:], xlabel="Index 1", ylabel="Index 2", title="f(z)",label=false)
scatter( DeltaXZ[3,:], DeltaXZ[2,:], xlabel="Index 1", ylabel="Index 2", title="Δ(z,x)", label=false)

histogram([fZ[2,:] DeltaXZ[2,:]], label=["f(z)" "Δ(z,x)"])

# histogram(err_norm[1,:], title="Linearization error in y1", label=false)
# histogram(err_norm[2,:], title="Linearization error in y2", label=false)
# histogram(err_norm[3,:], title="Linearization error in y3", label=false)


# Gx, dGx = blurMatrix_nonlinear(4.0, x, hgt, wdth)
# plotImgVec(Gx, hgt, wdth, "Nonlinear Blur")
# z = O * x
# Q = Γ_x * O' * invΓ_z
# G_Qz, _ = blurMatrix_nonlinear(4.0, Q * z, hgt, wdth)
# y2 = G_Qz + dGx * (I - Q * O) * x + sqrt(σ_ϵ²) .* randn(length(img))

# # Calculate distribution X | Y
# μ_xy, Γ_xy = posterior_XgY(μ_x, invΓ_x, invΓ_ϵ, G)
# plotImgVec(μ_xy, hgt, wdth, "X (pos mean)")
# # # savefig("plots/mu_x_y.png")
# Γ_xy_nonlinear =  inv(dGx' * invΓ_ϵ * dGx + invΓ_x)
# μ_xy_nonlinear = Γ_xy_nonlinear * (invΓ_x * μ_x + dGx' * invΓ_ϵ * y)
# # plotImgVec(μ_xy_nonlinear, hgt, wdth, "X (pos mean) - nonlinear")
# L = cholesky(Γ_x * diagm(ones(numPix))).L
# trilEig = tril(L' * dGx' * invΓ_ϵ * dGx * L) + tril(L' * dGx' * invΓ_ϵ * dGx * L, -1)' #- diagm(diag(L' * dGx' * invΓ_ϵ * dGx * L))
# Λ_orig1, Q_orig1 = eigen(trilEig)
# Γ = Λ_orig1[end:-1:1]
# Plots.plot(Γ[1:1000]/Γ[1], label="Eigenvalues of dGx", yaxis=:log, dpi=300)

# # Low Rank Approximations of X | Y
# μ_xy_tilde, Γ_xy_tilde, eigs_XgY = posterior_XgY_lowRank(μ_x, Γ_x, invΓ_ϵ, G, 200)
# plotImgVec(μ_xy_tilde, hgt, wdth, "X (pos mean) - low rank")
# # savefig("plots/mu_x_y_rank200.png")

# # Calculate distribution Z | Y
# μ_zy, Γ_zy = posterior_ZgY(μ_z, invΓ_z, Γ_x, Γ_ϵ, G, O)
# μ_zy_from_xy = O * μ_xy
# plotImgVec(μ_zy, hgt_comp, wdth_comp, "Z (pos mean)")
# # savefig("plots/mu_z_y.png")

# # Low-Rank Approximations of Z | Y

# μ_zy_tilde, Γ_zy_tilde, eigs_ZgY = posterior_ZgY_lowRank(μ_z, invΓ_z, Γ_x, Γ_ϵ, G, O, 500)
# plotImgVec(μ_zy_tilde, hgt_comp, wdth_comp, "Z (pos mean) - low rank")
# # savefig("plots/mu_z_y_rank100.png")
# Plots.plot(eigs_ZgY[1:500]/eigs_ZgY[1], label="Z | Y (Goal-Oriented)", yaxis=:log, dpi=300)
# # Plots.plot!(eigs_ZgY_Hessian_conv[1:500]/eigs_ZgY_Hessian_conv[1], label="Z | Y (Goal-Oriented)", yaxis=:log, dpi=300)
# Plots.plot!(eigs_XgY[1:500]/eigs_XgY[1], label="X | Y")
# # plot!(Γ[1:256]/Γ[1], label="X | Y with dGx")
# # savefig("plots/eigspec_compare.png")


# # Compare eigenvalues of Hessian way and other way
# Γ_Δ = Γ_ϵ + G * (Γ_x - Γ_x * O' * invΓ_z * O * Γ_x) * G'
# Γ_Δ_hermitian = tril(Γ_Δ) + tril(Γ_Δ, -1)' 
# invΓ_Δ = inv(Γ_Δ_hermitian)
# μ_ZgY_Hessian, _, eigs_ZgY_Hessian = posterior_XgY_lowRank(μ_z, Γ_z, invΓ_Δ, G * Q, 500)
# eigs_ZgY_Hessian_conv = eigs_ZgY_Hessian ./ (1 .+ eigs_ZgY_Hessian)
# Plots.plot(eigs_ZgY[1:500], label="Z | Y (Goal-Oriented)", yaxis=:log, dpi=300)
# Plots.plot!(eigs_ZgY_Hessian_conv, label="Z | Y (Goal-Oriented)", yaxis=:log, dpi=300)

# plotImgVec(μ_ZgY_Hessian, hgt_comp, wdth_comp, "Z (pos mean)")
# plotImgVec(μ_zy_tilde, hgt_comp, wdth_comp, "Z (pos mean)")


