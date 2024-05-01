include("mcmc.jl")
include("inverseProblem.jl")

# Finished April 21, 2024

## MCMC for X|Y - full rank
@time samps = mcmc_linear(μ_x, Γ_x, Γ_ϵ, G, y, 20, "NUTS")

## MCMC for Z|Y - full rank

# Load image and crop
raw_img = FileIO.load("images/masks.jpeg")
cropped_img = raw_img[250:450, 300:500]
gray_img = Gray.(cropped_img)
M = Float64.(gray_img)

# Scale down, remove colour
i1, i2 = size(M)
samples = 50;
inds1 = Int.(round.(collect(LinRange(1, i1, samples))))
inds2 = Int.(round.(collect(LinRange(1, i2, Int(round(samples * i2 / i1))))))
img = M[inds1,inds2]
hgt, wdth = size(img) 
x = img[:]
numPix = hgt*wdth
plotImgVec(x, hgt, wdth, "X (truth)")
# savefig("plots/x_true.png")


# Goal oriented - create linear transformation
# sf = 3 # scaling factor
# hgt_comp, wdth_comp = Int(floor(hgt/sf)), Int(floor(wdth/sf))
# numPix_comp = hgt_comp * wdth_comp
# O = goalMatrix(hgt_comp, wdth_comp, sf)'
crop_x, crop_y = 25:50, 1:20
hgt_comp, wdth_comp = length(crop_y), length(crop_x)
numPix_comp = hgt_comp * wdth_comp
O = goalMatrix_selection(hgt, wdth, crop_x, crop_y)
# plotImgVec(O * x, hgt_comp, wdth_comp, "Z (truth)")
# savefig("plots/z_true.png")

# Apply forward mode
G = blurMatrix_linear(σ_k², img)
y = G * x + sqrt(σ_ϵ²) .* randn(length(img))
# plotImgVec(y, hgt, wdth, "Y")

# Inference parameters
μ_x = 0.5*ones(numPix)
μ_z = O * μ_x
σ_X², σ_Z² = 0.04, 0.04
σ_k² = 4.0
σ_ϵ² = 1e-4
# Γ_x, Γ_ϵ = σ_X² * I, σ_ϵ² * I
Γ_x, Γ_ϵ = σ_X² * diagm(ones(numPix)), σ_ϵ² * diagm(ones(numPix))
Γ_z = O * Γ_x * O'
invΓ_x, invΓ_z, invΓ_ϵ = inv(Γ_x), inv(Γ_z), inv(Γ_ϵ)
Γ_Δ = Γ_ϵ + G * (Γ_x - Γ_x * O' * invΓ_z * O * Γ_x) * G'
Γ_Δ_hermitian = tril(Γ_Δ) + tril(Γ_Δ, -1)'
invΓ_Δ = inv(Γ_Δ_hermitian)

Q = Γ_x * O' * invΓ_z
F = G * Q



# r = 140

# L = cholesky(Γ_z).L
# LHL = L' * Q' * G' * invΓ_Δ * G * Q * L
# trilEig = tril(LHL) + tril(LHL, -1)'
# Λ_orig1, V_orig1 = eigen(trilEig)
# Λ = Λ_orig1[end:-1:1]
# V = V_orig1[:,end:-1:1]
# Plots.plot(Λ, yaxis=:log, dpi=300, legend=false)

# L2inv = inv(cholesky(Γ_Δ_hermitian).L)
# LHL2 = L2inv * G * Q * Γ_z * Q' * G' * L2inv'
# trilEig2 = tril(LHL2) + tril(LHL2, -1)'
# Λ_orig2, V_orig2 = eigen(trilEig2)
# V2 = V_orig2[:,end:-1:1]
# V2_r = V2[:,1:r]
# Φ2_r = L2inv' * V2_r

# Λ_r = Λ[1:r]
# V_r = V[:,1:r]
# Φ_r = L * V_r
# Φ_perp = L * V[:,r+1:end]

# Ξ_r = inv(L') * V_r

# # # my own implementation
# # z0 = μ_z
# # @time samps_lowrank_amm = mcmc_amm(z0, Γ_Δ, Φ_r, Ξ_r, G*Q, y, 100000, μ_z)
# # Nsamp = size(samps_lowrank_amm, 2)
# # pr_perp = MvNormal(zeros(numPix_comp - r), diagm(ones(numPix_comp - r)))
# # pr_perp_samp = rand(pr_perp, Nsamp)
# # samps_full = Φ_r * samps_lowrank_amm[:,:] + repeat(μ_z, 1, Nsamp) + Φ_perp * pr_perp_samp 

# # plotImgVec(dropdims(mean(samps_full[:,Int(Nsamp/5):end], dims=2), dims=2), hgt_comp, wdth_comp, "MCMC pos mean")

# μ_zgy_lin = μ_z + Φ_r * diagm(sqrt.(Λ_r) ./ (1 .+ Λ_r)) * Φ2_r' * y

# Γ_zgy_lin = Γ_z - Φ_r * diagm(Λ_r ./ (1 .+ Λ_r)) * Φ_r'
# μ_zgy_lin = Γ_zgy_lin * (invΓ_z * μ_z + Q' * G' * invΓ_Δ * y)


# plotImgVec(μ_zgy_lin, hgt_comp, wdth_comp, "Linear inversion, r = " * string(r))


r = 500
Λ, q, q_hat = posterior_ZgY_lowRank_eig(invΓ_z, Γ_x, Γ_ϵ, G, O)
q_tilde = Γ_x * O' * invΓ_z * O * Γ_x * G' * q
Φ_r = q_tilde[:,1:r]
Φ_perp = q_tilde[:,r+1:end]
Ξ_r = invΓ_x * Φ_r

 # my own implementation
x0 = μ_x
@time samps_lowrank_amm = mcmc_amm(x0, Γ_ϵ, Φ_r, Ξ_r, G, y, 30000, μ_x)
Nsamp = size(samps_lowrank_amm, 2)
pr_perp = MvNormal(zeros(numPix - r), diagm(ones(numPix - r)))
pr_perp_samp = rand(pr_perp, Nsamp)
samps_full = Φ_r * samps_lowrank_amm[:,:] + repeat(μ_x, 1, Nsamp) + Φ_perp * pr_perp_samp 


toPlot = extractGoal(dropdims(mean(samps_full[:,Int(Nsamp/5):end], dims=2), dims=2), hgt, wdth, crop_y, crop_x);
plotImgVec(toPlot, hgt_comp, wdth_comp, "MCMC pos mean")


gx, dgx = blurMatrix_nonlinear2(9, 1, x, hgt, wdth);
@time gqz, dgz = blurMatrix_nonlinear2(9, 1, Q*z, hgt, wdth);
plotImgVec(gqz, hgt, wdth, "Test1")
plotImgVec(dgz * (I - Q * O) * x , hgt, wdth, "Test2")
plotImgVec(gx - gqz - dgz * (I - Q * O) * x , hgt, wdth, "TestDiff")
