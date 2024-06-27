include("inverseProblem.jl")

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
sf = 3 # scaling factor
hgt_comp, wdth_comp = Int(floor(hgt/sf)), Int(floor(wdth/sf))
numPix_comp = hgt_comp * wdth_comp
O = goalMatrix(hgt_comp, wdth_comp, sf)
plotImgVec(O * x, hgt_comp, wdth_comp, "Z (truth)")
# savefig("plots/z_true.png")


# Inference parameters
μ_x = 0.5*ones(numPix)
μ_z = O * μ_x
σ_X², σ_Z² = 0.04, 0.04
σ_k² = 4.0
σ_ϵ² = 1e-4
Γ_x, Γ_ϵ = σ_X² * I, σ_ϵ² * I
Γ_z = O * Γ_x * O'
invΓ_x, invΓ_z, invΓ_ϵ = inv(Γ_x), inv(Γ_z), inv(Γ_ϵ)




# Apply forward model

G = blurMatrix_linear(σ_k², img)
y = G * x + sqrt(σ_ϵ²) .* randn(length(img))
plotImgVec(y, hgt, wdth, "Y")


# # savefig("plots/y.png")
# Gx, dGx = blurMatrix_nonlinear(4.0, img)
# plotImgVec(Gx, hgt, wdth, "Nonlinear Blur")
# heatmap(dGx)
# y2 = Gx + dGx * (I - Γ_x * O' * invΓ_z * O) * x + sqrt(σ_ϵ²) .* randn(length(img))
# plotImgVec(y2, hgt, wdth, "Y nonlinear")

# # evaluate at mu_x
# Gx_mu, dGx_mu = blurMatrix_nonlinear(4.0, reshape(μ_x, hgt, wdth))
# y3 = μ_x + dGx_mu * (x - μ_x) + sqrt(σ_ϵ²) .* randn(length(img))
# plotImgVec(y3, hgt, wdth, "Y nonlinear - Taylor")

# numSamp = 100
# dist = MvNormal(zeros(numPix_comp), Γ_z)
# sampsX =rand(dist, numSamp)
# sampRes = zeros(numPix, numSamp)
# for i in 1:numSamp
#     Gx, dGx = blurMatrix_nonlinear(4.0, img + reshape(sampsX[:,i], (hgt, wdth))) 
#     y2 = Gx + dGx * (I - Γ_x * O' * invΓ_z * O) * sampsX[:,i] + sqrt(σ_ϵ²) .* randn(length(img)) # this is wrong, it needs to be G(Qz)
#     sampRes[:,i] = y2 - Gx 
# end

# numSamp = 100
# dist = MvNormal(zeros(numPix), Γ_x)
# sampsX =rand(dist, numSamp)
# sampRes = zeros(numPix, numSamp)
# for i in 1:numSamp
#     y = G * x + sqrt(σ_ϵ²) .* randn(length(img))
#     sampRes[:,i] = y - G * x
# end


# Calculate distribution X | Y
μ_xy, Γ_xy = posterior_XgY(μ_x, invΓ_x, invΓ_ϵ, G)
plotImgVec(μ_xy, hgt, wdth, "X (pos mean)")
# savefig("plots/mu_x_y.png")
Γ_xy_nonlinear =  inv(dGx' * invΓ_ϵ * dGx + invΓ_x)
μ_xy_nonlinear = Γ_xy_nonlinear * (invΓ_x * μ_x + dGx' * invΓ_ϵ * y)
plotImgVec(μ_xy_nonlinear, hgt, wdth, "X (pos mean) - nonlinear")
L = cholesky(Γ_x * diagm(ones(numPix))).L
trilEig = tril(L' * dGx' * invΓ_ϵ * dGx * L) + tril(L' * dGx' * invΓ_ϵ * dGx * L)' - diagm(diag(L' * dGx' * invΓ_ϵ * dGx * L))
Λ_orig1, Q_orig1 = eigen(trilEig)
Γ = Λ_orig1[end:-1:1]
plot(Γ[1:256]/Γ[1], label="Eigenvalues of dGx", yaxis=:log, dpi=300)

# Low Rank Approximations of X | Y
μ_xy_tilde, Γ_xy_tilde, eigs_XgY = posterior_XgY_lowRank(μ_x, invΓ_x, invΓ_ϵ, G, 200)
plotImgVec(μ_xy_tilde, hgt, wdth, "X (pos mean) - low rank")
# savefig("plots/mu_x_y_rank200.png")

# Calculate distribution Z | Y
μ_zy, Γ_zy = posterior_ZgY(μ_z, invΓ_z, Γ_x, Γ_ϵ, G, O)
μ_zy_from_xy = O * μ_xy
plotImgVec(μ_zy, hgt_comp, wdth_comp, "Z (pos mean)")
# savefig("plots/mu_z_y.png")

# Low-Rank Approximations of Z | Y
μ_zy_tilde, Γ_zy_tilde, eigs_ZgY = posterior_ZgY_lowRank(μ_z, invΓ_z, Γ_x, Γ_ϵ, G, O, 100)
plotImgVec(μ_zy_tilde, hgt_comp, wdth_comp, "Z (pos mean) - low rank")
# savefig("plots/mu_z_y_rank100.png")
plot(eigs_ZgY[1:256]/eigs_ZgY[1], label="Z | Y (Goal-Oriented)", yaxis=:log, dpi=300)
plot!(eigs_XgY[1:256]/eigs_XgY[1], label="X | Y")
plot!(Γ[1:256]/Γ[1], label="X | Y with dGx")
# savefig("plots/eigspec_compare.png")

