include("mcmc.jl")
include("inverseProblem.jl")

Random.seed!(123)

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
crop_x, crop_y = 25:50, 1:20
hgt_comp, wdth_comp = length(crop_y), length(crop_x)
numPix_comp = hgt_comp * wdth_comp
O = goalMatrix_selection(hgt, wdth, crop_x, crop_y)
gx, ~ = blurMatrix_nonlinear2(9, 1, x, hgt, wdth)


# @time gx1, dGx1 = blurMatrix_nonlinear2(4, 0.01, x, hgt, wdth);
# @time gx2, dGx2 = blurMatrix_nonlinear2_fast(9, 1, x, hgt, wdth);



# Inference parameters
μ_x = 0.5*ones(numPix)
μ_z = O * μ_x
σ_X², σ_Z² = 0.04, 0.04
# σ_k² = 4.0
σ_ϵ² = 1e-4

Γ_ϵ = σ_ϵ² * I
Γ_x = σ_X² * construct_prior(hgt, wdth, 1)
Γ_z = O * Γ_x * O'
invΓ_x, invΓ_z, invΓ_ϵ = inv(Γ_x), inv(Γ_z), inv(Γ_ϵ)

# Q = Γ_x * O' * invΓ_z

priorHessian(μ_x, Γ_x, Γ_ϵ, O, 1000)
@load "mcmc/priorHessian.jld" # x_pr, H


y = gx + sqrt(σ_ϵ²) .* randn(numPix)

@time x_all = mcmc_lis_nonlinear(μ_x, Γ_x, Γ_ϵ, O, y; m=1000, epoch=1);
z_all = O * x_all
plotImgVec(mean(z_all[:,1001:end], dims=2), hgt_comp, wdth_comp, "MCMC posterior mean")






# toPlot = extractGoal(dropdims(mean(samps_full[:,Int(Nsamp/5):end], dims=2), dims=2), hgt, wdth, crop_y, crop_x);
# plotImgVec(toPlot, hgt_comp, wdth_comp, "MCMC pos mean")


# gx, dgx = blurMatrix_nonlinear2(9, 1, x, hgt, wdth);
# @time gqz, dgz = blurMatrix_nonlinear2(9, 1, Q*z, hgt, wdth);
# plotImgVec(gqz, hgt, wdth, "Test1")
# plotImgVec(dgz * (I - Q * O) * x , hgt, wdth, "Test2")
# plotImgVec(gx - gqz - dgz * (I - Q * O) * x , hgt, wdth, "TestDiff")
