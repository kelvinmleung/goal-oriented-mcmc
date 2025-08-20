using LinearAlgebra
using Statistics
using Plots, StatsPlots
using LaTeXStrings
using Distributions
using Random
using TransportBasedInference
# include("../src/goalorientedtransport.jl")

Random.seed!(1234)


function g(x)
    x1, x2, x3 = x[1], x[2], x[3]
    return [x1^2 - x2^2; x1^2 + x2^2 + x3^2; x1*x3]
end

function ∇g(x)
    x1, x2, x3 = x[1], x[2], x[3]
    gradg = [2*x1 -2*x2 0; 2*x1 2*x2 2*x3; x3 0 x1]
    return gradg
end

# function g(x)
#     x1, x2, x3 = x[1], x[2], x[3]
#     return [x1 + x2; x2 + x3; x3 + x1]
# end

# function ∇g(x)
#     x1, x2, x3 = x[1], x[2], x[3]
#     gradg = [1 1 0; 0 1 1; 1 0 1]
#     return gradg
# end


function diagnostic_matrix(N::Int, invsqrtΓ)
    # N = 10000
    H = zeros((n,n))
    println("Monte Carlo estimate of diagnostic matrix...")
    randx = rand(MvNormal(μ_x, Γ_x), N)
    dfx = map(∇g, eachcol(randx)) 
    # dfx = map(randx -> gradg(randx, g(randx)), eachcol(randx))  # Compute gra
    @time for i = 1:N
        dfx_i = dfx[i]  # Gradient for the i-th sample
        H += (1 / N) * invsqrtΓ * dfx_i * Γ_x * O' * invΓ_z * O * Γ_x * dfx_i' * invsqrtΓ
    end
    H
end

function apply_cond_gaussian(X::AbstractMatrix{T}, yobs::AbstractVector{T}, n::Int) where T <: Real #, V, setup, μ_y
    jointmean = mean(X, dims=2)
    jointcov = cov(X, dims=2)
    μ_y = @view jointmean[1:n]
    μ_z = @view jointmean[n+1:end]
    Σ_zz = @view jointcov[n+1:end, n+1:end]
    Σ_yz = @view jointcov[1:n, n+1:end]

    invΣ_yy = inv(jointcov[1:n, 1:n])
    μ_zgy = μ_z .- Σ_yz' * invΣ_yy * (μ_y - yobs)
    display(μ_z)
    display(Σ_yz' * invΣ_yy * (μ_y - yobs))
    Σ_zgy = Σ_zz .- Σ_yz' * invΣ_yy * Σ_yz
    
    μ_zgy, Σ_zgy
end

function apply_cond_transport(X::AbstractMatrix{T}, Ystar::AbstractMatrix{T}, Ny::Int; order::Int = 10) where T <: Real
    S = HermiteMap(order, X; diag = true, factor = 0.01, α = 1e-6, b = "ProHermiteBasis");
    @time S = optimize(S, X, "split"; maxterms = 30, withconstant = true, withqr = true, verbose = true, 
                                  maxpatience = 30, start = 1, hessprecond = true)
    
    F = evaluate(S, X; start = Ny+1)
    Xa = deepcopy(X)
    @time hybridinverse!(Xa, F, S, Ystar; apply_rescaling = true, start = Ny+1)
    Xa[Ny+1:end,:], S, F
end


m = 3
n = 3
p = 2
N = 10000





# O = [1/sqrt(3) -1/sqrt(3) 1/sqrt(3); 1/sqrt(2) 0 -1/sqrt(2)] #orthonormal basis
O = [1 1 0; 1 -1 0] / sqrt(2)
# O = diagm(ones(3))
μ_x, Γ_x = zeros(n), diagm(ones(n))
μ_z, Γ_z = O * μ_x, O * Γ_x * O'
invΓ_z = inv(Γ_z)
Γ_ϵ = diagm(ones(m)) * 0.01

x_true = rand(MultivariateNormal(μ_x, Γ_x)) .-1
z_true = O * x_true
y_obs = g(x_true) + rand(MultivariateNormal(zeros(m), Γ_ϵ))

x_samp = rand(MultivariateNormal(μ_x, Γ_x), N)
z_samp = O* x_samp
ϵ_samp = rand(MultivariateNormal(zeros(m), Γ_ϵ), N)
y_samp = zeros(m, N)
for i in 1:N
    y_samp[:,i] = g(x_samp[:,i]) + ϵ_samp[:,i]
end

meany = mean(y_samp,dims=2)
covy = cov(y_samp, dims=2) 
invsqrtcovy = inv(sqrt(covy))
invsqrtcovϵ = inv(sqrt(Γ_ϵ))

H_ϵ = diagnostic_matrix(N, invsqrtcovϵ)
H_y = diagnostic_matrix(N, invsqrtcovy)

eigen(H_ϵ)
eigvals(H_y)




## TRANSPORT DENSITY PLOTS
r = 2 #energy_cutoff(eigvals(H_y), 0.99) 
# V_r = eigvecs(H_ϵ)[:,end:-1:1][:,1:r]
V_r = eigvecs(H_y)[:,end:-1:1][:,1:r]


X = vcat(V_r' * (y_samp .- meany), sqrt(invΓ_z) * (z_samp .- μ_z))[:,1:N]
yobs_whiten = repeat(V_r' * (y_obs - meany), 1, N)

# z_possamp_whiten, S, F = apply_cond_transport(X, yobs_whiten, r; order=10)
# z_possamp_transport =  sqrt(Γ_z) * z_possamp_whiten .+ μ_z 

μ_gauss, Σ_gauss = apply_cond_gaussian(vcat(y_samp, z_samp), vec(y_obs), m)
# μ_gauss, Σ_gauss = apply_cond_gaussian(X, vec(V_r' * (y_obs - meany)), r)
Σ_gauss = tril(Σ_gauss) + tril(Σ_gauss, -1)'
z_possamp_gauss =  sqrt(Γ_z) * rand(MultivariateNormal(μ_gauss, Σ_gauss), N) .+ μ_z 



plot(xlabel=L"$z_1$", title="Component 1")
plot!([z_true[1]], seriestype="vline", color=:black, linestyle=:dot,linewidth=3, label="Truth")
density!(z_samp[1,:], label="Prior")
# density!(z_possamp_transport[1,:], label="Posterior")
density!(z_possamp_gauss[1,:], label="Posterior - Gauss")


