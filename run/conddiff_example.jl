using LinearAlgebra
using Distributions
using Random
using Plots, StatsPlots
using LaTeXStrings
using TransportBasedInference

mutable struct CondDiffModel
    beta::Float64
    T::Float64
    n::Int
    k::Int
    dt::Float64
    H::Matrix{Float64}
end

function CondDiffModel(; beta=10.0, T=1.0, n=100, k=1)
    dt = T / n
    m = div(n, k)
    H = zeros(m, n + 1)
    for i in 1:m
        H[i, i * k] = 1.0
    end
    return CondDiffModel(beta, T, n, k, dt, H)
end

function name(model::CondDiffModel)
    return "conddiff"
end

function d(model::CondDiffModel)
    return model.n
end

function m(model::CondDiffModel)
    return div(model.n, model.k)
end

function integrate_model(model::CondDiffModel, w::Vector{Float64})
    u = zeros(model.n + 1)
    for i in 1:model.n
        f = model.beta * u[i] * (1 - u[i]^2) / (1 + u[i]^2)
        u[i+1] = u[i] + model.dt * f + sqrt(model.dt) * w[i]
        # α=0.5
        # f(u) = -α * u[i]
        # u[i+1] = (1 - α * model.dt) * u[i] + sqrt(model.dt) * w[i]
    end
    return u
end

function compute_G(model::CondDiffModel, w::Vector{Float64})
    u = integrate_model(model, w)
    return model.H * u
end

function compute_gradG(model::CondDiffModel, w::Vector{Float64}; states::Bool=false)
    u = integrate_model(model, w)
    N = model.n + 1
    tmp = u.^2
    dfdu = model.beta * ((1 .- 3 .* tmp) ./ (1 .+ tmp) .- 2 .* tmp .* (1 .- tmp) ./ (1 .+ tmp).^2)
    off_diag = -1 .- dfdu[1:end-1] .* model.dt
    tmp_mat = Matrix{Float64}(I, N, N)
    for i in 2:N
        tmp_mat[i, i-1] = off_diag[i-1]
    end
    dudw = sqrt(model.dt) .* inv(tmp_mat)[:, 2:end]
    gradG = model.H * dudw
    return states ? (u, gradG) : gradG
end


function diagnostic_matrix(N::Int, model, invsqrtΓ, O, n)
    # N = 10000
    H = zeros((n,n))
    println("Monte Carlo estimate of diagnostic matrix...")
    randx = rand(MvNormal(μ_x, Γ_x), N)'
    # dfx = map(∇g, eachcol(randx)) 
    # dfx = map(randx -> gradg(randx, g(randx)), eachcol(randx))  # Compute gra
    # dfx = map(randx -> compute_gradG(model, randx), eachcol(randx))  
    @time for i = 1:N
        # dfx_i = dfx[i]  # Gradient for the i-th sample
        dfx_i = compute_gradG(model, randx[i,:])
        H += (1 / N) * invsqrtΓ * dfx_i * Γ_x * O' * O * Γ_x * dfx_i' * invsqrtΓ
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
    # display(μ_z)
    Σ_zgy = Σ_zz .- Σ_yz' * invΣ_yy * Σ_yz
    
    μ_zgy, Σ_zgy
end

function apply_cond_transport(X::AbstractMatrix{T}, Ystar::AbstractMatrix{T}, Ny::Int; order::Int = 10) where T <: Real
    # S = HermiteMap(order, X; diag = true, factor = 0.5, α = 1e-6, b = "ProHermiteBasis");
    S = HermiteMap(order, X; diag = true, factor = 1., α = 1e-6, b = "ProHermiteBasis");
    @time S = optimize(S, X, "split"; maxterms = 5000, withconstant = true, withqr = true, verbose = true, 
                                  maxpatience = 3000, start = 1, hessprecond = true)
    
    F = evaluate(S, X; start = Ny+1)
    Xa = deepcopy(X)
    @time hybridinverse!(Xa, F, S, Ystar; apply_rescaling = true, start = Ny+1)
    Xa[Ny+1:end,:], S, F
end

function energy_cutoff(Λy::AbstractVector{T}, ratio::T; rydefault = 50) where T <: Real
    ry = ratio < 1.0 ? findfirst(x-> x >= ratio, cumsum(Λy)./sum(Λy)) : rydefault
    ry = isnothing(ry) ? 1 : ry 
end

# function plot_eig_computations(n)
#     H = diagnostic_matrix(Nsamp, model, invsqrtΓ, U_r', m(model))
#     H_ϵ = diagnostic_matrix(Nsamp, model, invsqrtcovϵ, U_r', m(model))
#     H_y = diagnostic_matrix(Nsamp, model, invsqrtcovy, U_r', m(model))
#     plt = plot(xlabel=L"$s$", ylabel=L"$\lambda_{max}(\Gamma, \Gamma_\epsilon) \, \sum_{i=1}^s \lambda_i(H,\Gamma)$", yaxis=:log, dpi=300, size=(600,400))
#     plot!(1:n, eigvals(H)[end:-1:1][1:n] * eigvals(inv(Γ_ϵ))[end], linewidth=2, label="No Whitening")
#     # plot!(1:100, eigvals(H_ϵ)[end:-1:1][1:100] * eigvals(Γ_ϵ)[end], label=L"$\Gamma_\epsilon$")
#     plot!(1:n, eigvals(H_ϵ)[end:-1:1][1:n] , linewidth=2, label=L"Whiten with $\Gamma_\epsilon$")
#     plot!(1:n, eigvals(H_y)[end:-1:1][1:n] * eigvals(inv(Γ_ϵ)* covy)[end],linewidth=2,  label=L"Whiten with $\Gamma_y$")
#     display(plt)
# end

# function mcmc(μ_pr, Γ_pr, Γ_obs, y, N, samp_factor=1) 

function mcmc_amm_simple(x0, μ_pr, Γ_pr, Γ_obs, y, N)

    function logpos_simple(x, μ_pr, invΓ_pr, invΓ_obs, y)
        gx = compute_G(model,x)
        # gx = aoe_fwdfun(x)
        -1/2 * (x - μ_pr)' * invΓ_pr * (x - μ_pr) - 1/2 * (y - gx)' * invΓ_obs * (y - gx)
    end

    function proposal(μ, cholΓ)
        n = length(μ)
        mvn = MvNormal(zeros(n), diagm(ones(n)))
        μ + cholΓ * rand(mvn)
    end

    function proposal_1d(μ, σ)
        μ + σ * randn(1)[1]
    end


    function alpha_simple(x, z, μ_pr, invΓ_x, invΓ_obs, y)
        lpz = logpos_simple(z, μ_pr, invΓ_x, invΓ_obs, y)
        lpx = logpos_simple(x, μ_pr, invΓ_x, invΓ_obs, y)
        return minimum((1, exp(lpz-lpx))), lpz, lpx
    end

    r = length(μ_pr)
    x_vals = zeros(r, N)
    invΓ_pr, invΓ_obs = inv(Γ_pr), inv(Γ_obs)
    logpos, accept = zeros(N), zeros(N)
    propcov = (2.38^2) / r * Γ_pr #diagm(ones(r))
    propChol = cholesky(propcov).L 
    sd, eps = 2.38^2 / r, 1e-10
    meanXprev = zeros(r)

    x = x0 
    for i in 1:N
        z = proposal(x, propChol)
        # z = rand(MvNormal(x, propcov))

        α, lpz, lpx = alpha_simple(x, z, μ_pr, invΓ_pr, invΓ_obs, y)
        if rand(Uniform(0,1)) < α
            x, lpx = z, lpz
            accept[i] = 1
        end
        x_vals[:,i] = x
        logpos[i] = lpx
        
        if i % 500 == 0
            if i % 10000 == 0
                display("Sample: " * string(i))
                display("   Accept Rate: " * string(mean(accept[i-9999:i])))
            end
            propChol = cholesky(tril(propcov) + tril(propcov,-1)').L
        end

        if i == 1500
            propcov = sd * cov(x_vals[:,1:1500], dims=2) + eps * I
            meanXprev = mean(x_vals[:,1:1500], dims=2)
        elseif i > 1500
            meanX = i / (i + 1) * meanXprev + 1 / (i + 1) * x_vals[:,i]
            propcov = (i-1) / i * propcov + sd / i * (i * meanXprev * meanXprev' - (i+1) * meanX * meanX' + x_vals[:,i] * x_vals[:,i]' + eps * I)
            meanXprev = meanX
        end
    end
    # display(plot(logpos[Int(N/10):end]))
    return x_vals
end










Random.seed!(1234)
model = CondDiffModel(beta=10.0, T=1.0, n=10, k=1) #MAKE N SMALLER, 40, SO WE CAN RUN MCMC
# model = CondDiffModel(beta=0.02, T=1.0, n=100, k=1)




# w = randn(model.n) # X ~ N(0,I)
# u = CondDiff.integrate_model(model, w)
# G = CondDiff.compute_G(model, w)


Nsamp = 20000

μ_x, Γ_x = zeros(model.n), diagm(ones(model.n))
# Γ_ϵ = diagm(ones(m(model))) * 0.1
# Γ_ϵ = diagm(rand(Uniform(0.0004,0.01), m(model))) # abs.(randn(m(model)))) * 0.1
Γ_ϵ = diagm(rand(Uniform(0.01,0.1), m(model))) 
# A = randn(m(model), m(model))
# Γ_ϵ = (A*A' + 1e-10 *I) / 1e3

# plot(diag(Γ_ϵ), ylim=(0,0.0101), xlabel=L"$t$", ylabel=L"$\sigma_i^2$", color=:navy, linewidth=2, dpi=300, label=false)
# savefig("plots/07192025_conddiff/cd_noiseparam.pdf")


# truth-obs pair
x_true = randn(model.n)
y_obs = compute_G(model, x_true) + rand(MultivariateNormal(zeros(m(model)), Γ_ϵ))

# prior samples
# x_samp = randn(Nsamp, model.n)
x_samp = rand(MultivariateNormal(μ_x, Γ_x), Nsamp)
ϵ_samp = rand(MultivariateNormal(zeros(m(model)), Γ_ϵ), Nsamp) #randn(Nsamp, m(model))
y_samp = zeros(m(model), Nsamp)
u_samp = zeros(model.n+1, Nsamp)
for i in 1:Nsamp
    u_samp[:,i] = integrate_model(model, x_samp[:,i])
    y_samp[:,i] = compute_G(model, x_samp[:,i]) + ϵ_samp[:,i]
end

plt = plot(ylabel=L"$u$", xlabel=L"$t$", dpi=300)
t_plot = range(0, stop=1, length=model.n+1)
for i in 1:100
    plot!(t_plot, u_samp[:,i], color=:navy, alpha=0.2, label=false)
end
plot!(t_plot, integrate_model(model, x_true), color=:red, linewidth=2, label="True Trajectory")
display(plt)
savefig("plots/07192025_conddiff/cd_traj_wtruth.pdf")


# get some eigendirections to project it
H_forU = zeros((model.n, model.n))
for i = 1:10000
    dfx_i = compute_gradG(model, x_samp[:,i])
    H_forU += (1 / Nsamp) * dfx_i' * dfx_i
end
numProj = 1
# eigval, eigvec = eigen(cov(y_samp, dims=2) )
eigval, eigvec = eigen(H_forU)
# U_r = eigvec[:,end:-1:end-numProj+1]
# U_r = [vcat(zeros(50), ones(10), zeros(40)) vcat(zeros(60), ones(10), zeros(30)) vcat(zeros(70), ones(10), zeros(20)) vcat(zeros(80), ones(10), zeros(10)) vcat(zeros(90), ones(10))] / sqrt(10)
# U_r = [vcat(ones(10), zeros(30)) vcat(zeros(10), ones(10), zeros(20)) vcat(zeros(20), ones(10), zeros(10)) vcat(zeros(30), ones(10))] / sqrt(10)
# U_r = [vcat(ones(20), zeros(80)) vcat(zeros(20), ones(20), zeros(60)) vcat(zeros(40), ones(20), zeros(40)) vcat(zeros(60), ones(20), zeros(20)) vcat(zeros(80), ones(20))] / sqrt(20)
# U_r = reshape(collect(1:1:100) / norm(collect(1:1:100)), 100, 1)
U_r = ones(model.n,1) / sqrt(10)



plt = plot(xlabel=L"$t$", dpi=300)
# custom_colors = [:crimson, :seagreen, :navy, :orange3, :purple]
t_plot = range(0, stop=1, length=model.n+1)
for i in 1:numProj
    plot!(t_plot[2:end], U_r[:,i], linewidth=2, label=latexstring("O_{", string(i), "}")) #color=custom_colors[i]
end
display(plt)
savefig("plots/07192025_conddiff/cd_goalop.pdf")


μ_z, Γ_z = U_r' * μ_x, U_r' * Γ_x * U_r
z_samp = U_r' * x_samp
z_true = U_r' * x_true
invΓ_z = inv(Γ_z)


meany = mean(y_samp,dims=2)
covy = cov(y_samp, dims=2)
invsqrtcovy = inv(sqrt(covy))
invsqrtcovϵ = inv(sqrt(Γ_ϵ))







invsqrtΓ = I
H = diagnostic_matrix(Nsamp, model, I, U_r', m(model))
H_ngo = diagnostic_matrix(Nsamp, model, I, I, m(model))
H_ϵ = diagnostic_matrix(Nsamp, model, invsqrtcovϵ, U_r', m(model))
H_y = diagnostic_matrix(Nsamp, model, invsqrtcovy, U_r', m(model))
n = model.n-1
# n = 30


valforH, valforHϵ, valforHy  = zeros(n), zeros(n), zeros(n)
valforH[1] = eigvals(inv(Γ_ϵ))[end] * (sum(eigvals(H_ngo)) - eigvals(H)[end:-1:1][1])
valforHϵ[1] = sum(eigvals(invsqrtcovϵ * H_ngo * invsqrtcovϵ)) - real.(eigvals(H_ϵ))[end:-1:1][1]
valforHy[1] = eigvals( covy * inv(Γ_ϵ))[end] * (sum(eigvals(invsqrtcovy * H_ngo * invsqrtcovy)) - real.(eigvals(H_y))[end:-1:1][1])
for i in 2:n
    valforH[i] = valforH[i-1] - eigvals(inv(Γ_ϵ))[end] * eigvals(H)[end:-1:1][i]
    valforHϵ[i] = valforHϵ[i-1] -real.(eigvals(H_ϵ))[end:-1:1][i]
    valforHy[i] = valforHy[i-1] - eigvals( covy * inv(Γ_ϵ))[end] * real.(eigvals(H_y))[end:-1:1][i]
end
plt = plot(xlabel=L"$r$", ylabel="Upper Bound", dpi=300, size=(500,400), ylims=(0,1.5e4))# yaxis=:log,
plot!(1:n, valforH, linewidth=2, color=:crimson, label=L"$\Gamma = I$")
plot!(1:n, valforHϵ, linewidth=2, color=:blue2, label=L"$\Gamma = \Gamma_\epsilon$")
plot!(1:n, valforHy,linewidth=2, color=:green4, label=L"$\Gamma = \Gamma_Y$")
display(plt)
savefig("plots/07192025_conddiff/cd_upperbounds.pdf")

# Here, gamma_epsilon has the lowest bound so it's the closest to the posterior.... !!!! 



# # plt = plot(xlabel=L"$s$", ylabel=L"$\lambda_{max}(\Gamma, \Gamma_\epsilon) \, \sum_{i=1}^s \lambda_i(H,\Gamma)$", yaxis=:log, dpi=300, size=(600,400))
# plt = plot(xlabel=L"$r$", ylabel=L"$\lambda_{max}(\Gamma, \Gamma_\epsilon) \, \lambda_i(H,\Gamma)$", yaxis=:log, dpi=300, size=(600,400))
# plot!(1:n, eigvals(H)[end:-1:1][1:n] * eigvals(inv(Γ_ϵ))[end], linewidth=2, color=:crimson, label=L"$\Gamma = I$")
# # plot!(1:100, eigvals(H_ϵ)[end:-1:1][1:100] * eigvals(Γ_ϵ)[end], label=L"$\Gamma_\epsilon$")
# plot!(1:n, real.(eigvals(H_ϵ))[end:-1:1][1:n] , linewidth=2, color=:blue2, label=L"$\Gamma = \Gamma_\epsilon$")
# plot!(1:n, real.(eigvals(H_y))[end:-1:1][1:n] * eigvals(covy * inv(Γ_ϵ))[end],linewidth=2, color=:green4, label=L"$\Gamma = \Gamma_Y$")
# display(plt)
# # savefig("plots/07192025_conddiff/cd_lambdastar.pdf")
# # eigH = eigvals(H)[end:-1:1]
# # plot(eigH[1:end-1], title="Eigenvalues of H", yaxis=:log)

# plot_eig_computations(m(model)-1)

# λstar, λstar_ϵ, λstar_y = zeros(n), zeros(n), zeros(n)
# for i in 1:n
#     λstar[i] = sum(eigvals(H)[end:-1:1][1:i]) * eigvals(inv(Γ_ϵ))[end]
#     λstar_ϵ[i] = sum(real.(eigvals(H_ϵ))[end:-1:1][1:n])
#     λstar_y[i] = sum(real.(eigvals(H_y))[end:-1:1][1:n]) * eigvals( covy * inv(Γ_ϵ))[end]
# end
# # plt = plot(xlabel=L"$s$", ylabel=L"$\lambda_{max}(\Gamma, \Gamma_\epsilon) \, \sum_{i=1}^s \lambda_i(H,\Gamma)$", yaxis=:log, dpi=300, size=(600,400))
# plt = plot(xlabel=L"$r$", ylabel=L"$\lambda^*(\Gamma)$", yaxis=:log, dpi=300, size=(600,400))
# plot!(1:n, λstar, linewidth=2, color=:crimson, label=L"$\Gamma = I$")
# # plot!(1:100, eigvals(H_ϵ)[end:-1:1][1:100] * eigvals(Γ_ϵ)[end], label=L"$\Gamma_\epsilon$")
# plot!(1:n, λstar_ϵ, linewidth=2, color=:blue2, label=L"$\Gamma = \Gamma_\epsilon$")
# plot!(1:n, λstar_y, linewidth=2, color=:green4, label=L"$\Gamma = \Gamma_Y$")
# display(plt)
# savefig("plots/07192025_conddiff/cd_lambdastar.pdf")



n = model.n-1
plt = plot(xlabel=L"$r$", ylabel="Normalized Eigenvalue", yaxis=:log, dpi=300, size=(500,400))
plot!(1:n, eigvals(H)[end:-1:1][1:n] / eigvals(H)[end], linewidth=2, color=:crimson, label=L"$\Gamma = I$")
plot!(1:n, eigvals(H_ϵ)[end:-1:1][1:n] / eigvals(H_ϵ)[end] , linewidth=2, color=:blue2, label=L"$\Gamma = \Gamma_\epsilon$")
plot!(1:n, eigvals(H_y)[end:-1:1][1:n] / eigvals(H_y)[end],linewidth=2, color=:green4, label=L"$\Gamma = \Gamma_Y$")
# plot!(1:n, eigvals(H_y)[end:-1:1][1:n],linewidth=2,  label=false)
display(plt)
savefig("plots/07192025_conddiff/cd_eigvals.pdf")



@time x_mcmc = mcmc_amm_simple(μ_x, μ_x, Γ_x, Γ_ϵ, y_obs, 1000000)
z_mcmc = U_r' * x_mcmc#[:,200001:end]



## TRANSPORT DENSITY PLOTS
# r = energy_cutoff(eigvals(H_ϵ)[end:-1:1], 0.999) 
# r = model.n
r = 5
# Ntrans = 2000


# V_r = eigvecs(H)[:,end:-1:1][:,1:r]
# X = vcat(V_r' * (y_samp .- meany), sqrt(invΓ_z) * (z_samp .- μ_z))[:,1:Nsamp]
# yobs_whiten = repeat(V_r' * (y_obs - meany), 1, Nsamp)
# z_possamp_whiten, S, F = apply_cond_transport(X, yobs_whiten, r; order=10)
# z_possamp_transport_H =  sqrt(Γ_z) * z_possamp_whiten .+ μ_z 
# # μ_gauss, Σ_gauss = apply_cond_gaussian(X, vec(V_r' * (y_obs - meany)), r)
# # Σ_gauss = tril(Σ_gauss) + tril(Σ_gauss, -1)'
# # z_possamp_gauss_H =  sqrt(Γ_z) * rand(MultivariateNormal(μ_gauss, Σ_gauss), Nsamp) .+ μ_z 

# V_r = invsqrtcovϵ * real.(eigvecs(H_ϵ))[:,end:-1:1][:,1:r]
# X = vcat(V_r' * (y_samp .- meany), sqrt(invΓ_z) * (z_samp .- μ_z))[:,1:Nsamp]
# for i in 1:r
#     X[i,:] = X[i,:] / std(X[i,:])
# end
# yobs_whiten = repeat(V_r' * (y_obs - meany), 1, Nsamp)
# z_possamp_whiten, S, F = apply_cond_transport(X, yobs_whiten, r; order=50)
# z_possamp_transport_Hϵ =  sqrt(Γ_z) * z_possamp_whiten .+ μ_z 
# # μ_gauss, Σ_gauss = apply_cond_gaussian(X, vec(V_r' * (y_obs - meany)), r)
# # Σ_gauss = tril(Σ_gauss) + tril(Σ_gauss, -1)'
# # z_possamp_gauss_Hϵ =  sqrt(Γ_z) * rand(MultivariateNormal(μ_gauss, Σ_gauss), Nsamp) .+ μ_z 

# V_r = invsqrtcovy * real.(eigvecs(H_y))[:,end:-1:1][:,1:r]
# X = vcat(V_r' * (y_samp .- meany), sqrt(invΓ_z) * (z_samp .- μ_z))[:,1:Nsamp]
# for i in 1:r
#     X[i,:] = X[i,:] / std(X[i,:])
# end
# yobs_whiten = repeat(V_r' * (y_obs - meany), 1, Nsamp)
# z_possamp_whiten, S, F = apply_cond_transport(X, yobs_whiten, r; order=50)
# z_possamp_transport_Hy =  sqrt(Γ_z) * z_possamp_whiten .+ μ_z 
# # μ_gauss, Σ_gauss = apply_cond_gaussian(X, vec(V_r' * (y_obs - meany)), r)
# # Σ_gauss = tril(Σ_gauss) + tril(Σ_gauss, -1)'
# # z_possamp_gauss_Hy =  sqrt(Γ_z) * rand(MultivariateNormal(μ_gauss, Σ_gauss), Nsamp) .+ μ_z 

# for i in 1:r
#     display(histogram(X[i,:]))
# end

# # NO TRANSFORM
# X = vcat(invsqrtcovy * (y_samp .- meany), z_samp)[:,1:Nsamp]
# yobs_whiten = repeat(invsqrtcovy * (y_obs - meany), 1, Nsamp)
# z_possamp_transport, S, F = apply_cond_transport(X, yobs_whiten, r; order=10)

# # Full parameters
# X = vcat(invsqrtcovy * (y_samp .- meany), x_samp)
# yobs_whiten = repeat(invsqrtcovy * (y_obs - meany), 1, Nsamp)
# x_possamp_transport, S, F = apply_cond_transport(X, yobs_whiten, r; order=30)
# # z_possamp_transport_full = U_r' * x_possamp_transport

# X = vcat((y_samp .- meany), x_samp)
# for i in 1:r
#     X[i,:] = X[i,:] / std(X[i,:])
# end
# yobs_whiten = repeat((y_obs - meany) ./ sqrt.(diag(covy)), 1, Nsamp)
# x_possamp_transport_2, S, F = apply_cond_transport(X, yobs_whiten, r; order=30)
# # z_possamp_transport_full_2 = U_r' * x_possamp_transport_2

# # μ_gauss, Σ_gauss = apply_cond_gaussian(vcat(y_samp, z_samp), vec(y_obs), m(model))
# # # μ_gauss, Σ_gauss = apply_cond_gaussian(X, vec(V_r' * (y_obs - meany)), r)
# # Σ_gauss = tril(Σ_gauss) + tril(Σ_gauss, -1)'
# z_possamp_gauss =  sqrt(Γ_z) * rand(MultivariateNormal(μ_gauss, Σ_gauss), Nsamp) .+ μ_z 


maporder = 100

V_r = eigvecs(H)[:,end:-1:1][:,1:r]
X = vcat(V_r' * (y_samp .- meany), z_samp)
yobs_whiten = repeat(V_r' * (y_obs - meany), 1, Nsamp)
# for i in 1:r
#     X[i,:] = (X[i,:] .- mean(X[i,:])) / std(X[i,:])
#     yobs_whiten[i,:] = (yobs_whiten[i,:] .- mean(X[i,:])) / std(X[i,:])
# end
z_possamp_transport_H, S, F = apply_cond_transport(X, yobs_whiten, r; order=maporder)

V_r = invsqrtcovϵ * real.(eigvecs(H_ϵ))[:,end:-1:1][:,1:r]
X = vcat(V_r' * (y_samp .- meany), z_samp)
yobs_whiten = repeat(V_r' * (y_obs - meany), 1, Nsamp)
# for i in 1:r
#     X[i,:] = (X[i,:] .- mean(X[i,:])) / std(X[i,:])
#     yobs_whiten[i,:] = (yobs_whiten[i,:] .- mean(X[i,:])) / std(X[i,:])
# end
z_possamp_transport_Hϵ, S, F = apply_cond_transport(X, yobs_whiten, r; order=maporder)

V_r = invsqrtcovy * real.(eigvecs(H_y))[:,end:-1:1][:,1:r]
X = vcat(V_r' * (y_samp .- meany), z_samp)
yobs_whiten = repeat(V_r' * (y_obs - meany), 1, Nsamp)
# for i in 1:r
#     X[i,:] = (X[i,:] .- mean(X[i,:])) / std(X[i,:])
#     yobs_whiten[i,:] = (yobs_whiten[i,:] .- mean(X[i,:])) / std(X[i,:])
# end
z_possamp_transport_Hy, S, F = apply_cond_transport(X, yobs_whiten, r; order=maporder)

plotrange = -3:0.1:3
kde_pr = pdf.(Normal(0, 1), plotrange)

# for ind in 1:model.n
#     plt = plot(xlabel=L"$z$", title="Component " * string(ind), xlim = (-3,3), dpi=300) #, size=(400,300), ylim=(0, 1)
#     plot!([x_true[ind]], seriestype="vline", color=:black, linestyle=:dot,linewidth=3, label="Truth")
#     plot!(plotrange, kde_pr, linewidth=1, color=:black, label="Prior")
#     density!(x_mcmc[ind,:] , linewidth=2, color=:black, linestyle=:dash, label="Posterior, MCMC")
#     # density!(x_possamp_transport[ind,:], linewidth=2, color=:crimson, label="Posterior, Transport")
#     density!(x_possamp_transport[ind,:], linewidth=2, color=:crimson, label="Posterior, Transport")
#     density!(x_possamp_transport_2[ind,:], linewidth=2, color=:blue, label="Posterior, Transport")
#     display(plt)
#     # savefig("plots/07192025_conddiff/cd_inference_$(ind).pdf")
# end

# save("plots/07192025_conddiff/plotdata.jld", "mcmc", z_mcmc,"transportH", z_possamp_transport_H, "transportHϵ", z_possamp_transport_Hϵ, "transportHy", z_possamp_transport_Hy)
z_mcmc,z_possamp_transport_H, z_possamp_transport_Hϵ, z_possamp_transport_Hy = load("plots/07192025_conddiff/plotdata.jld", "mcmc","transportH", "transportHϵ", "transportHy")


for ind = 1:numProj
    plt = plot(xlabel=L"$z$", xlim = (-3,3), dpi=300) #, size=(400,300), ylim=(0, 1) title="Component " * string(ind),
    plot!([z_true[ind]], seriestype="vline", color=:black, linestyle=:dot,linewidth=3, label="Truth")
    plot!(plotrange, kde_pr, linewidth=1, color=:black, label="Prior")
    density!(z_mcmc[ind,100000:800000] , linewidth=2, color=:black, linestyle=:dash, label="Posterior, MCMC")
    # density!(z_possamp_transport[ind,:], linewidth=2, color=:crimson, label="Posterior, Transport")
    # density!(z_possamp_transport_full[ind,:], linewidth=2, color=:crimson, label="Posterior, Transport")
    # density!(z_possamp_transport_full_2[ind,:], linewidth=2, color=:blue, label="Posterior, Transport")

    density!(z_possamp_transport_H[ind,:], linewidth=2, color=:crimson, label=L"Posterior, $\Gamma=I$")
    density!(z_possamp_transport_Hϵ[ind,:], linewidth=2, color=:blue2, label=L"Posterior, $\Gamma=\Gamma_\epsilon$")
    density!(z_possamp_transport_Hy[ind,:], linewidth=2, color=:green4, label=L"Posterior, $\Gamma=\Gamma_Y$")
    # density!(z_possamp_gauss[ind,:], linewidth=2, label="Posterior - Gauss")
    # density!(z_possamp_gauss_Hϵ[ind,:], linewidth=2,  label="Posterior - Gauss")
    # density!(z_possamp_gauss_Hy[ind,:], linewidth=2,  label="Posterior - Gauss")
    display(plt)
    savefig("plots/07192025_conddiff/cd_inference_$(ind).pdf")
end




# plt = plot(title="Trajectories", ylabel=L"$u$", xlabel=L"$t$", dpi=300)
# t_plot = range(0, stop=1, length=model.n+1)
# plot!(t_plot[2:end], U_r * z_true, color=:black, linewidth=2, label="True")
# plot!(t_plot[2:end], U_r * mean(z_possamp_gauss_H, dims=2), color=:crimson, linewidth=2, label="True")
# plot!(t_plot[2:end], U_r * mean(z_possamp_gauss_Hϵ, dims=2), color=:navy, linewidth=2, label="True")
# plot!(t_plot[2:end], U_r * mean(z_possamp_gauss_Hy, dims=2), color=:green4, linewidth=2, label="True")
# display(plt)


# function rmse(z_approx, z_true, U_r)
#     n, p = size(U_r)
#     p, Nsamp = size(z_approx)
#     err = zeros(n)
#     for i = 1:Nsamp
#         err = err + 1/Nsamp * abs.(U_r * (z_approx[:,i] - z_true))
#     end
#     err
# end


# plt = plot(ylabel="Mean Sample Error", xlabel=L"$t$", dpi=300)
# t_plot = range(0, stop=1, length=model.n+1)
# plot!(t_plot[2:end], rmse(z_possamp_transport_H, z_true, U_r), color=:crimson, linewidth=2, label=L"$\Gamma = I$")
# # plot!(t_plot[2:end], rmse(z_possamp_transport_Hngo, z_true, U_r), color=:black, linewidth=2, label=L"$\Gamma = I$")
# plot!(t_plot[2:end], rmse(z_possamp_transport_Hϵ, z_true, U_r), color=:blue2, linewidth=2, label=L"$\Gamma = \Gamma_\epsilon$")
# plot!(t_plot[2:end], rmse(z_possamp_transport_Hy, z_true, U_r),  color=:green4, linewidth=2, label=L"$\Gamma = \Gamma_Y$")
# # plot!(t_plot[2:end], abs.(U_r * (mean(z_possamp_gauss_H, dims=2) - z_true)), color=:crimson, linewidth=2, label=L"$\Gamma = I$")
# # plot!(t_plot[2:end], abs.(U_r * (mean(z_possamp_gauss_Hϵ, dims=2)- z_true)), color=:navy, linewidth=2, label=L"$\Gamma = \Gamma_\epsilon$")
# # plot!(t_plot[2:end], abs.(U_r * (mean(z_possamp_gauss_Hy, dims=2)- z_true)), color=:orange3, linewidth=2, label=L"$\Gamma = \Gamma_Y$")
# display(plt)
# # savefig("plots/07192025_conddiff/cd_meansamperr.pdf")


# μ_gauss, Σ_gauss = apply_cond_gaussian(vcat(eigvecs(H)[:,end:-1:1][:,1:90]' * (y_samp .-meany), z_samp), eigvecs(H)[:,end:-1:1][:,1:90]' * vec(y_obs -meany), m(model))

# V_r = eigvecs(H)[:,end:-1:1][:,1:20]
# X = vcat(V_r' * (y_samp .- meany), sqrt(invΓ_z) * (z_samp .- μ_z))[:,1:Nsamp]
# μ_gauss, Σ_gauss = apply_cond_gaussian(X, vec(V_r' * (y_obs - meany)), 20)

# plt = plot(title="Trajectories", ylabel=L"$u$", xlabel=L"$t$", dpi=300)
# t_plot = range(0, stop=1, length=model.n+1)
# plot!(t_plot[2:end], U_r * z_true)
# plot!(t_plot[2:end], U_r * μ_gauss)


# ## ERROR in THE MEAN


# p = numProj
# N_mc = 100
# dimlist = vcat(1:5:30,40:10:99)
# numdim = length(dimlist)
# μ_zgy_vec, Σ_zgy_vec= zeros(p, N_mc), zeros(p, N_mc)
# μ_zgy, μ_zgy_Hy, μ_zgy_Hϵ, μ_zgy_H = zeros(p, N_mc), zeros(p, numdim, N_mc), zeros(p, numdim, N_mc), zeros(p, numdim, N_mc)
# Σ_zgy, Σ_zgy_Hy, Σ_zgy_Hϵ, Σ_zgy_H = zeros(p, p, N_mc), zeros(p, p, numdim, N_mc), zeros(p, p, numdim, N_mc), zeros(p, p, numdim, N_mc)
# z_true = zeros(p, N_mc)
# Random.seed!(123)
# pry, prz = y_samp, z_samp #prsamp.y[:,1:100:end], prsamp.z[:,1:100:end]

# @time for i in 1:N_mc
#     fullx = rand(MultivariateNormal(μ_x, Γ_x))
#     fully = compute_G(model, fullx) + rand(MultivariateNormal(zeros(m(model)), Γ_ϵ))
#     z_true[:,i] = U_r' * fullx
#     # μ_zgy_vec[i], Σ_zgy_vec[i] = apply_cond_gaussian(vcat(pry, prz), setup.y, n) 
#     μ_zgy[:,i], Σ_zgy[:, :,i] = apply_cond_gaussian(vcat(invsqrtcovy * (pry .- meany), prz), invsqrtcovy * (fully - meany)[:,1], model.n)
    
#     for j in 1:numdim #r in dimlist
#         r = dimlist[j]
#         Vr = eigvecs(H)[:,end:-1:1][:,1:r]
#         μ_zgy_H[:, j,i], Σ_zgy_H[:,:,j,i] = apply_cond_gaussian(vcat(Vr' * (pry .- meany), prz), (Vr' * (fully .- meany))[:,1] , r) 

#         Vr = eigvecs(H_ϵ)[:,end:-1:1][:,1:r]
#         μ_zgy_Hϵ[:, j,i], Σ_zgy_Hϵ[:, :, j,i] = apply_cond_gaussian(vcat(Vr' * (pry .- meany), prz), Vr' * (fully .- meany)[:,1], r) 

#         Vr = eigvecs(H_y)[:,end:-1:1][:,1:r]
#         μ_zgy_Hy[:, j,i], Σ_zgy_Hy[:, :, j,i] = apply_cond_gaussian(vcat(Vr' * (pry .- meany), prz), Vr' * (fully .- meany)[:,1], r) 

         
#     end

# end

# err_H, err_Hϵ, err_Hy = zeros(numdim), zeros(numdim), zeros(numdim)
# br_H, br_Hϵ, br_Hy= zeros(numdim), zeros(numdim), zeros(numdim)
# for j in 1:numdim
#     for i in 1:N_mc
#         br_H[j] = br_H[j] + 1/N_mc * (μ_zgy_H[:,j,i] - z_true[:,i])' * inv(Σ_zgy_H[:,:,j,i]) * (μ_zgy_H[:,j,i] - z_true[:,i])
#         br_Hϵ[j] = br_Hϵ[j] + 1/N_mc *  (μ_zgy_Hϵ[:,j,i] - z_true[:,i])' * inv(Σ_zgy_Hϵ[:,:,j,i]) * (μ_zgy_Hϵ[:,j,i] - z_true[:,i])
#         br_Hy[j] = br_Hy[j] + 1/N_mc *  (μ_zgy_Hy[:,j,i] - z_true[:,i])' * inv(Σ_zgy_Hy[:,:,j,i]) * (μ_zgy_Hy[:,j,i] - z_true[:,i])
            
#         for indQOI in 1:p
#             err_H[j] = err_H[j] + 1/N_mc/p * abs.(μ_zgy_H[indQOI,j,i] - μ_zgy[indQOI,i]) / abs(μ_zgy[indQOI,i])  # #/ Σ_zgy_godr[indQOI,indQOI,j,i]
#             err_Hϵ[j] = err_Hϵ[j] + 1/N_mc/p *  abs.(μ_zgy_Hϵ[indQOI,j,i] - μ_zgy[indQOI,i]) / abs(μ_zgy[indQOI,i])#-  #/ sqrt(Σ_zgy_nogoal[indQOI,indQOI,j,i])
#             err_Hy[j] = err_Hy[j] + 1/N_mc/p *  abs.(μ_zgy_Hy[indQOI,j,i] - μ_zgy[indQOI,i]) / abs(μ_zgy[indQOI,i])#-  #/ sqrt(Σ_zgy_pca[indQOI,indQOI,j,i])
#         end
#     end
# end
# err0, br0 = 0, 0
# for i in 1:N_mc
#     br0 = br0 + 1/N_mc * (μ_z[:,1] - μ_zgy[:,i])' * invΓ_z * (μ_z[:,1] - μ_zgy[:,i])
#     for indQOI in 1:p
#         # err0 = err0 + 1/N_mc * abs.(setup.μ_z[indQOI,1] - z_true[indQOI,i]) #/ sqrt(setup.Γ_z[indQOI,indQOI])
#         err0 = err0 + 1/N_mc * abs.(μ_z[indQOI,1] - z_true[indQOI,i]) / abs(z_true[indQOI,i])#/ sqrt(setup.Γ_z[indQOI,indQOI])
#     end
# end


# # save("data/data_CliMA/err_gauss_rank.jld", "godr", err_godr, "nogoal", err_nogoal, "pca", err_pca)

# plot(dpi=300, xlims=[0,100],legend=:topright)#, ylims=[5e-2,1e1]),
# plot!(ylabel="Relative Error", xlabel="Rank")#, title="Norm Error")
# plot!(vcat([0],dimlist), vcat([err0], err_H), linewidth=3, color=:green, linestyle=:dash, label="No Whitening")
# plot!(vcat([0],dimlist),  vcat([err0], err_Hϵ), linewidth=3, color=:blue3, linestyle=:dot, label=L"Whiten with $\Gamma_\epsilon$")
# plot!(vcat([0],dimlist), vcat([err0], err_Hy), linewidth=3, color=:red, label=L"Whiten with $\Gamma_y$")

