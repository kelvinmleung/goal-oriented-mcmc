using Distributed, Random, Base
using Mamba, LinearAlgebra, JLD

function mcmc_linear(μ_pr, Γ_pr, Γ_obs, G, y, N, alg)
    model = Model(
        y = Stochastic(1, (mu) ->  MvNormal(mu, Γ_obs), false),
        mu = Logical(1, (xmat, beta) -> xmat * beta, false),
        beta = Stochastic(1, () -> MvNormal(μ_pr, Γ_pr))
    )
    # Sampling scheme
    if alg == "AMM"
        scheme = [AMM([:beta])]
    elseif alg == "NUTS"
        scheme = [NUTS([:beta])]
    end

    ## Data
    line = Dict{Symbol, Any}(:y => y)
    line[:xmat] = G 

    ## Initial Values
    inits = [Dict{Symbol, Any}(:y => line[:y], :beta => μ_pr)]

    ## MCMC Simulations
    setsamplers!(model, scheme)
    sim = Base.invokelatest(mcmc, model, line, inits, N, burnin=Int(N/5), thin=1, chains=1)
    return sim.value
end

#need Phi_r, samp from z_perp to add on later, 

function mcmc_linear_lowrank(x0, Γ_obs, Φ, y, N, alg)

    r = size(Φ,2)
    model = Model(
        y = Stochastic(1, (mu) ->  MvNormal(mu, Γ_obs), false),
        mu = Logical(1, (xmat, beta) -> xmat * (Φ * beta + x0), false),
        beta = Stochastic(1, () -> MvNormal(zeros(r), diagm(ones(r))))
    )
    # Sampling scheme
    if alg == "AMM"
        scheme = [AMM([:beta])] #######
    elseif alg == "NUTS"
        scheme = [NUTS([:beta])]
    end

    ## Data
    line = Dict{Symbol, Any}(:y => y)
    line[:xmat] = G 

    ## Initial Values
    inits = [Dict{Symbol, Any}(:y => line[:y], :beta => zeros(r))]

    ## MCMC Simulations
    setsamplers!(model, scheme)
    sim = Base.invokelatest(mcmc, model, line, inits, N, burnin=Int(N/5), thin=1, chains=1)
    return sim.value
end


# function logpos(xr, x0, invΓ_obs, Φ, Ξ, G, y, μ_pr)
#     tPr = xr + Ξ' * (x0 - μ_pr)
#     logprior = -1/2 * tPr' * tPr

#     xFull = Φ * xr + x0
#     tLH = y - G * xFull
#     loglikelihood = -1/2 * tLH' * invΓ_obs * tLH
    
#     logprior + loglikelihood
# end

function logpos(xr, x0, invΓ_obs, Φ, Ξ, y, μ_pr)
    tPr = xr + Ξ' * (x0 - μ_pr)
    logprior = -1/2 * tPr' * tPr

    xFull = Φ * xr + x0
    gx, _ = evalg(xFull)
    loglikelihood = -1/2 * (y - gx)' * invΓ_obs * (y - gx)
    
    logprior + loglikelihood
end

function logpos(xr, μ_pr, invΓ_obs, Φ, y)
    logprior = -1/2 * xr' * xr
    xFull = Φ * xr + μ_pr
    gx, _ = evalg(xFull)
    loglikelihood = -1/2 * (y - gx)' * invΓ_obs * (y - gx)
    logprior + loglikelihood
end

function proposal(μ, cholΓ)
    n = length(μ)
    mvn = MvNormal(zeros(n), diagm(ones(n)))
    μ + cholΓ * rand(mvn)
end

function alpha(x, z, x0, invΓ_obs, Φ, Ξ, y, μ_pr)
    lpz = logpos(z, x0, invΓ_obs, Φ, Ξ, y, μ_pr)
    lpx = logpos(x, x0, invΓ_obs, Φ, Ξ, y, μ_pr)
    ratio = lpz-lpx
    return minimum((1, exp(ratio))), lpz, lpx
end

function alpha(x, z, μ_pr, invΓ_obs, Φ, y)
    lpz = logpos(z, μ_pr, invΓ_obs, Φ, y)
    lpx = logpos(x, μ_pr, invΓ_obs, Φ, y)
    ratio = lpz-lpx
    return minimum((1, exp(ratio))), lpz, lpx
end

# function mcmc_amm(x0, Γ_obs, Φ, Ξ, G, y, N, μ_pr)
#     n, r = size(Φ)
#     x_vals = zeros(r, N)
#     invΓ_obs = inv(Γ_obs)
#     logpos, accept = zeros(N), zeros(N)
#     propcov = (2.38^2) / r * diagm(ones(r))
#     propChol = cholesky(propcov).L
#     sd, eps = 2.38^2 / r, 1e-10
#     meanXprev = zeros(r)

#     x = zeros(r)

#     for i in 1:N
#         z = proposal(x, propChol)
#         α, lpz, lpx = alpha(x, z, x0, invΓ_obs, Φ, Ξ, G, y, μ_pr)
#         if rand(Uniform(0,1)) < α
#             x, lpx = z, lpz
#             accept[i] = 1
#         end
#         x_vals[:,i] = x
#         logpos[i] = lpx
        
#         if i % 500 == 0
#             display("Sample: " * string(i))
#             display("   Accept Rate: " * string(mean(accept[i-499:i])))
#             propChol = cholesky(tril(propcov) + tril(propcov,-1)').L
#         end

#         if i == 1500
#             propcov = sd * cov(x_vals[:,1:1500], dims=2) + eps * I
#             meanXprev = mean(x_vals[:,1:1500], dims=2)
#         elseif i > 1500
#             meanX = i / (i + 1) * meanXprev + 1 / (i + 1) * x_vals[:,i]
#             propcov = (i-1) / i * propcov + sd / i * (i * meanXprev * meanXprev' - (i+1) * meanX * meanX' + x_vals[:,i] * x_vals[:,i]' + eps * I)
#             meanXprev = meanX
#         end
#     end
#     return x_vals
# end

function mcmc_amm(x0, μ_pr, Γ_obs, Φ, Ξ, y, N)

    n, r = size(Φ)
    x_vals = zeros(r, N)
    invΓ_obs = inv(Γ_obs)
    logpos, accept = zeros(N), zeros(N)
    propcov = (2.38^2) / r * diagm(ones(r))
    propChol = cholesky(propcov).L
    sd, eps = 2.38^2 / r, 1e-10
    meanXprev = zeros(r)

    x = zeros(r)
    for i in 1:N
        z = proposal(x, propChol)
        α, lpz, lpx = alpha(x, z, x0, invΓ_obs, Φ, Ξ, y, μ_pr)
        if rand(Uniform(0,1)) < α
            x, lpx = z, lpz
            accept[i] = 1
        end
        x_vals[:,i] = x
        logpos[i] = lpx
        
        if i % 500 == 0
            display("Sample: " * string(i))
            display("   Accept Rate: " * string(mean(accept[i-499:i])))
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
    return x_vals
end

function mcmc_amm_prestart(x_prev, μ_pr, Γ_obs, Φ, y, N)
    # ensure that x_prev is the whitened version (subtract prior)! 
    n, r = size(Φ)
    x_vals = zeros(r, N)
    invΓ_obs = inv(Γ_obs)
    logpos, accept = zeros(N), zeros(N)
    sd, eps = 2.38^2 / r, 1e-10
    # x = zeros(r)
    x = x_prev[:,end]

    # get proposal covariance using existing samples
    propcov = sd * cov(x_prev, dims=2) + eps * I
    propChol = cholesky(propcov).L
    meanXprev = mean(x_prev, dims=2)

    for i in 1:N
        z = proposal(x, propChol)
        α, lpz, lpx = alpha(x, z, μ_pr, invΓ_obs, Φ, y)
        if rand(Uniform(0,1)) < α
            x, lpx = z, lpz
            accept[i] = 1
        end
        x_vals[:,i] = x
        logpos[i] = lpx
        
        if i % 100 == 0
            display("Sample: " * string(i))
            display("   Accept Rate: " * string(mean(accept[i-99:i])))
            propChol = cholesky(tril(propcov) + tril(propcov,-1)').L
        end
        meanX = i / (i + 1) * meanXprev + 1 / (i + 1) * x_vals[:,i]
        propcov = (i-1) / i * propcov + sd / i * (i * meanXprev * meanXprev' - (i+1) * meanX * meanX' + x_vals[:,i] * x_vals[:,i]' + eps * I)
        meanXprev = meanX
    end
    return x_vals
end

function evalg(x)
    # Evaluate g(x) and its gradient at x
    σ_1², σ_2² = 9, 1
    hgt, wdth = 50, 50
    gx, dgx = blurMatrix_nonlinear2(σ_1², σ_2², x, hgt, wdth)
    return gx, dgx
end

# function evalg_multi(xmat)
#     # Evaluate g(x) and its gradient at x
#     σ_1², σ_2² = 9, 1
#     hgt, wdth = 50, 50
#     gx, dgx = blurMatrix_nonlinear2_multi(σ_1², σ_2², xmat, hgt, wdth)
#     return gx, dgx
# end

function evalHessian(z, Q, O, Γ_x, Γ_ϵ)
    # Monte Carlo estimate of the global Hessian given samples z, which are preprocessed into function evaluations as dgQz
    # m = size(dgQz, 3)
    p = size(O,1)
    H = zeros(p,p)
    for i in 1:m
        _, dgQzj = evalg(Q*z[:,i])
        H = H + 1/m * Q' * dgQzj' * inv(dgQzj * (Γ_x - Q * O * Γ_x) * dgQzj' + Γ_ϵ) * dgQzj * Q
    end
    return H
end

function eigensolve(L, H, Q)
    eigmat = L' * H * L
    Λ0, W0 = eigen(tril(eigmat) + tril(eigmat, -1)')
    n = size(eigmat,1)

    Λ = Λ0[end:-1:1]
    W = W0[:,end:-1:1]
    r = 10
    for i in 1:n-1
        if Λ[i] >= 1 && Λ[i+1] < 1
            r = i
        end
    end
    invL = inv(L)
    W_r, W_perp = W[:,1:r], W[:,r+1:end]
    Φ_r, Φ_perp = Q * L * W_r, Q * L * W_perp
    Ξ_r, Ξ_perp = Q * invL' * W_r, Q * invL' * W_perp
    return Λ, r, Φ_r, Φ_perp, Ξ_r, Ξ_perp
end

function priorHessian(μ_x, Γ_x, Γ_ϵ, O, Nsamp)

    p, n = size(O)
    # μ_z = O * μ_x
    Γ_z = O * Γ_x * O'
    invΓ_z = inv(Γ_z)
    # L_z = cholesky(Γ_z).L

    Q = Γ_x * O' * invΓ_z

    # Generate prior samples and estimate Hessian
    priorDist = MvNormal(μ_x, Γ_x)
    x_pr = rand(priorDist, Nsamp)
    z = O * x_pr

    # gQz, dgQz = zeros(n,Nsamp), zeros(n,n,Nsamp)
    # display("Function evaluation...")

    H = zeros(p,p)
    @time for i in 1:Nsamp
        # evaluate the function
        # gQz[:,i], dgQz[:,:,i] = evalg(Q*z[:,i])
        _, dgQzj = evalg(Q*z[:,i])
        H = H + 1/Nsamp * Q' * dgQzj' * inv(dgQzj * (Γ_x - Q * O * Γ_x) * dgQzj' + Γ_ϵ) * dgQzj * Q
    end

    display("Hessian...")
    # @time H = evalHessian(dgQz, Q, O, Γ_x, Γ_ϵ)

    @save "mcmc/priorHessian.jld" H x_pr

end

function mcmc_lis_nonlinear(μ_x, Γ_x, Γ_ϵ, O, y; m=1000, epoch=1) #m0=10, 

    p, n = size(O)
    μ_z = O * μ_x
    Γ_z = O * Γ_x * O'
    invΓ_z = inv(Γ_z)
    L_z = cholesky(Γ_z).L

    Q = Γ_x * O' * invΓ_z
    Λ_all = zeros(n, epoch)

    x_all = zeros(n, m * epoch)

    # # Generate prior samples and estimate Hessian
    # priorDist = MvNormal(μ_x, Γ_x)
    # x_pr = rand(priorDist, m)
    # z = O * x_pr

    
    # gQz, dgQz = zeros(n,m0), zeros(n,n,m0)
    # display("Function evaluation...")

    # # @time gQz, dgQz = evalg_multi(Q*z)
    # @time for i in 1:m0
    #     # evaluate the function
    #     gQz[:,i], dgQz[:,:,i] = evalg(Q*z[:,i])
    # end

    # display("Hessian...")
    # @time H = evalHessian(dgQz, Q, O, Γ_x, Γ_ϵ)

    @load "mcmc/priorHessian.jld" H x_pr


    for t in 1:epoch
        # Solve eigenvalue problem
        Λ_all[1:p, t], r, Φ_r, Φ_perp, Ξ_r, _ = eigensolve(L_z, H, Q) 
        Plots.plot(Λ_all[1:p, t], yaxis=:log)

        display("Rank: " * string(r))
        # Run MCMC
        x0 = x_all[:,m*(t-1)+1] # begin where the chain ended last # Ξ_r' * 
        display("MCMC...")
        # @time x_r = mcmc_amm_(x0, μ_x, Γ_ϵ, Φ_r, Ξ_r, y, m)
    
        ## HOW TO PROJECT x_pr to x_prev
        x_prev = Ξ_r' * (x_pr - repeat(μ_x, 1, size(x_pr,2)))
        @time x_r = mcmc_amm_prestart(x_prev, μ_x, Γ_ϵ, Φ_r, y, m)


        pr_perp = MvNormal(zeros(p - r), diagm(ones(p - r)))
        pr_perp_samp = rand(pr_perp, m)
        # display(pr_perp_samp)
        # display(Φ_perp)
        # samps_full = Φ_r * samps_lowrank_amm[:,:] + repeat(μ_x, 1, Nsamp) + Φ_perp * pr_perp_samp 
        x_all[:,m*(t-1)+1 : m*t] = Φ_r * x_r[:,:] + repeat(μ_x, 1, m) + Φ_perp * pr_perp_samp 

        # # Estimate Hessian
        # z = O * x_all
        # for i in 1:m
        #     gQz[:,i], dgQz[:,:,i] = evalg(Q*z[:,i])
        # end
        # display("Hessian...")
        # @time H = evalHessian(dgQz, Q, O, Γ_x, Γ_ϵ)
    end
    return hcat(x_pr, x_all)
end



    
    