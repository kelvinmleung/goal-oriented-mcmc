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
    logprior + loglikelihood #, dgx
end

function logpos_1d(xr, x0, μ_pr, invΓ_pr, Γ_obs, Φ, O, y)

    # tPr = xr + (O * (x0 - μ_pr))[1]
    Qz = Φ * (xr + x0) 
    
    # x_pr = x_prsamp[:,rand(1:100000)] # [0; -0.13691952528258547; -0.06135762296452887] #
    # x_pr[1] = xr + x0 #+ x_pr[1] 
    # gx = fwdtoy(Qz) # + dfwdtoy(Qz) * (I - Φ * O) * x_pr #dfwdtoy(Q * z_true) * (I - Q * O) * x_true
    # gdx = dfwdtoy(Qz)
    gx = aoe_fwdfun(Qz)
    # @time gdx = aoe_gradfwdfun(Qz)
    gdx = aoe_gradfwdfun(Qz, gx)
    Γ_Δ = Γ_obs + gdx * (Γ_x - Φ * O * Γ_x) * gdx'
    invΓ_obs = inv(cholesky(tril(Γ_Δ)+ tril(Γ_Δ,-1)')) #inv(Γ_obs + gdx * (Γ_x -  Q * O * Γ_x) * gdx')
    logprior = -1/2 * (Qz - μ_pr)' * invΓ_pr * (Qz - μ_pr)
    loglikelihood = -1/2 * (y - gx)' * invΓ_obs * (y - gx) - 1/2 * logdet(Γ_Δ)
    logprior[1] + loglikelihood
end


function logpos_simple(x, μ_pr, invΓ_pr, invΓ_obs, y)
    gx = fwdtoy(x)
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

# function alpha(x, z, x0, invΓ_obs, Φ, Ξ, y, μ_pr)
#     lpz = logpos(z, x0, invΓ_obs, Φ, Ξ, y, μ_pr)
#     lpx = logpos(x, x0, invΓ_obs, Φ, Ξ, y, μ_pr)
#     ratio = lpz-lpx
#     return minimum((1, exp(ratio))), lpz, lpx
# end

function alpha_simple(x, z, μ_pr, invΓ_x, invΓ_obs, y)
    lpz = logpos_simple(z, μ_pr, invΓ_x, invΓ_obs, y)
    lpx = logpos_simple(x, μ_pr, invΓ_x, invΓ_obs, y)
    return minimum((1, exp(lpz-lpx))), lpz, lpx
end

function alpha(x, z, μ_pr, invΓ_obs, Φ, y)
    lpz = logpos(z, μ_pr, invΓ_obs, Φ, y)
    lpx = logpos(x, μ_pr, invΓ_obs, Φ, y)
    ratio = lpz-lpx
    return minimum((1, exp(ratio))), lpz, lpx #, dgz, dgx
end


function alpha_1d(x, x0, z, μ_pr, invΓ_pr, Γ_obs, Φ, O, y)
    lpz = logpos_1d(z, x0, μ_pr, invΓ_pr, Γ_obs, Φ, O, y)
    lpx = logpos_1d(x, x0, μ_pr, invΓ_pr, Γ_obs, Φ, O, y)
    ratio = lpz-lpx
    return minimum((1, exp(ratio))), lpz, lpx #, dgz, dgx
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

# function mcmc_amm(x0, μ_pr, Γ_obs, Φ, Ξ, y, N)

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
#         α, lpz, lpx = alpha(x, z, x0, invΓ_obs, Φ, Ξ, y, μ_pr)
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

function mcmc_amm_simple(x0, μ_pr, Γ_pr, Γ_obs, y, N)

    r = length(μ_pr)
    x_vals = zeros(r, N)
    invΓ_pr, invΓ_obs = inv(Γ_pr), inv(Γ_obs)
    logpos, accept = zeros(N), zeros(N)
    propcov = (2.38^2) / r * diagm(ones(r))
    propChol = cholesky(propcov).L 
    sd, eps = 2.38^2 / r, 1e-10
    meanXprev = zeros(r)

    x = x0 #zeros(r)
    # x[:,1] = x0
    for i in 1:N
        z = proposal(x, propChol)
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
        α, lpz, lpx = alpha(x, z, μ_pr, invΓ_obs, Φ, y) #, dgz, dgx
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
    # σ_k² = 4
    σ_1², σ_2² = 9, 1
    hgt, wdth = 50, 50
    gx, dgx = blurMatrix_nonlinear2(σ_1², σ_2², x, hgt, wdth) ### TRY THE LINEAR , Which hessian do we get
    # gx, dgx = blurMatrix_linear2(σ_k², x, hgt, wdth) 
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
    m = size(z,2)
    p = size(O,1)
    H = zeros(p,p)
    display("Computing Hessian...")
    for i in 1:m
        _, dgQzj = evalg(Q*z[:,i])
        H = H + 1/m * Q' * dgQzj' * inv(dgQzj * (Γ_x - Q * O * Γ_x) * dgQzj' + Γ_ϵ) * dgQzj * Q
    end
    return H
end

# Q' * dgQzj' * inv(dgQzj * (Γ_x - Q * O * Γ_x * O' * Q') * dgQzj' + Γ_ϵ) * dgQzj * Q

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

    # display("Hessian...")
    # @time H = evalHessian(dgQz, Q, O, Γ_x, Γ_ϵ)

    @save "mcmc/priorHessian.jld" H x_pr

end


function mcmc_lis_1d(x0, μ_pr, Γ_pr, Γ_obs, Φ, O, y; N=1000)
    # ensure that x_prev is the whitened version (subtract prior)! 
    x_vals = zeros(N)
    invΓ_pr, invΓ_obs = inv(Γ_pr), inv(Γ_obs)
    logpos, accept = zeros(N), zeros(N)
    sd, eps = 2.38^2, 1e-10

    x = x0
    propcov = sd * O * Γ_pr * O'
    propChol = sqrt(propcov)
    meanXprev = 0.

    for i in 1:N
        z = proposal_1d(x, propChol)
        # α, lpz, lpx = alpha_1d(x, z, μ_pr, invΓ_obs, Φ, y) #, dgz, dgx
        α, lpz, lpx = alpha_1d(x, x0, z, μ_pr, invΓ_pr, Γ_obs, Φ, O, y)
        if rand(Uniform(0,1)) < α
            x, lpx = z, lpz
            accept[i] = 1
        end
        x_vals[i] = x
        logpos[i] = lpx

        if i % 100 == 0
            if mean(accept[i-99:i]) < 0.1
                propChol = propChol / sd
            end
        end

        if i % 500 == 0
            
            if i % 1000 == 0
                display("Sample: " * string(i))
                display("   Accept Rate: " * string(mean(accept[i-999:i])))
            end
            # display("Sample: " * string(i))
            # display("   Accept Rate: " * string(mean(accept[i-499:i])))
            propChol = sqrt(propcov)
        end

        if i == 1500
            propcov = sd * var(x_vals[1:1500]) + eps
            meanXprev = mean(x_vals[1:1500])
        elseif i > 1500
            meanX = i / (i + 1) * meanXprev + 1 / (i + 1) * x_vals[i]
            propcov = (i-1) / i * propcov + sd / i * (i * meanXprev^2 - (i+1) * meanX^2 + x_vals[i]^2 + eps)
            meanXprev = meanX
        end
    end
    return x_vals

end

function mcmc_lis_nonlinear(μ_x, Γ_x, Γ_ϵ, O, H, x_prev, y; m=1000) #m0=10, 

    p, n = size(O)
    Γ_z = O * Γ_x * O'
    invΓ_z = inv(Γ_z)
    L_z = cholesky(Γ_z).L

    Q = Γ_x * O' * invΓ_z
    Λ_all = zeros(n)

    # x_all = zeros(n, m )

    # @load "mcmc/priorHessian.jld" H x_pr


    # Solve eigenvalue problem
    Λ_all[1:p], r, Φ_r, Φ_perp, Ξ_r, _ = eigensolve(L_z, H, Q) 
    # Plots.plot(Λ_all[1:p], yaxis=:log)
    # Plots.plot(eigvals(H)[end:-1:1], yaxis=:log)

    display("Rank: " * string(r))
    # Run MCMC

    display("MCMC...")
    # @time x_r = mcmc_amm_(x0, μ_x, Γ_ϵ, Φ_r, Ξ_r, y, m)

    x_prev_r = Ξ_r' * (x_prev - repeat(μ_x, 1, size(x_prev,2)))
    @time x_r = mcmc_amm_prestart(x_prev_r, μ_x, Γ_ϵ, Φ_r, y, m)

    pr_perp = MvNormal(zeros(p - r), diagm(ones(p - r)))
    pr_perp_samp = rand(pr_perp, m)
    # display(pr_perp_samp)
    # display(Φ_perp)
    # samps_full = Φ_r * samps_lowrank_amm[:,:] + repeat(μ_x, 1, Nsamp) + Φ_perp * pr_perp_samp 
    x_all = Φ_r * x_r[:,:] + repeat(μ_x, 1, m) + Φ_perp * pr_perp_samp 

    # # Estimate Hessian
    # z = O * x_all
    # for i in 1:m
    #     gQz[:,i], dgQz[:,:,i] = evalg(Q*z[:,i])
    # end
    # display("Hessian...")
    # @time H = evalHessian(dgQz, Q, O, Γ_x, Γ_ϵ)

    return hcat(x_prev, x_all)
end



    
    