# using Distributed, Random, Base
using LinearAlgebra, Random, LogExpFunctions


function proposal_1d(μ, σ)
    μ + σ * randn(1)[1]
end

function logpos_1d_approx(xr, μ_pr, invΓ_pr, Γ_obs, Φ, O, y) # use the linearized likelihood

    Qz = Φ * xr #+ x0 
    gx = aoe_fwdfun(Qz + μ_pr) 
    gdx = aoe_gradfwdfun(Qz + μ_pr, gx)

    Γ_Δ = Γ_obs + gdx * (Γ_x - Φ * O * Γ_x) * gdx'
    invΓ_obs = inv(cholesky(tril(Γ_Δ)+ tril(Γ_Δ,-1)'))
    logprior = -1/2 * (Qz)' * invΓ_pr * (Qz)
    loglikelihood = -1/2 * (y - gx)' * invΓ_obs * (y - gx) - 1/2 * logdet(Γ_Δ)

    logprior[1] + loglikelihood
end

function logpos_1d(xr, μ_pr, invΓ_pr, invΓ_obs, Φ, O, y)
    Qz = Φ * xr
    logprior = -1/2 * (Qz)' * invΓ_pr * (Qz)
    m = 100
    prSamp = rand(normDist, m)
    expTerm = zeros(m)
    for i in 1:m
        # gx_augment = fwdtoy(Qz + (I - Q*O)*(prSamp[:,i] - μ_pr) + μ_pr)
        gx_augment = aoe_fwdfun(Qz + (I - Q*O)*(prSamp[:,i] - μ_pr) + μ_pr)
        expTerm[i] = -1/2 * (y - gx_augment)' * invΓ_obs * (y - gx_augment)
    end
    logprior[1] + logsumexp(expTerm)#log(likelihood) 
end

# function logpos_1d_gmm(xr, μ_pr, invΓ_pr, invΓ_obs, Φ, O, y)


function alpha_1d(x, z, μ_pr, invΓ_pr, invΓ_obs, Φ, O, y)
    lpz = logpos_1d(z, μ_pr, invΓ_pr, invΓ_obs, Φ, O, y)
    lpx = logpos_1d(x, μ_pr, invΓ_pr, invΓ_obs, Φ, O, y)
    ratio = lpz-lpx
    return minimum((1, exp(ratio))), lpz, lpx #, dgz, dgx
end

function alpha_1d_approx(x, z, μ_pr, invΓ_pr, Γ_obs, Φ, O, y)
    lpz = logpos_1d_approx(z, μ_pr, invΓ_pr, Γ_obs, Φ, O, y)
    lpx = logpos_1d_approx(x, μ_pr, invΓ_pr, Γ_obs, Φ, O, y)
    ratio = lpz-lpx
    return minimum((1, exp(ratio))), lpz, lpx #, dgz, dgx
end




## Next 5 functions are for the approach from the unified paper

function logpos_beta(xr, xr0, μ_pr, invΓ_pr, invΓ_obs, Q, y) 
    Qz = Q * (xr + xr0) 
    # gx = aoe_fwdfun(Qz) 
    gx = fwdtoy(Qz)
    gdx = dfwdtoy(Qz)
    invΓ_obs = inv(inv(invΓ_obs) + gdx * (Γ_x -  Q * O * Γ_x) * gdx')
    logprior = -1/2 * (Qz - μ_pr)' * invΓ_pr * (Qz - μ_pr)   
    loglikelihood = -1/2 * (y - gx)' * invΓ_obs * (y - gx) 
    logprior[1] + loglikelihood
end

function logpos_alpha(xr, x_full, xr0, μ_xr, invΓ_xr, invΓ_obs, y) 
    # gx = aoe_fwdfun(x_full)
    gx = fwdtoy(x_full)
    logprior = -1/2 * (xr + xr0 - μ_xr)^2 * invΓ_xr  
    loglikelihood = -1/2 * (y - gx)' * invΓ_obs * (y - gx)

    logprior[1] + loglikelihood 

end

function beta(xr, xrprop, x0, μ_pr, invΓ_pr, invΓ_obs, Q, y)
    lpxprop = logpos_beta(xrprop, x0, μ_pr, invΓ_pr, invΓ_obs, Q, y) 
    lpx = logpos_beta(xr, x0, μ_pr, invΓ_pr, invΓ_obs, Q, y) 
    ratio = lpxprop - lpx
    return minimum((1, exp(ratio))), lpxprop, lpx
end

function alpha(xr, xrprop, xr0, x_full, x_fullprop, μ_xr, invΓ_xr, invΓ_obs, y, lpx_beta, lpxprop_beta)
    lpxprop = logpos_alpha(xrprop, x_fullprop, xr0, μ_xr, invΓ_xr, invΓ_obs, y) 
    lpx = logpos_alpha(xr, x_full, xr0, μ_xr, invΓ_xr, invΓ_obs, y) 
    ratio =  lpxprop - lpx - lpxprop_beta + lpx_beta
    return minimum((1, exp(ratio))), lpxprop-lpxprop_beta, lpx-lpx_beta
end


function mcmc_lis_1d(x0, μ_pr, Γ_pr, Γ_obs, Φ, O, y; N=1000)
    # ensure that x_prev is the whitened version (subtract prior)! 
    x_vals = zeros(N)
    invΓ_pr, invΓ_obs = inv(Γ_pr), inv(Γ_obs)
    logpos, accept = zeros(N), zeros(N)
    sd, eps = 2.38^2, 1e-10

    x = (O * (x0 - μ_pr))[1,1]# 0. #x0
    propcov = (sd * O * Γ_pr * O')[1,1] 
    propChol = sqrt(propcov)
    meanXprev = 0.

    for i in 1:N
        z = proposal_1d(x, propChol)
        # α, lpz, lpx = alpha_1d(x, z, μ_pr, invΓ_pr, invΓ_obs, Φ, O, y)
        α, lpz, lpx = alpha_1d_approx(x, z, μ_pr, invΓ_pr, Γ_obs, Φ, O, y)

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
    return x_vals .+ O*μ_x

end




function mcmc_lis_unified(x0, μ_pr, Γ_pr, Γ_obs, Q, O, y; N=1000)

    x_vals = zeros(N)
    x_vals_full = zeros((size(Q,1),N))
    invΓ_pr, invΓ_obs = inv(Γ_pr), inv(Γ_obs)
    μ_xr, invΓ_xr = (O*μ_pr)[1], inv(O * invΓ_pr * O')[1,1]
    accept_beta, accept_alpha = ones(N), zeros(N)
    sd, eps = 2.38^2, 1e-10

    prDist = MvNormal(μ_pr, Γ_pr)

    x = 0.
    x_pr = zeros(size(Q,1))#rand(prDist)
    propcov = (sd * O * Γ_pr * O')[1,1]
    propChol = sqrt(propcov)
    meanXprev = 0.
    # plot(Q*(x+x0)+(I - Q*O) * x_pr)
    for i in 1:N
        xprop = proposal_1d(x, propChol)

        β, lpxprop_beta, lpx_beta = beta(x, xprop, x0, μ_pr, invΓ_pr, invΓ_obs, Q, y)
        if rand(Uniform(0,1)) < 1-β
            xprop = x
            accept_beta[i] = 0
        end

        xprop_pr = rand(prDist) 
        x_full = Q * (x + x0) + (I - Q*O) * x_pr
        x_fullprop = Q * (xprop + x0) + (I - Q*O) * xprop_pr

        α, lpxprop, lpx = alpha(x, xprop, x0, x_full, x_fullprop, μ_xr, invΓ_xr , invΓ_obs, y, lpx_beta, lpxprop_beta)
        
        if rand(Uniform(0,1)) < α
            x = xprop
            x_pr = xprop_pr
            accept_alpha[i] = 1
        end
        x_vals[i] = x + x0
        x_vals_full[:,i] = Q*(x+x0) + (I - Q*O) * x_pr

        if i % 500 == 0
            
            if i % 1000 == 0
                display("Sample: " * string(i))
                display("   Accept Rate 1: " * string(mean(accept_beta[i-999:i])))
                display("   Accept Rate 2: " * string(mean(accept_alpha[i-999:i])))
            end
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
    return x_vals, x_vals_full
end

function mcmc_lis_unified_scrap(x0, μ_pr, Γ_pr, Γ_obs, Φ, O, y; N=1000)
    # ensure that x_prev is the whitened version (subtract prior)! 
    x_vals = zeros(N)
    invΓ_pr, invΓ_obs = inv(Γ_pr), inv(Γ_obs)
    accept1 , accept2 = zeros(N), zeros(N)
    sd, eps = 2.38^2, 1e-10

    x = x0
    x_prime = x0
    propcov = (sd * O * Γ_pr * O')[1,1]
    propChol = sqrt(propcov)
    meanXprev = 0.

    for i in 1:N
        z = proposal_1d(x, propChol)
        α, lpz, lpx = alpha_1d(x, x0, z, μ_pr, invΓ_pr, Γ_obs, Φ, O, y)

        # from unified paper:
        if rand(Uniform(0,1)) < 1 - α
            x_prime = x
        else
            x_prime = z
            accept1[i] = 1
        end

        α_second, lpz_second, lpx_second = alpha_1d_second(x_prime, x0, z, μ_pr, invΓ_pr, Γ_obs, Φ, O, y, lpx, lpz)
        if rand(Uniform(0,1)) < α_second
            x = x_prime
            accept2[i] = 1
        end

        x_vals[i] = x

        # if i % 100 == 0
        #     if mean(accept2[i-99:i]) < 0.1
        #         propChol = propChol / sd
        #     end
        # end

        if i % 500 == 0
            
            if i % 1000 == 0
                display("Sample: " * string(i))
                display("   Accept Rate 1: " * string(mean(accept1[i-999:i])))
                display("   Accept Rate 2: " * string(mean(accept2[i-999:i])))
            end
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