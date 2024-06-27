using LinearAlgebra, Random, LogExpFunctions


function proposal_1d(μ, σ)
    μ + σ * randn(1)[1]
end


#### INTEGRATE THE LIKELIHOOD into THESE FUNCTION ####

function logpos_1d_covexpand(xr, μ_pr, Γ_pr, Γ_obs, invΓ_xr, Q, O, y) # use the linearized likelihood

    Qz = Q * xr 
    gx = aoe_fwdfun(Qz + μ_pr) 
    gdx = aoe_gradfwdfun(Qz + μ_pr, gx)
    # gx = fwdtoy(Qz + μ_pr) 
    # gdx = dfwdtoy(Qz + μ_pr)

    Γ_Δ = Γ_obs + gdx * (Γ_pr - Φ * O * Γ_pr) * gdx'
    invΓ_obs = inv(cholesky(tril(Γ_Δ)+ tril(Γ_Δ,-1)'))
    logprior = 1/2 * xr' * invΓ_xr * xr
    loglikelihood = -1/2 * (y - gx)' * invΓ_obs * (y - gx) - 1/2 * logdet(Γ_Δ)

    logprior[1] + loglikelihood
end

function logpos_1d(xr, μ_pr, Γ_pr, invΓ_obs, invΓ_xr, Q, O, y; m=1000)
    Qz = Q * xr
    logprior = -1/2 * xr' * invΓ_xr * xr
    
    expTerm = zeros(m)
    for i in 1:m
        gx_augment = vec(fwdtoy(Qz + μ_pr + sqrt((I - Q * O) * Γ_pr) * randn(3)))
        # gx_augment = vec(aoe_fwdfun(Qz + μ_pr + sqrt((I - Q * O) * Γ_pr) * randn(328)))
        expTerm[i] = -1/2 * (y - gx_augment)' * invΓ_obs * (y - gx_augment)
    end
    logprior[1] + logsumexp(expTerm)
end

function logpos_1d_gmm(xr, gmm, invΓ_xr, y)
    logprior = -1/2 * xr' * invΓ_xr * xr
    logprior[1] + gmm_likelihood(gmm, xr, y) #logpdf(gmm, y) 
end

function alpha_1d(x, z, μ_pr, Γ_pr, invΓ_obs, invΓ_xr, Φ, O, y)
    lpz = logpos_1d(z, μ_pr, Γ_pr, invΓ_obs, invΓ_xr, Φ, O, y)
    lpx = logpos_1d(x, μ_pr, Γ_pr, invΓ_obs, invΓ_xr, Φ, O, y)
    ratio = lpz-lpx
    return minimum((1, exp(ratio))), lpz, lpx #, dgz, dgx
end

function alpha_1d_covexpand(x, z, μ_pr, Γ_pr, Γ_obs, invΓ_xr, Φ, O, y)
    lpz = logpos_1d_covexpand(z, μ_pr, Γ_pr, Γ_obs, invΓ_xr, Φ, O, y)
    lpx = logpos_1d_covexpand(x, μ_pr, Γ_pr, Γ_obs, invΓ_xr, Φ, O, y)
    ratio = lpz-lpx
    return minimum((1, exp(ratio))), lpz, lpx #, dgz, dgx
end

function alpha_1d_gmm(x, z, gmm, invΓ_xr, y)
    lpz = logpos_1d_gmm(z, gmm, invΓ_xr, y)
    lpx = logpos_1d_gmm(x, gmm, invΓ_xr, y)
    ratio = lpz-lpx
    return minimum((1, exp(ratio))), lpz, lpx #, dgz, dgx
end


function mcmc_lis_1d(x0, μ_pr, Γ_pr, Γ_obs, Φ, O, y; N=1000, offset=0, logposmethod="covexpand")
    # ensure that x_prev is the whitened version (subtract prior)! 
    x_vals = zeros(N)
    invΓ_pr, invΓ_obs, invΓ_xr = inv(Γ_pr), inv(Γ_obs), inv(O* Γ_pr * O')

    logpos, accept = zeros(N), zeros(N)
    sd, eps = 2.38^2, 1e-10

    x = (O * (x0 - μ_pr))[1,1]# 0. #x0
    propcov = (sd * O * Γ_pr * O')[1,1] 
    propChol = sqrt(propcov)
    meanXprev = 0.

    for i in 1:N
        z = proposal_1d(x, propChol)
        if logposmethod == "pseudomarg"
            α, lpz, lpx = alpha_1d(x, z, μ_pr, invΓ_pr, invΓ_obs, invΓ_xr, Φ, O, y)
        elseif logposmethod == "gmm"
            α, lpz, lpx = alpha_1d_gmm(x, z, gmm, invΓ_xr, y)
        elseif logposmethod == "covexpand"
            α, lpz, lpx = alpha_1d_covexpand(x, z, μ_pr, Γ_pr, Γ_obs, invΓ_xr, Φ, O, y)
        end

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
    return x_vals .+ O * μ_pr

end


