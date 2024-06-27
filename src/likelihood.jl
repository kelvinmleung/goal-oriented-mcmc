function cdr_likelihood(z, μ_pr, invΓ_obs, Q, O, y; offset=0)
    Qz = Q * z
    m = 1000
    prSamp = rand(normDist, m)
    expTerm = zeros(m)
    for i in 1:m

        # gx_augment = vec(fwdtoy(Qz + μ_pr + sqrt((I - Q * O) * Γ_x) * randn(3)))
        # gx_augment = vec(fwdtoy(Qz + μ_pr + (I - Q*O)*(prSamp[:,i] - μ_pr)))
        gx_augment = vec(aoe_fwdfun(Qz + μ_pr + sqrt((I - Q * O) * Γ_x) * randn(328)))


        expTerm[i] = -1/2 * (y - gx_augment)' * invΓ_obs * (y - gx_augment)
    end
    logsumexp(expTerm) + log(1/m)
end


function gmm_likelihood(gmm, z_input, y)
    yz_weights = weights(gmm)
    yz_means = means(gmm)
    yz_covs = covars(gmm)

    nComp = length(yz_weights)
    ygz_means = zeros((nComp, n))
    ygz_covs = [zeros(n,n) for _ in 1:nComp]

    for i in 1:nComp
        ygz_covs[i] = yz_covs[i][1:end-1,1:end-1] - yz_covs[i][1:end-1,end] * yz_covs[i][1:end-1,end]' / yz_covs[i][end,end]
        ygz_means[i,:] = yz_means[i,1:end-1] + yz_covs[i][1:end-1,end] * (z_input - yz_means[i,end])  / yz_covs[i][end,end] 
    end

    mixtureDef = []
    for i in 1:nComp
        push!(mixtureDef, (ygz_means[i,:], ygz_covs[i]))
    end

    gmm_cond = MixtureModel(MvNormal, mixtureDef, yz_weights)
    logpdf(gmm_cond, y)
        


end


function covexpand_likelihood(z_input, μ_pr, Γ_pr, Γ_obs, Q, O, y; offset=0) # use the linearized likelihood

    Qz = Q * z_input#+ x0 
    # gx = fwdtoy(Qz + μ_pr) 
    # gdx = dfwdtoy(Qz + μ_pr)
    gx = aoe_fwdfun(Qz + μ_pr)
    gdx = aoe_gradfwdfun(Qz + μ_pr, gx)

    Γ_Δ = Γ_obs + gdx * (Γ_pr - Q * O * Γ_pr) * gdx'
    invΓ_obs = inv(cholesky(tril(Γ_Δ)+ tril(Γ_Δ,-1)'))
    loglikelihood = -1/2 * (y - gx)' * invΓ_obs * (y - gx) - 1/2 * logdet(Γ_Δ)
    
    loglikelihood
end