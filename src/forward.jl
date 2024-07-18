using LinearAlgebra, Random, Distributions, GaussianMixtures
using AOE

λ_ranges = [400.0 1300.0; 1450.0 1780.0; 2051.0 2451.0]
priormodel, wls = get_priormodel(:standard; λ_ranges) 
rtmodel = AOE.get_radiative_transfer(:modtran; λ_ranges);

function aoe_fwdfun(x)
    xa = x[1:2]
    xs = x[3:end]
    AOE.fwdfun(xa, xs, rtmodel) 
end

function aoe_gradfwdfun(x, fx)
    xa = x[1:2]
    xs = x[3:end]
    AOE.gradfwd_accel(xa, xs, rtmodel, fx) 
end


function fwdtoy(x)
    x1, x2, x3 = x
    g1 = cos(x1)+ x2^2 
    g2 = x1 - x3 
    g3 = x1^2 + sin(x2)
    # g1 = x1 + x2
    # g2 = x1 - x3 
    # g3 = x1 + x2 + x3
    return [g1; g2; g3]
end

function dfwdtoy(x)
    x1, x2, x3 = x
    dg = [[-sin(x1) 2*x2 0]; 
        [1 0 -1]; 
        [2*x1 cos(x2) 0]]
    # dg = [[1 1 0]; 
    #     [1 0 -1]; 
    #     [1 1 1]]
    return dg
end

# function fwdtoy(x)
#     x1, x2, x3 = x
#     g1 = x1 + x2^2 
#     g2 = x1 - x3 
#     # g3 = x1^2 + cos(x2)
#     g3 = x1^2 + x2
#     return [g1; g2; g3]
# end

# function dfwdtoy(x)
#     x1, x2, x3 = x
#     dg = [[1 2*x2 0]; 
#         [1 0 -1]; 
#         [2*x1 1 0]]
#     return dg
# end

# function fwdtoy(x)
#     x1, x2 = x
#     g1 = x1 + 10*x2
#     g2 = x1 - x2
#     return [g1; g2]
# end

# function dfwdtoy(x)
#     x1, x2 = x
#     dg = [[1 10]; 
#         [1 -1]]
#     return dg
# end


function cdr_likelihood(z, μ_pr, invΓ_obs, Q, O, y)
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
    logsumexp(expTerm) #+ log(1/m)
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

function gmm_pos_samp(gmm, y, m)
    yz_weights = weights(gmm)
    yz_means = means(gmm)
    yz_covs = covars(gmm)

    nComp = length(yz_weights)
    ygz_means = zeros(nComp)
    ygz_covs = zeros(nComp)

    for i in 1:nComp
        invcovyy = inv(yz_covs[i][1:end-1,1:end-1])
        ygz_covs[i] = yz_covs[i][end,end] - dot(yz_covs[i][end,1:end-1], invcovyy * yz_covs[i][1:end-1,end])
        ygz_means[i] = yz_means[i,end] + dot(yz_covs[i][end,1:end-1], invcovyy * (y - yz_means[i,1:end-1]) ) 
    end

    mixtureDef = []
    for i in 1:nComp
        push!(mixtureDef, (ygz_means[i], ygz_covs[i]))
    end

    gmm_cond = MixtureModel(Normal, mixtureDef, yz_weights)
    
    rand(gmm_cond, m)
end

function covexpand_likelihood(z_input, μ_pr, Γ_pr, Γ_obs, Q, O, y) # use the linearized likelihood

    Qz = Q * z_input
    # gx = fwdtoy(Qz + μ_pr) 
    # gdx = dfwdtoy(Qz + μ_pr)
    gx = aoe_fwdfun(Qz + μ_pr)
    gdx = aoe_gradfwdfun(Qz + μ_pr, gx)

    Γ_Δ = Γ_obs + gdx * (Γ_pr - Q * O * Γ_pr) * gdx'
    invΓ_obs = inv(cholesky(tril(Γ_Δ)+ tril(Γ_Δ,-1)'))
    loglikelihood = -1/2 * (y - gx)' * invΓ_obs * (y - gx) - 1/2 * logdet(Γ_Δ)
    
    loglikelihood
end