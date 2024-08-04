include("forward.jl")
using Random, LinearAlgebra, Distributions
using AOE
λ_ranges = [400.0 1300.0; 1450.0 1780.0; 2051.0 2451.0]
priormodel, wls = get_priormodel(:standard; λ_ranges) # PriorModel instance
rtmodel = AOE.get_radiative_transfer(:modtran; λ_ranges);


function logpos_bm(x, μ_pr, invΓ_pr, invΓ_obs, y)
    gx = aoe_fwdfun(x)
    -1/2 * (x - μ_pr)' * invΓ_pr * (x - μ_pr) - 1/2 * (y - gx)' * invΓ_obs * (y - gx)
end

function alpha_bm(x, z, μ_pr, invΓ_pr, invΓ_obs, y)
    lpz = logpos_bm(z, μ_pr, invΓ_pr, invΓ_obs, y)
    lpx = logpos_bm(x, μ_pr, invΓ_pr, invΓ_obs, y)
    return minimum((1, exp(lpz-lpx))), lpz, lpx
end

function proposal(μ, cholΓ)
    n = length(μ)
    mvn = MvNormal(zeros(n), diagm(ones(n)))
    μ + cholΓ * rand(mvn)
end

function mcmc_bm(μ_pr, Γ_pr, Γ_obs, y, N) 
    # block metropolis

    r = 2
    x_vals = zeros(length(μ_pr), N)
    invΓ_pr, invΓ_obs = inv(Γ_pr), inv(Γ_obs)

    rdbufs = get_RetrievalData_bufs(length(wls)) 
    xa, xs = AOE.invert(y, rdbufs[1], rtmodel, priormodel)
    fx = AOE.fwdfun(xa, xs, rtmodel) 
    dfx = AOE.gradfwd_accel(xa, xs, rtmodel, fx)[:,3:end]

    dfx = AOE.gradfwd_accel(xa, xs, rtmodel, fx)[:,3:end]
    # propCovRefl = Γ_pr[3:end,3:end] / 200#  inv(inv(Γ_x[3:end,3:end]) + dfx' * inv(Γ_ϵ) * dfx)
    propCovRefl = inv(invΓ_pr[3:end,3:end] + dfx' * invΓ_obs * dfx) 
    propCholRefl = cholesky(tril(propCovRefl) + tril(propCovRefl,-1)').L * 0.11


    acceptAtm, acceptRefl = zeros(N), zeros(N)
    propCovAtm = (2.38^2) / r * Γ_pr[1:2,1:2] / 100 #[0.01 0; 0 0.01] #
    propCholAtm = sqrt.(propCovAtm)
    sd, eps = 2.38^2 / r, 1e-10
    meanXprev = zeros(r)

    # x = x0 
    x_atm = xa# [0.2,1.5] #
    x_refl = xs
    x = vcat(x_atm, x_refl)
    # x = μ_pr

    for i in 1:N

        z_atm = copy(x)
        z_atm[1:2] = proposal(x[1:2], propCholAtm)

        α_atm, logposZ, logposX = alpha_bm(x, z_atm, μ_pr, invΓ_pr, invΓ_obs, y)
        if rand() < α_atm
            x[1:2] = z_atm[1:2]
            # logposX = logposZ
            acceptAtm[i] += 1
            # x_atm = x[1:2]
        end

        z_ref = copy(x)
        z_ref[3:end] = proposal(x[3:end], propCholRefl)
        α_ref, logposZ, logposX = alpha_bm(x, z_ref, μ_pr, invΓ_pr, invΓ_obs, y)
        if rand() < α_ref
            x[3:end] = z_ref[3:end]
            # logposX = logposZ
            acceptRefl[i] += 1
        end

        x_vals[:,i] = x

        if i % 100 == 0
            if mean(acceptAtm[i-99:i]) < 0.1
                propCholAtm = propCholAtm / sd
            end
        end
        
        if i % 500 == 0
            if i % 1000 == 0
                display("Sample: " * string(i))
                display("   Atm Accept Rate: " * string(mean(acceptAtm[i-499:i])))
                display("   Ref Accept Rate: " * string(mean(acceptRefl[i-499:i])))
                display(cov(x_vals[1:2,i-499:i], dims=2))
            end
            # propCovAtm = sd * cov(x_vals[1:2,i-499:i], dims=2) + eps * I ####T RY THIS FOR NOW
            propCholAtm = cholesky(tril(propCovAtm) + tril(propCovAtm,-1)').L
            # display(propCholAtm)
        end

        if i == 1500
            propCovAtm = sd * cov(x_vals[1:2,1:1500], dims=2) + eps * I
            meanXprev = mean(x_vals[1:2,1:1500], dims=2)
        elseif i > 1500
            meanX = i / (i + 1) * meanXprev + 1 / (i + 1) * x_vals[1:2,i]
            propCovAtm = (i-1) / i * propCovAtm + sd / i * (i * meanXprev * meanXprev' - (i+1) * meanX * meanX' + x_vals[1:2,i] * x_vals[1:2,i]' + eps * I)
            meanXprev = meanX
        end
    end
    # display(plot(logpos[Int(N/10):end]))
    return x_vals
end


function mcmc_bm_3block(μ_pr, Γ_pr, Γ_obs, y, N; xa=[0.2,1.45]) 
    # block metropolis

    r = 2
    x_vals = zeros(length(μ_pr), N)
    invΓ_pr, invΓ_obs = inv(Γ_pr), inv(Γ_obs)

    rdbufs = get_RetrievalData_bufs(length(wls)) 
    _, xs = AOE.invert(y, rdbufs[1], rtmodel, priormodel; xa0guess=xa)
    
    fx = AOE.fwdfun(xa, xs, rtmodel) 
    dfx = AOE.gradfwd_accel(xa, xs, rtmodel, fx)[:,3:end]

    dfx = AOE.gradfwd_accel(xa, xs, rtmodel, fx)[:,3:end]
    propCovRefl = inv(invΓ_pr[3:end,3:end] + dfx' * invΓ_obs * dfx) 
    propCholRefl = cholesky(tril(propCovRefl) + tril(propCovRefl,-1)').L * 0.11


    acceptAOD, acceptH2O, acceptRefl = zeros(N), zeros(N), zeros(N)
    propCovAOD, propCovH2O = (2.38^2) * Γ_pr[1,1] / 100, (2.38^2) * Γ_pr[2,2] / 100
    propCholAOD, propCholH2O = sqrt(propCovAOD), sqrt(propCovH2O)
    
    sd, eps = 2.38^2 , 1e-10
    meanXprevAOD, meanXprevH2O = 0., 0.

    # x = x0 
    x_aod = xa[1]# [0.2,1.5] #
    x_h2o = xa[2]
    x_refl = xs
    x = vcat(x_aod, x_h2o, x_refl)
    # x = μ_pr

    for i in 1:N

        z_atm = copy(x)
        
        propDistAOD = Truncated(Normal(x[1], propCholAOD), 0.001, 0.5)
        propDistH2O = Truncated(Normal(x[2], propCholH2O), 1.31, 1.59)

        # z_atm[1:2] = rand(propDist)
        # println(rand(propDist))
        z_atm[1] = rand(propDistAOD)

        α_aod, logposZ, logposX = alpha_bm(x, z_atm, μ_pr, invΓ_pr, invΓ_obs, y)
        if rand() < α_aod
            x[1] = z_atm[1]
            acceptAOD[i] += 1
        end

        z_atm[2] = rand(propDistH2O)

        α_h2o, logposZ, logposX = alpha_bm(x, z_atm, μ_pr, invΓ_pr, invΓ_obs, y)
        if rand() < α_h2o
            x[2] = z_atm[2]
            acceptH2O[i] += 1
        end

        z_ref = copy(x)
        z_ref[3:end] = proposal(x[3:end], propCholRefl)
        α_ref, logposZ, logposX = alpha_bm(x, z_ref, μ_pr, invΓ_pr, invΓ_obs, y)
        if rand() < α_ref
            x[3:end] = z_ref[3:end]
            acceptRefl[i] += 1
        end

        x_vals[:,i] = x

        if i % 100 == 0
            if mean(acceptAOD[i-99:i]) < 0.1
                propCholAOD = propCholAOD / sd
            end
            if mean(acceptH2O[i-99:i]) < 0.1
                propCholH2O = propCholH2O / sd
            end
        end
        
        if i % 500 == 0
            if i % 1000 == 0
                display("Sample: " * string(i))
                display("   AOD Accept Rate: " * string(mean(acceptAOD[i-499:i])))
                display("   H2O Accept Rate: " * string(mean(acceptH2O[i-499:i])))
                display("   Ref Accept Rate: " * string(mean(acceptRefl[i-499:i])))
                display("   Variance: " * string([cov(x_vals[1,i-499:i]) cov(x_vals[2,i-499:i])]))
            end
            # propCovAtm = sd * cov(x_vals[1:2,i-499:i], dims=2) + eps * I ####T RY THIS FOR NOW
            propCholAOD = sqrt(propCovAOD)
            propCholH2O = sqrt(propCovH2O)
            # display(propCholAtm)
        end

        if i == 1500
            propCovAOD = sd * var(x_vals[1,1:1500]) + eps
            propCovH2O = sd * var(x_vals[2,1:1500]) + eps
            meanXprevAOD = mean(x_vals[1,1:1500])
            meanXprevH2O = mean(x_vals[2,1:1500])
        elseif i > 1500
            meanXAOD = i / (i + 1) * meanXprevAOD + 1 / (i + 1) * x_vals[1,i]
            propCovAOD = (i-1) / i * propCovAOD + sd / i * (i * meanXprevAOD^2 - (i+1) * meanXAOD^2 + x_vals[1,i]^2 + eps)
            meanXprevAOD = meanXAOD

            meanXH2O = i / (i + 1) * meanXprevH2O + 1 / (i + 1) * x_vals[2,i]
            propCovH2O = (i-1) / i * propCovH2O + sd / i * (i * meanXprevH2O^2 - (i+1) * meanXH2O^2 + x_vals[2,i]^2 + eps)
            meanXprevH2O = meanXH2O
        end
    end
    # display(plot(logpos[Int(N/10):end]))
    return x_vals
end