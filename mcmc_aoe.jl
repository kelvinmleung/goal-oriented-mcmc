include("inverseProblem.jl")
using Random, LinearAlgebra, Distributions
using AOE
λ_ranges = [400.0 1300.0; 1450.0 1780.0; 2051.0 2451.0]
priormodel, wls = get_priormodel(:standard; λ_ranges) # PriorModel instance
rtmodel = AOE.get_radiative_transfer_modtran(:LUTRT1; λ_ranges);


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
    propCovAtm = (2.38^2) / r * [0.01 0; 0 0.01] #Γ_pr[1:2,1:2] / 100
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

        ###### Begin truncated proposal ######
        # use this truncated proposal!
        # propDist = Truncated(MvNormal(x[1:2], propCholAtm * propCholAtm'), zeros(2), ones(2)*Inf)

        # propDist1 = Truncated(Normal(x[1], propCholAtm[1,1]^2), 0., Inf)
        # propDist2 = Truncated(Normal(x[2], propCholAtm[2,2]^2), 0., Inf)

        # z_atm[1:2] = rand(propDist)
        # println(rand(propDist))
        # z_atm[1] = rand(propDist1)
        # z_atm[2] = rand(propDist2)

        # println(x[1:2])
        # println(z_atm[1:2])
        ###### End truncated proposal ######

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
            # propCholAtm = cholesky(tril(propCovAtm) + tril(propCovAtm,-1)').L
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
