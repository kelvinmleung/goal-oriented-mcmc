using Distributed, Random, Base, LinearAlgebra, JLD

function logpos_simple(x, μ_pr, invΓ_pr, invΓ_obs, y)
    gx = fwdtoy(x)
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

function mcmc_amm_simple(x0, μ_pr, Γ_pr, Γ_obs, y, N)

    r = length(μ_pr)
    x_vals = zeros(r, N)
    invΓ_pr, invΓ_obs = inv(Γ_pr), inv(Γ_obs)
    logpos, accept = zeros(N), zeros(N)
    propcov = (2.38^2) / r * Γ_pr #diagm(ones(r))
    propChol = cholesky(propcov).L 
    sd, eps = 2.38^2 / r, 1e-10
    meanXprev = zeros(r)

    x = x0 #zeros(r)
    # x[:,1] = x0
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
