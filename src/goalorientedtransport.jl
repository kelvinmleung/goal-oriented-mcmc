using TransportBasedInference, StatsPlots, NPZ, AOE, KernelDensity, Random, Distributions
using LinearAlgebra

struct GODRdata{T}
    x_true::Vector{T}
    z_true::Vector{T}
    y::Vector{T}
    O::AbstractMatrix{T}
    O_offset::Vector{T}
    μ_x::Vector{T}
    μ_z::Vector{T}
    Γ_x::AbstractMatrix{T}
    Γ_z::AbstractMatrix{T}
    Γ_ϵ::AbstractMatrix{T}
    invΓ_x::AbstractMatrix{T}
    invΓ_z::AbstractMatrix{T}
    invΓ_ϵ::AbstractMatrix{T}
end

struct EnsembleSamples{T}
    fx::AbstractMatrix{T}
    y::AbstractMatrix{T}
    z::AbstractMatrix{T}
end

Random.seed!(123);

function initialize_GODRdata(n::Int, p::Int; T::Type = Float64)
    GODRdata(zeros(T, n), zeros(T, p), zeros(T, n), zeros(T, (p,n+2)), zeros(T, p), zeros(T, n+2), zeros(T, p), zeros(T, (n+2,n+2)), zeros(T, (p,p)), zeros(T, (n,n)), zeros(T, (n+2,n+2)), zeros(T, (p,p)), zeros(T, (n,n)))
end

function initialize_GODRdata_toy(n::Int, p::Int; T::Type = Float64)
    GODRdata(zeros(T, n), zeros(T, p), zeros(T, n), zeros(T, (p,n)), zeros(T, p), zeros(T, n), zeros(T, p), zeros(T, (n,n)), zeros(T, (p,p)), zeros(T, (n,n)), zeros(T, (n,n)), zeros(T, (p,p)), zeros(T, (n,n)))
end

function initialize_EnsembleSamples(n::Int, p::Int, m::Int; T::Type = Float64)
    EnsembleSamples(zeros(T, (n,m)), zeros(T, (n,m)), zeros(T, (p,m)))
end

function GODRdata_from_10pix!(setup::GODRdata{T}, indx::Int, indy::Int; n::Int = 326, p::Int = 1) where T <: Real
    prmean = npzread("data/data_refl/10pix/prmean_10pix.npy")
    prcov = npzread("data/data_refl/10pix/prcov_10pix.npy")
    prclass = npzread("data/data_refl/10pix/prclass_10pix.npy")
    prscale = npzread("data/data_refl/10pix/prscale_10pix.npy")
    s_all = npzread("data/data_refl/10pix/s_10pix.npy")
    y_all = npzread("data/data_refl/10pix/y_10pix.npy")

    setup.O .= vcat(zeros(2), load("data/data_canopy/goal_op_unscaled.jld")[string(Int(prclass[indx,indy]))])' / prscale[indx,indy]
    setup.O_offset .= load("data/data_canopy/goal_op_const_unscaled.jld")[string(Int(prclass[indx,indy]))]
    setup.x_true .= npzread("x_true_prsamp.npy") # s_all[indx,indy,:]  #
    setup.z_true .= setup.O[3:end]' * setup.x_true .+ setup.O_offset
    setup.y .= aoe_fwdfun(vcat([0.2,1.45], setup.x_true)) #y_all[indx,indy,:] + rand(MvNormal(zeros(n), diagm(AOE.dummy_noisemodel(y_all[indx,indy,:]))))

    # plot(aoe_fwdfun(vcat([0.2,1.5], setup.x_true)))
    # display(plot!(setup.y))

    setup.μ_x .= vcat([0.2; 1.45], prmean[indx,indy,:])
    setup.Γ_x[1:2,1:2] .= [0.01 0; 0 0.004]
    setup.Γ_x[3:end,3:end] .= prcov[indx,indy,:,:]
    setup.Γ_ϵ .= diagm(AOE.dummy_noisemodel(setup.y)) 

    setup.μ_z .= setup.O * setup.μ_x 
    setup.Γ_z .= setup.O * setup.Γ_x * setup.O'
    setup.invΓ_x .= inv(setup.Γ_x)
    setup.invΓ_z .= inv(setup.Γ_z)
    setup.invΓ_ϵ .= inv(setup.Γ_ϵ)

    setup
end

function GODRdata_toy!(setup::GODRdata{T}; n::Int = 2, p::Int = 1) where T <: Real

    G = [10. 2.; -1 2]

    setup.μ_x .= [0., 0.]
    setup.Γ_x .= [1. 0.; 0. 0.5]
    setup.Γ_ϵ .= [0.1 0.; 0 0.1]
    setup.O .= [0.1, 0.9]'
    setup.x_true .= [1., 1.]
    setup.z_true .= setup.O * setup.x_true
    setup.y .= G * setup.x_true + rand(MvNormal(zeros(n), setup.Γ_ϵ))

    setup.μ_z .= setup.O * setup.μ_x 
    setup.Γ_z .= setup.O * setup.Γ_x * setup.O'
    setup.invΓ_x .= inv(setup.Γ_x)
    setup.invΓ_z .= inv(setup.Γ_z)
    setup.invΓ_ϵ .= inv(setup.Γ_ϵ)

    Γ_xgy = inv(G' * setup.invΓ_ϵ * G + setup.invΓ_x)
    μ_xgy = Γ_xgy * (setup.invΓ_x * setup.μ_x + G' * setup.invΓ_ϵ * setup.y)

    setup, G, μ_xgy, Γ_xgy
end

function replace_with_linearized_model!(setup::GODRdata{T}) where T <: Real

    fx = aoe_fwdfun(setup.μ_x)
    gradG = aoe_gradfwdfun(setup.μ_x, fx)



    # plot(setup.y, label="nonlinear")
    setup.y .= gradG * (vcat([0.2,1.45], setup.x_true)) # fx +  - setup.μ_x) 
    # display(plot!(setup.y, label="linear"))
    Γ_xgy = inv(gradG' * setup.invΓ_ϵ * gradG + setup.invΓ_x)
    μ_xgy = Γ_xgy * (setup.invΓ_x * setup.μ_x + gradG' * setup.invΓ_ϵ * setup.y)
    # plot(setup.μ_x[3:end], label="prior mean")
    # display(plot!(μ_xgy[3:end], label="posterior mean"))
    # display(plot!(setup.x_true, label="truth"))

    fx, gradG, μ_xgy, Γ_xgy
end

function gen_pr_samp(setup::GODRdata{T}, m::Int ; n::Int = 326, p::Int = 1) where T <: Real

    π_x = MvNormal(setup.μ_x, setup.Γ_x)
    π_ϵ = MvNormal(zeros(n), setup.Γ_ϵ)

    prsamp = initialize_EnsembleSamples(n, p, m)
    x_samp = rand(π_x, m)
    prsamp.z .= setup.O * x_samp
    

    println("Applying forward model to prior samples...")
    @time for i = 1:m
        prsamp.fx[:,i] .= aoe_fwdfun(x_samp[:,i])
    end

    prsamp.y .= prsamp.fx + rand(π_ϵ, m) 
    prsamp
end

function diagnostic_matrix(N::Int, invsqrtΓ_y)
    # N = 10000
    H = zeros((n,n))
    # invsqrtΓ_ϵ = inv(sqrt(setup.Γ_ϵ))
    println("Monte Carlo estimate of diagnostic matrix...")
    @time for i = 1:N
        randx = rand(MvNormal(setup.μ_x, setup.Γ_x))
        fx = aoe_fwdfun(randx)
        dfx = aoe_gradfwdfun(randx, fx)
        # H = H + 1/N * invsqrtΓ_ϵ * dfx * setup.Γ_x * setup.O' * setup.invΓ_z * setup.O * setup.Γ_x * dfx' * invsqrtΓ_ϵ
        H = H + 1/N * invsqrtΓ_y * dfx * setup.Γ_x * setup.O' * setup.invΓ_z * setup.O * setup.Γ_x * dfx' * invsqrtΓ_y

    end
    H
end

function diagnostic_matrix_no_goal(N::Int, invsqrtΓ_y)
    # N = 10000
    H = zeros((n,n))
    println("Monte Carlo estimate of diagnostic matrix...")
    @time for i = 1:N
        randx = rand(MvNormal(setup.μ_x, setup.Γ_x))
        fx = aoe_fwdfun(randx)
        dfx = aoe_gradfwdfun(randx, fx)
        H = H + 1/N * invsqrtΓ_y * dfx * setup.Γ_x * dfx' * invsqrtΓ_y
    end
    H
end


function diagnostic_matrix_nonwhiten(N::Int)
    # N = 10000
    H = zeros((n,n))
    invsqrtΓ_ϵ = inv(sqrt(setup.Γ_ϵ))
    println("Monte Carlo estimate of diagnostic matrix...")
    @time for i = 1:N
        randx = rand(MvNormal(setup.μ_x, setup.Γ_x))
        fx = aoe_fwdfun(randx)
        dfx = aoe_gradfwdfun(randx, fx)
        H = H + 1/N * dfx * reshape(setup.O', n+2,1) * setup.O * dfx' 
    end
    H
end

function pca_matrix(covy)
    invstdy = diagm(1 ./ sqrt.(diag(covy)))
    invstdy * covy * invstdy
end

function diagnostic_eigendecomp(H::AbstractMatrix{T}; showplot::Bool = false, setup::GODRdata{T}) where T <: Real
    # eigenvalues
    
    eigs = real.(eigvals(H)[end:-1:1])
    # eigenvectors
    V = real.(eigvecs(H)[:,end:-1:1])
    if showplot
        p = plot(title="Leading eigenvectors")
        plot!(wls, setup.O[3:end] ./ norm(setup.O[3:end]), linewidth=1, color=:black, label="O")
        for i in 1:1
            if maximum(abs.(V[50:end,i])) == maximum(V[50:end,i])
                plot!(wls,V[:,i], linewidth=2) #sqrt(setup.Γ_ϵ)* 
            else
                plot!(wls,-V[:,i], linewidth=2) # sqrt(setup.Γ_ϵ)* 
            end
        end
        display(p)
        display(plot(eigs, yaxis=:log, label="", title="Diagnostic Matrix - Whitened"))
    end
    
    eigs, V 
end

function energy_cutoff(Λy::AbstractVector{T}, ratio::T; rydefault = 50) where T <: Real
    ry = ratio < 1.0 ? findfirst(x-> x >= ratio, cumsum(Λy)./sum(Λy)) : rydefault
    ry = isnothing(ry) ? 1 : ry 
end

function whiten_samples(setup::GODRdata{T}, samps::EnsembleSamples{T}, μ_y::Vector{T}, V_r::AbstractMatrix{T}) where T <: Real
    
    invsqrtΓ_y = sqrt(inv(cov(samps.y,dims=2)))
    invsqrtΓ_ϵ = sqrt(setup.invΓ_ϵ)
    invsqrtΓ_z = inv(sqrt(setup.Γ_z))
    # y_whiten = V_r' * (invsqrtΓ_y * (samps.y .- μ_y))

    covy = cov(V_r' * (invsqrtΓ_ϵ * (samps.y .- μ_y)), dims=2)
    # y_whiten = inv(sqrt(covy)) * V_r' * (invsqrtΓ_ϵ * (samps.y .- μ_y))
    y_whiten = V_r' * invsqrtΓ_ϵ * samps.y 
    # y_whiten = V_r' * (invsqrtΓ_ϵ * (samps.y .- μ_y))
    # z_whiten = invsqrtΓ_z * (samps.z .- setup.μ_z)
    z_whiten = samps.z
    
    # yobs_whiten = V_r' * (invsqrtΓ_y * (setup.y - μ_y))
    # yobs_whiten = V_r' * (invsqrtΓ_ϵ * (setup.y - μ_y))
    # yobs_whiten = inv(sqrt(covy)) * V_r' * (invsqrtΓ_ϵ * (setup.y - μ_y))
    yobs_whiten = V_r' * invsqrtΓ_ϵ * setup.y 
    

    
    # y_whiten = V_r' * (samps.y .- μ_y)
    # z_whiten = samps.z .- setup.μ_z
    # yobs_whiten = V_r' * (setup.y .- μ_y)
    y_whiten, yobs_whiten, z_whiten
end

function kldiv(μ1, Σ1, μ2, Σ2)
    invΣ2 = inv(Σ2)
    # https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
    0.5 * ( logdet(Σ2) - logdet(Σ1) - length(μ1) + (μ1 - μ2)' * invΣ2 * (μ1 - μ2) + tr(invΣ2 * Σ1) )
end


function cond_gaussian_analytic(Vr, G, O, μ_x, Γ_x, Γ_y, y)
    μ_z = O * μ_x
    Γ_z = O * Γ_x * O'
    cov_vry_z = Vr' * G * Γ_x * O' 
    cov_vry = Vr' * Γ_y * Vr
    μ_z_g_vry = μ_z .- cov_vry_z' * inv(cov_vry) * (Vr' * G * μ_x - Vr' * y)
    Γ_z_g_vry = Γ_z - cov_vry_z' * inv(cov_vry) * cov_vry_z
    μ_z_g_vry, Γ_z_g_vry
end

function apply_cond_gaussian(X::AbstractMatrix{T}, yobs::AbstractVector{T}; whitened=false) where T <: Real #, V, setup, μ_y
    jointmean = mean(X, dims=2)
    jointcov = cov(X, dims=2)
    μ_y = @view jointmean[1:end-1]
    μ_z = @view jointmean[end]
    Σ_zz = @view jointcov[end, end]
    Σ_yz = @view jointcov[1:end-1, end]

    # display(jointcov)
    if whitened
        μ_zgy = Σ_yz'* yobs
        Σ_zgy = Σ_zz .- Σ_yz' * Σ_yz
    else
        invΣ_yy = inv(jointcov[1:end-1, 1:end-1])
        μ_zgy = μ_z .- Σ_yz' * invΣ_yy * (μ_y - yobs)
        Σ_zgy = Σ_zz .- Σ_yz' * invΣ_yy * Σ_yz
    end
       
    μ_zgy[1], Σ_zgy
end

function apply_cond_transport(X::AbstractMatrix{T}, Ystar::AbstractMatrix{T}, Ny::Int; order::Int = 10) where T <: Real
    # S = HermiteMap(order, X; diag = true, factor = 1., α = 1e-6, b = "CstProHermiteBasis");
    S = HermiteMap(order, X; diag = true, factor = 0.5, α = 1e-6, b = "ProHermiteBasis");
    @time S = optimize(S, X, "split"; maxterms = 30, withconstant = true, withqr = true, verbose = true, 
                                  maxpatience = 30, start = 1, hessprecond = true)
    
    # plot(S; start = Ny+1)
    # plot(S)
    F = evaluate(S, X; start = Ny+1)
    # F0 = deepcopy(F)
    # @time hybridinverse!(Xa[], F, S, Ystar; apply_rescaling = true, start = Ny+1)
    Xa = deepcopy(X)
    @time hybridinverse!(Xa, F, S, Ystar; apply_rescaling = true, start = Ny+1)
    
    Xa[end,:], S, F
end

