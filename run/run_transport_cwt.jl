include("../src/forward.jl")
include("../src/mcmc_simple.jl")
include("../src/mcmc_1d.jl")

Random.seed!(123)
using TransportBasedInference, StatsPlots, NPZ, AOE, KernelDensity

λ_ranges = [400.0 1300.0; 1450.0 1780.0; 2051.0 2451.0]
priormodel, wls = get_priormodel(:standard; λ_ranges) # PriorModel instance
rtmodel = AOE.get_radiative_transfer(:modtran; λ_ranges);
n, p = 326, 1
m = 100000
m_naive = 3000000

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
    y::AbstractMatrix{T}
    z::AbstractMatrix{T}
end

function initialize_GODRdata(n::Int, p::Int; T::Type = Float64)
    GODRdata(zeros(T, n), zeros(T, p), zeros(T, n), zeros(T, (p,n+2)), zeros(T, p), zeros(T, n+2), zeros(T, p), zeros(T, (n+2,n+2)), zeros(T, (p,p)), zeros(T, (n,n)), zeros(T, (n+2,n+2)), zeros(T, (p,p)), zeros(T, (n,n)))
end

function initialize_EnsembleSamples(n::Int, p::Int, m::Int; T::Type = Float64)
    EnsembleSamples(zeros(T, (n,m)), zeros(T, (p,m)))
end

function GODRdata_from_10pix!(setup::GODRdata{T}, indx::Int, indy::Int; n::Int = 326, p::Int = 1) where T <: Real
    prmean = npzread("data/data_refl/10pix/prmean_10pix.npy")
    prcov = npzread("data/data_refl/10pix/prcov_10pix.npy")
    prclass = npzread("data/data_refl/10pix/prclass_10pix.npy")
    prscale = npzread("data/data_refl/10pix/prscale_10pix.npy")
    s_all = npzread("data/data_refl/10pix/s_10pix.npy")
    y_all = npzread("data/data_refl/10pix/y_10pix.npy")

    setup.O .= vcat(zeros(2), npzread("data/data_canopy/goal_op_"* string(Int(prclass[indx,indy])) *"_unscaled.npy"))' / prscale[indx,indy]
    setup.O_offset .= npzread("data/data_canopy/goal_op_const_"* string(Int(prclass[indx,indy])) *"_unscaled.npy")
    setup.x_true .= s_all[indx,indy,:] 
    setup.z_true .= setup.O[3:end]' * setup.x_true .+ setup.O_offset
    setup.y .= y_all[indx,indy,:] + rand(MvNormal(zeros(n), diagm(AOE.dummy_noisemodel(y_all[indx,indy,:]))))

    setup.μ_x .= vcat([0.2; 1.3], prmean[indx,indy,:])
    setup.Γ_x[1:2,1:2] .= [0.01 0; 0 0.04]
    setup.Γ_x[3:end,3:end] .= prcov[indx,indy,:,:]
    setup.Γ_ϵ .= diagm(AOE.dummy_noisemodel(setup.y))

    setup.μ_z .= setup.O * setup.μ_x 
    setup.Γ_z .= setup.O * setup.Γ_x * setup.O'
    setup.invΓ_x .= inv(setup.Γ_x)
    setup.invΓ_z .= inv(setup.Γ_z)
    setup.invΓ_ϵ .= inv(setup.Γ_ϵ)

    setup
end

function gen_pr_samp(setup::GODRdata{T}, m::Int ; n::Int = 326, p::Int = 1) where T <: Real

    normDist = MvNormal(setup.μ_x, setup.Γ_x)
    noiseDist = MvNormal(zeros(n), setup.Γ_ϵ)

    prsamp = initialize_EnsembleSamples(n, p, m)
    x_samp = rand(normDist, m)
    prsamp.z .= setup.O * x_samp
    prsamp.y .= rand(noiseDist, m)

    println("Applying forward model to prior samples...")
    @time for i = 1:m
        prsamp.y[:,i] .= prsamp.y[:,i] + aoe_fwdfun(x_samp[:,i])
    end
    prsamp
end

function diagnostic_matrix(N::Int)
    # N = 10000
    H = zeros((n,n))
    invsqrtΓ_ϵ = inv(sqrt(setup.Γ_ϵ))
    println("Monte Carlo estimate of diagnostic matrix...")
    @time for i = 1:N
        randx = rand(MvNormal(setup.μ_x, setup.Γ_x))
        fx = aoe_fwdfun(randx)
        dfx = aoe_gradfwdfun(randx, fx)
        H = H + 1/N * invsqrtΓ_ϵ * dfx * setup.Γ_x * reshape(setup.O', n+2,1) * setup.invΓ_z * setup.O * setup.Γ_x * dfx' * invsqrtΓ_ϵ
    end
    H
end

function energy_cutoff(Λy::AbstractVector{T}, ratio::T; rydefault = 50) where T <: Real
    ry = ratio < 1.0 ? findfirst(x-> x >= ratio, cumsum(Λy)./sum(Λy)) : rydefault
    ry = isnothing(ry) ? 1 : ry 
end

function whiten_samples(setup::GODRdata{T}, samps::EnsembleSamples{T}, μ_y::Vector{T}, V_r::AbstractMatrix{T}) where T <: Real
    
    # r = size(V_r, 2)
    # whiten_samp = initialize_EnsembleSamples(r, p, m)
    invsqrtΓ_ϵ = inv(sqrt(setup.Γ_ϵ))
    invsqrtΓ_z = inv(sqrt(setup.Γ_z))
    y_whiten = V_r' * invsqrtΓ_ϵ * (samps.y .- μ_y)
    z_whiten = invsqrtΓ_z * (samps.z .- setup.μ_z)
    yobs_whiten = V_r' * invsqrtΓ_ϵ * (setup.y - μ_y)
    y_whiten, yobs_whiten, z_whiten
end

function apply_cond_transport(X::AbstractMatrix{T}, Ystar::AbstractVector{T}, Ny::Int; order::Int = 10) where T <: Real
    # S = HermiteMap(order, X; diag = true, factor = 1., α = 1e-6, b = "CstProHermiteBasis");
    S = HermiteMap(order, X; diag = true, factor = 0.5, α = 1e-6, b = "ProHermiteBasis");
    @time S = optimize(S, X, "split"; maxterms = 30, withconstant = true, withqr = true, verbose = true, 
                                  maxpatience = 30, start = Ny+1, hessprecond = true)
    

    # plot(S; start = Ny+1)
    F = evaluate(S, X; start = Ny+1)
    # F0 = deepcopy(F)
    # @time hybridinverse!(Xa[], F, S, Ystar; apply_rescaling = true, start = Ny+1)
    Xa = deepcopy(X)
    @time hybridinverse!(Xa, F, S, Ystar; apply_rescaling = true, start = Ny+1)
    
    Xa[end,:]
end



setup = initialize_GODRdata(n, p)

GODRdata_from_10pix!(setup, 1,1);
prsamp = gen_pr_samp(setup, m);
H = diagnostic_matrix(10000);

# eigenvalues
plot(eigvals(H)[end:-1:1],yaxis=:log, label="", title="Diagnostic Matrix - Whitened")
eigs = eigvals(H)[end:-1:1]

r = energy_cutoff(eigs, 0.9999999);

# eigenvectors
V = eigvecs(H)[:,end:-1:1]
V_r = V[:,1:r]
# p = plot(title="Leading eigenvectors", ylims=[-0.05,0.1])
# for i in 1:5
#     if maximum(abs.(V[50:end,i])) == maximum(V[50:end,i])
#         plot!(wls,sqrt(Γ_ϵ)* V[:,i], linewidth=2)
#     else
#         plot!(wls,-sqrt(Γ_ϵ)* V[:,i], linewidth=2)
#     end
# end
# display(p)

μ_y = vec(mean(prsamp.y, dims=2));
y_whiten, yobs_whiten, z_whiten = whiten_samples(setup, prsamp, μ_y, V_r);
X = vcat(y_whiten, z_whiten)
# scatter(y_whiten[1,1:100:end], z_whiten[1:100:end])

z_possamp_whiten = apply_cond_transport(X, yobs_whiten, r)
z_possamp_transport = sqrt(setup.Γ_z) .* z_possamp_whiten .+ setup.μ_z .+ setup.O_offset 


plotrange = 0.:0.001:0.5
kde_transport = kde(vec(z_possamp_transport), plotrange)

z_possamp_naive = npzread("data/data_canopy/june28/10pix_ind(1,1)/z_naive.npy")
z_possamp_covexpand = npzread("data/data_canopy/june28/10pix_ind(1,1)/z_covexpand.npy")
z_possamp_gmm = npzread("data/data_canopy/june28/10pix_ind(1,1)/z_gmm.npy")

density(z_possamp_naive[1000000:10:end], color=:black, linewidth=2, label="Full Rank MCMC", title="1D Goal Posterior - Marginal Density", legend=:topright, dpi=800, xlim=[0.1,0.4])
density!(z_possamp_gmm[2000:1:end], color=:red, linewidth=2, label="Low Rank MCMC")#, xlim=[0.15,0.3])
plot!(kde_transport.x, kde_transport.density, color=:green, linewidth=2, label="Transport")
display(plot!([setup.z_true], seriestype="vline", color=:black, linewidth=3, label="Truth"))

