include("../src/forward.jl")
include("../src/mcmc_simple.jl")
include("../src/mcmc_1d.jl")
include("../src/goalorientedtransport.jl")

Random.seed!(123)

λ_ranges = [400.0 1300.0; 1450.0 1780.0; 2051.0 2451.0]
priormodel, wls = get_priormodel(:standard; λ_ranges) # PriorModel instance
rtmodel = AOE.get_radiative_transfer(:modtran; λ_ranges);
n, p = 326, 1
m = 1000000


setup = initialize_GODRdata(n, p)
GODRdata_from_10pix!(setup, 1,1);
setup.x_true .= load("values_for_plot.jld", "x")
setup.y .= load("values_for_plot.jld", "y")
setup.z_true .= load("values_for_plot.jld", "z")

## Linear version to test
# fx, gradG, μ_xgy, Γ_xgy = replace_with_linearized_model!(setup);
# μ_zgy_true, Σ_zgy_true = setup.O * μ_xgy .+ setup.O_offset, setup.O * Γ_xgy * setup.O'

# make the samples adhere to the new linear model
prsamp = gen_pr_samp(setup, m);

covy = cov(prsamp.y, dims=2) # gradG * setup.Γ_x * gradG' + setup.Γ_ϵ #
invsqrtcovy = inv(sqrt(covy))
meany = mean(prsamp.y, dims=2)


H_mat = diagnostic_matrix(10000, invsqrtcovy)# invsqrtcovy * gradG * setup.Γ_x * setup.O' * setup.invΓ_z * setup.O * setup.Γ_x * gradG' * invsqrtcovy
H_nogoal = diagnostic_matrix_no_goal(10000, invsqrtcovy)# invsqrtcovy * gradG * setup.Γ_x * gradG' * invsqrtcovy
H_pca = covy #pca_matrix(covy)

Λ_godr, V_godr = diagnostic_eigendecomp(H_mat; showplot=false, setup=setup)
Λ_nogoal, V_nogoal = diagnostic_eigendecomp(H_nogoal; showplot=false, setup=setup)
Λ_pca, V_pca = diagnostic_eigendecomp(H_pca; showplot=false, setup=setup)

V_godr = invsqrtcovy * V_godr
V_nogoal = invsqrtcovy * V_nogoal

μ_zgy_true, Σ_zgy_true = apply_cond_gaussian(vcat(prsamp.y, prsamp.z), setup.y; whitened=false) 


# ### EIGENVALUE AND EIGENVECTOR PLOTS
# plot(Λ_godr ./ Λ_godr[1], yaxis=:log, label=false, linewidth=3, color=:blue4, dpi=300)
# savefig("plots/JSM/eigval.png")
# plot!(Λ_nogoal ./ Λ_nogoal[1], yaxis=:log, label=false, linewidth=3, color=:blue3)
# plot!(Λ_pca ./ Λ_pca[1], yaxis=:log, label=false, linewidth=3, color=:blue3)

# O_tilde = (inv(sqrt(setup.Γ_z)) *  setup.O * sqrt(setup.Γ_x))[3:end]
# wls_ind = collect(1:n)[abs.(O_tilde) .> 0.075]
# p = plot(dpi=300)
# for i in wls_ind
#     plot!([wls[i]], seriestype=:vline, color=:lightblue, linewidth=2, alpha=0.5, label=false)
# end
# plot!(wls, -sqrt(covy) * V_godr[:,1], linewidth=2, color=:darkblue, label=false, ylims=[-0.2,0.15])
# for i in vcat(1300:5:1450, 1780:5:2050)
#     plot!([i], seriestype=:vline, color=:lightgrey, linewidth=2, label=false)
# end

# plot!([-0.2], seriestype=:hline, color=:black, linewidth=3, alpha=0.2, label=false)
# savefig("plots/JSM/eigvec.png")

# display(p)

# ### PLOT OF MCMC VS TRANSPORT
# z_possamp_naive = npzread("data/data_canopy/aug2/10pix_ind(1,1)/z_naive.npy")
# z_possamp_transport_godr = npzread("data/data_canopy/aug2/10pix_ind(1,1)/z_transport_godr.npy")
# z_possamp_transport_nogoal = npzread("data/data_canopy/aug2/10pix_ind(1,1)/z_transport_nogoal.npy")

# plot([setup.z_true], seriestype="vline", color=:black, linestyle=:dot,linewidth=3, label="Truth", dpi=300, xlims=[0,0.3], ylims=[0,100], legend=:topleft)
# density!(z_possamp_naive[2000000:100:end], linewidth=4, linestyle=:dash, label="MCMC, rank 326 (full)", color=:blue4)
# density!(z_possamp_transport_nogoal[1:10:end], color=:green, linewidth=2, label="SBI without goal, rank 2")
# density!(z_possamp_transport_godr[1:10:end], color=:red, linewidth=2, label="SBI goal-oriented, rank 2")

# savefig("plots/JSM/poszgy.png")




#### PLOT OF ERROR COMPARISON
N_mc = 100
dimlist = vcat(1:9,10:5:45,50:10:150,160:20:320,326)
numdim = length(dimlist)
μ_zgy_vec, Σ_zgy_vec= zeros(N_mc), zeros(N_mc)
μ_zgy_godr, μ_zgy_nogoal, μ_zgy_pca = zeros(numdim, N_mc), zeros(numdim, N_mc), zeros(numdim, N_mc)
Σ_zgy_godr, Σ_zgy_nogoal, Σ_zgy_pca = zeros(numdim, N_mc), zeros(numdim, N_mc), zeros(numdim, N_mc)
z_true = zeros(N_mc)
Random.seed!(123)
pry, prz = prsamp.y[:,1:100:end], prsamp.z[:,1:100:end]
for i in 1:N_mc
    
    fullx = rand(MvNormal(setup.μ_x, setup.Γ_x))
    setup.x_true .= fullx[3:end]
    setup.z_true .= setup.O[3:end]' * setup.x_true .+ setup.O_offset
    setup.y .= aoe_fwdfun(fullx) .+ rand(MvNormal(zeros(n), setup.Γ_ϵ))
    z_true[i] = setup.O[3:end]' * setup.x_true
    μ_zgy_vec[i], Σ_zgy_vec[i] = apply_cond_gaussian(vcat(pry, prz), setup.y; whitened=false) 

    @time for j in 1:numdim #r in dimlist
        r = dimlist[j]
        # godr
        Vr = V_godr[:,1:r]
        μ_zgy_godr[j,i], Σ_zgy_godr[j,i] = apply_cond_gaussian(vcat(Vr' * pry, prz), Vr' * setup.y; whitened=false) 

        # nogoal
        Vr = V_nogoal[:,1:r]
        μ_zgy_nogoal[j,i], Σ_zgy_nogoal[j,i] = apply_cond_gaussian(vcat(Vr' * pry, prz), Vr' * setup.y; whitened=false) 

        # pca
        Vr = V_pca[:,1:r]
        μ_zgy_pca[j,i], Σ_zgy_pca[j,i] = apply_cond_gaussian(vcat(Vr' * pry, prz), Vr' * setup.y; whitened=false) 

    end

end

# for j in 1:numdim

    
#     for i in 1:N_mc
#         err_godr[j] = err_godr[j] + 1/N_mc * (μ_zgy_godr[j,i] - μ_zgy_vec[i])^2 / Σ_zgy_godr[j,i]
#         err_nogoal[j] = err_nogoal[j] + 1/N_mc * (μ_zgy_nogoal[j,i] - μ_zgy_vec[i])^2 / Σ_zgy_nogoal[j,i]
#         err_pca[j] = err_pca[j] + 1/N_mc * (μ_zgy_pca[j,i] - μ_zgy_vec[i])^2 / Σ_zgy_pca[j,i]

#     end
# end

err_godr, err_nogoal, err_pca = zeros(numdim), zeros(numdim), zeros(numdim)
for j in 1:numdim
    for i in 1:N_mc
        err_godr[j] = err_godr[j] + 1/N_mc * (μ_zgy_godr[j,i] - z_true[i])^2 # / Σ_zgy_godr[j,i]
        err_nogoal[j] = err_nogoal[j] + 1/N_mc * (μ_zgy_nogoal[j,i] -z_true[i])^2 #/ Σ_zgy_nogoal[j,i]
        err_pca[j] = err_pca[j] + 1/N_mc * (μ_zgy_pca[j,i] - z_true[i])^2 #/ Σ_zgy_pca[j,i]
    end
end


err0 = 0
for i in 1:N_mc
    err0 = err0 + 1/N_mc * (setup.μ_z[1] - z_true[i])^2 #/ setup.Γ_z[1,1]
end

plot(dpi=300, yaxis=:log, xlims=[-2,50], legend=:topright)#, ylims=[5e-2,1e1])
# plot!(ylabel="Error", xlabel="Rank")
plot!(vcat([0],dimlist), vcat([err0], err_pca), linewidth=2, color=:blue3, linestyle=:dot, label="PCA")
plot!(vcat([0],dimlist),  vcat([err0], err_nogoal), linewidth=2, color=:green, linestyle=:dash, label="Non-Goal-Oriented")
plot!(vcat([0],dimlist), vcat([err0], err_godr), linewidth=2, color=:red, label="Goal-Oriented")

savefig("plots/JSM/errorvsrank_50.png")

# save("mu_compare.jld2", "godr", μ_godr, "nogoal", μ_nogoal, "pca", μ_pca, "true", μ_zgy_true)
# save("cov_compare.jld2", "godr", Σ_godr, "nogoal", Σ_nogoal, "pca", Σ_pca, "true", Σ_zgy_true)
