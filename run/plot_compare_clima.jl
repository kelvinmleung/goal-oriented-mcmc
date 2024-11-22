include("../src/forward.jl")
include("../src/goalorientedtransport.jl")

Random.seed!(123)

λ_ranges = [400.0 1300.0; 1450.0 1780.0; 2051.0 2451.0]
priormodel, wls = get_priormodel(:standard; λ_ranges) # PriorModel instance
rtmodel = AOE.get_radiative_transfer(:modtran; λ_ranges);
n, p = 326, 4
m = 1000000

setup = initialize_GODRdata(n, p)
GODRdata_pr8!(setup; n=n, p=p);
prsamp = gen_pr_samp(setup, m; n=n, p=p);

# save("data/data_CliMA/truth_obs_pairs.jld", "x", setup.x_true, "y", setup.y, "z", setup.z_true)

covy = cov(prsamp.y, dims=2) 
invsqrtcovy = inv(sqrt(covy))
meany = mean(prsamp.y, dims=2)


H_mat = diagnostic_matrix(10000, invsqrtcovy)
H_nogoal = diagnostic_matrix_no_goal(10000, invsqrtcovy)
H_pca = covy 

Λ_godr, V_godr = diagnostic_eigendecomp(H_mat; showplot=false, setup=setup)
Λ_nogoal, V_nogoal = diagnostic_eigendecomp(H_nogoal; showplot=false, setup=setup)
Λ_pca, V_pca = diagnostic_eigendecomp(H_pca; showplot=false, setup=setup)

V_godr = invsqrtcovy * V_godr
V_nogoal = invsqrtcovy * V_nogoal

### EIGENVALUE AND EIGENVECTOR PLOTS
# plot(Λ_godr ./ Λ_godr[1], yaxis=:log, label=false, linewidth=3, color=:blue4, dpi=300, title=L"Eigenvalue Spectrum of $H_{GO}$", ylabel="Eigenvalue")
# savefig("plots/11222024(pdf)/eigval.pdf")
# plot!(Λ_nogoal ./ Λ_nogoal[1], yaxis=:log, label=false, linewidth=3, color=:blue3)
# plot!(Λ_pca ./ Λ_pca[1], yaxis=:log, label=false, linewidth=3, color=:blue3)

# O_tilde = (inv(sqrt(setup.Γ_z)) *  setup.O * sqrt(setup.Γ_x))[indQOI,3:end]
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
# savefig("plots/11112024/eigvec.png")

# display(p)

# ### PLOT OF MCMC VS TRANSPORT
# z_possamp_naive = npzread("data/data_canopy/aug2/10pix_ind(1,1)/z_naive.npy")
# z_possamp_transport_godr = npzread("data/data_canopy/aug2/10pix_ind(1,1)/z_transport_godr.npy")
# z_possamp_transport_nogoal = npzread("data/data_canopy/aug2/10pix_ind(1,1)/z_transport_nogoal.npy")

# plot([setup.z_true], seriestype="vline", color=:black, linestyle=:dot,linewidth=3, label="Truth", dpi=300, xlims=[0,0.3], ylims=[0,100], legend=:topleft)
# # density!(z_possamp_naive[2000000:100:end], linewidth=4, linestyle=:dash, label="MCMC, rank 326 (full)", color=:blue4)
# density!(z_possamp_transport_nogoal[1:10:end], color=:green, linewidth=2, label="SBI without goal, rank 2")
# density!(z_possamp_transport_godr[1:10:end], color=:red, linewidth=2, label="SBI goal-oriented, rank 2")
# savefig("plots/JSM/poszgy.png")


### TRANSPORT DENSITY PLOTS
m = 10000
r = energy_cutoff(Λ_godr, 0.9999)
V_r = V_godr[:,1:r]


X = vcat(V_r' * (prsamp.y .- meany), sqrt(setup.invΓ_z) * (prsamp.z .- setup.μ_z))[:,1:m]
yobs_whiten = repeat(V_r' * (setup.y - meany), 1, m)




z_possamp_whiten, S, F = apply_cond_transport(X, yobs_whiten, r; order=10)
z_possamp_transport =  sqrt(setup.Γ_z) * z_possamp_whiten .+ setup.μ_z .+ setup.O_offset
npzwrite("data/data_clima/z_transport.npy", z_possamp_transport)

X_nogoal = vcat(V_nogoal[:,1:r]' * (prsamp.y .- meany), sqrt(setup.invΓ_z) * (prsamp.z .- setup.μ_z))[:,1:m]
z_possamp_whiten_nogoal, S, F = apply_cond_transport(X_nogoal, repeat(V_nogoal[:,1:r]' * (setup.y - meany), 1, m), r; order=10)
z_possamp_nogoal =  sqrt(setup.Γ_z) * z_possamp_whiten_nogoal .+ setup.μ_z .+ setup.O_offset

X_pca = vcat(V_pca[:,1:r]' * (prsamp.y .- meany), sqrt(setup.invΓ_z) * (prsamp.z .- setup.μ_z))[:,1:m]
z_possamp_whiten_pca, S, F = apply_cond_transport(X_pca, repeat(V_pca[:,1:r]' * (setup.y - meany), 1, m), r; order=10)
z_possamp_pca =  sqrt(setup.Γ_z) * z_possamp_whiten_pca .+ setup.μ_z .+ setup.O_offset


# z_possamp_transport = npzread("data/data_clima/z_transport.npy")
z_possamp_naive = npzread("data/data_clima/z_naive_mcmc.npy")[:,200000:10:1000000]

using LaTeXStrings
keys_goal = ["BROWN","CHL","LMA","LWC"]
xlabels = ["Senescent Material Fraction", L"Cholorophyll Content [$\mu$g / cm$^2$]", L"Dry Matter Content [g/cm$^2$]", "Equivalent Water Thickness [units?]"]
# selectQOI = [2,3,7,10]
for i in 1:p

    plt=plot(title=keys_goal[i], legend=:topleft, dpi=300, size=(500,300))#xlim=[0.15,0.3])
    if i == 2
        plot!(legend=:topright)
    end
    plot!(ylabel="Marginal Density", xlabel=xlabels[i])
    xmin = setup.μ_z[i] + setup.O_offset[i] - 2*sqrt(setup.Γ_z[i,i])
    xmax = setup.μ_z[i] + setup.O_offset[i] + 2*sqrt(setup.Γ_z[i,i])
    plotrange = xmin:(xmax-xmin)/100:xmax
    plot!(xlims=[xmin, xmax])
    kde_pr = pdf.(Normal(setup.μ_z[i] .+ setup.O_offset[i], sqrt(setup.Γ_z[i,i])), plotrange) 
    plot!(plotrange, kde_pr, color=:black, linewidth=1, label="Prior")
    plot!([setup.z_true[i]], seriestype="vline", color=:black, linestyle=:dot,linewidth=3, label="Truth")

    density!(z_possamp_naive[i,:], color=:black, linewidth=3, label="Naive MCMC")

    density!(z_possamp_nogoal[i,:], color=:blue, linewidth=3, linestyle=:dash, label="NGO Transport, r=$r")
    density!(z_possamp_pca[i,:], color=:green, linewidth=3, linestyle=:dash,  label="PCA Transport, r=$r")
    density!(z_possamp_transport[i,:], color=:red, linewidth=3, label="GO Transport, r=$r")
    display(plt)
    savefig("plots/11222024(pdf)/transportdensity_qoi_" * keys_goal[i] * "_r$r.pdf")
end





#### PLOT OF ERROR COMPARISON
N_mc = 500
dimlist = vcat(1:9,10:2:30,40:10:150,160:20:320, 325)
# N_mc = 1000
# dimlist = vcat(1:9,10:5:45)
numdim = length(dimlist)
μ_zgy_vec, Σ_zgy_vec= zeros(p, N_mc), zeros(p, N_mc)
μ_zgy, μ_zgy_godr, μ_zgy_nogoal, μ_zgy_pca = zeros(p, N_mc), zeros(p, numdim, N_mc), zeros(p, numdim, N_mc), zeros(p, numdim, N_mc)
Σ_zgy, Σ_zgy_godr, Σ_zgy_nogoal, Σ_zgy_pca = zeros(p, p, N_mc), zeros(p, p, numdim, N_mc), zeros(p, p, numdim, N_mc), zeros(p, p, numdim, N_mc)
z_true = zeros(p, N_mc)
Random.seed!(123)
pry, prz = prsamp.y[:,1:10:end], prsamp.z[:,1:10:end]
for i in 1:N_mc
    
    fullx = rand(MvNormal(setup.μ_x, setup.Γ_x))
    setup.x_true .= fullx[3:end]
    setup.z_true .= setup.O[:,3:end] * setup.x_true .+ setup.O_offset
    setup.y .= aoe_fwdfun(fullx) .+ rand(MvNormal(zeros(n), setup.Γ_ϵ))
    z_true[:,i] = setup.O[:,3:end] * setup.x_true
    # μ_zgy_vec[i], Σ_zgy_vec[i] = apply_cond_gaussian(vcat(pry, prz), setup.y, n) 
    μ_zgy[:,i], Σ_zgy[:, :,i] = apply_cond_gaussian(vcat(invsqrtcovy * (pry .- meany), prz), invsqrtcovy * (setup.y - meany)[:,1], n)
    
    for j in 1:numdim #r in dimlist
        r = dimlist[j]
        # godr
        Vr = V_godr[:,1:r]
        μ_zgy_godr[:, j,i], Σ_zgy_godr[:,:,j,i] = apply_cond_gaussian(vcat(Vr' * (pry .- meany), prz), (Vr' * (setup.y .- meany))[:,1] , r) 

        # nogoal
        Vr = V_nogoal[:,1:r]
        μ_zgy_nogoal[:, j,i], Σ_zgy_nogoal[:, :, j,i] = apply_cond_gaussian(vcat(Vr' * (pry .- meany), prz), Vr' * (setup.y .- meany)[:,1], r) 

        # pca
        Vr = V_pca[:,1:r]
        μ_zgy_pca[:, j,i], Σ_zgy_pca[:, :, j,i] = apply_cond_gaussian(vcat(Vr' * (pry .- meany), prz), Vr' * (setup.y .- meany)[:,1], r) 

         
    end

end


err_godr, err_nogoal, err_pca = zeros(numdim), zeros(numdim), zeros(numdim)
br_godr, br_nogoal, br_pca = zeros(numdim), zeros(numdim), zeros(numdim)
for j in 1:numdim
    for i in 1:N_mc
        br_godr[j] = br_godr[j] + 1/N_mc * (μ_zgy_godr[:,j,i] - z_true[:,i])' * inv(Σ_zgy_godr[:,:,j,i]) * (μ_zgy_godr[:,j,i] - z_true[:,i])
        br_nogoal[j] = br_nogoal[j] + 1/N_mc *  (μ_zgy_nogoal[:,j,i] - z_true[:,i])' * inv(Σ_zgy_nogoal[:,:,j,i]) * (μ_zgy_nogoal[:,j,i] - z_true[:,i])
        br_pca[j] = br_pca[j] + 1/N_mc *  (μ_zgy_pca[:,j,i] - z_true[:,i])' * inv(Σ_zgy_pca[:,:,j,i]) * (μ_zgy_pca[:,j,i] - z_true[:,i])
            
        for indQOI in 1:p
            err_godr[j] = err_godr[j] + 1/N_mc * abs.(μ_zgy_godr[indQOI,j,i] - z_true[indQOI,i]) / abs(z_true[indQOI,i])  # #/ Σ_zgy_godr[indQOI,indQOI,j,i]
            err_nogoal[j] = err_nogoal[j] + 1/N_mc *  abs.(μ_zgy_nogoal[indQOI,j,i] - z_true[indQOI,i]) / abs(z_true[indQOI,i])#-  #/ sqrt(Σ_zgy_nogoal[indQOI,indQOI,j,i])
            err_pca[j] = err_pca[j] + 1/N_mc *  abs.(μ_zgy_pca[indQOI,j,i] - z_true[indQOI,i]) / abs(z_true[indQOI,i])#-  #/ sqrt(Σ_zgy_pca[indQOI,indQOI,j,i])
        end
    end
end
err0, br0 = 0, 0
for i in 1:N_mc
    br0 = br0 + 1/N_mc * (setup.μ_z[:,1] - μ_zgy[:,i])' * setup.invΓ_z * (setup.μ_z[:,1] - μ_zgy[:,i])
    for indQOI in 1:p
        # err0 = err0 + 1/N_mc * abs.(setup.μ_z[indQOI,1] - z_true[indQOI,i]) #/ sqrt(setup.Γ_z[indQOI,indQOI])
        err0 = err0 + 1/N_mc * abs.(setup.μ_z[indQOI,1] - z_true[indQOI,i]) / abs(z_true[indQOI,i])#/ sqrt(setup.Γ_z[indQOI,indQOI])
    end
end

plot(dpi=300,  xlims=[0,325],  ylims=[0.055,0.10], legend=:topright)#, ylims=[5e-2,1e1]),
plot!(ylabel="Relative Difference from Truth", xlabel="Rank")#, title="Norm Error")
plot!(vcat([0],dimlist), vcat([err0], err_pca), linewidth=3, color=:blue3, linestyle=:dot, label="PCA")
plot!(vcat([0],dimlist),  vcat([err0], err_nogoal), linewidth=2, color=:green, linestyle=:dash, label="Non-Goal-Oriented")
plot!(vcat([0],dimlist), vcat([err0], err_godr), linewidth=3, color=:red, label="Goal-Oriented")
savefig("plots/11122024/reldiff_allrank_zoomin.png")

plot(dpi=300, xlims=[0,30],  legend=:topright)#, ylims=[5e-2,1e1]),
plot!(ylabel="Relative Difference from Truth", xlabel="Rank")#, title="Norm Error")
plot!(vcat([0],dimlist), vcat([err0], err_pca), linewidth=3, color=:blue3, linestyle=:dot, label="PCA")
plot!(vcat([0],dimlist),  vcat([err0], err_nogoal), linewidth=2, color=:green, linestyle=:dash, label="Non-Goal-Oriented")
plot!(vcat([0],dimlist), vcat([err0], err_godr), linewidth=3, color=:red, label="Goal-Oriented")
savefig("plots/11122024/reldiff_r30.png")


plot(dpi=300,  yaxis=:log,xlims=[0,300], legend=:topright)#, ylims=[5e-2,1e1]),
plot!(ylabel="Error Weighted by Precision", xlabel="Rank", title="Bayes Risk")
plot!(vcat([0],dimlist), vcat([br0], br_pca), linewidth=2, color=:blue3, linestyle=:dot, label="PCA")
plot!(vcat([0],dimlist),  vcat([br0], br_nogoal), linewidth=2, color=:green, linestyle=:dash, label="Non-Goal-Oriented")
plot!(vcat([0],dimlist), vcat([br0], br_godr), linewidth=2, color=:red, label="Goal-Oriented")



# savefig("plots/10172024/errorvsrank_30_nocov.png")

# save("mu_compare.jld2", "godr", μ_godr, "nogoal", μ_nogoal, "pca", μ_pca, "true", μ_zgy_true)
# save("cov_compare.jld2", "godr", Σ_godr, "nogoal", Σ_nogoal, "pca", Σ_pca, "true", Σ_zgy_true)
