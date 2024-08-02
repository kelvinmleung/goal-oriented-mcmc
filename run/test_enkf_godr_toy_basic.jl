using Random, Distributions, StatsPlots, LinearAlgebra
Random.seed!(123)

n = 2
m = 100000
μ_x = [0., 0.]
Γ_x = [1. 0.9; 0.9 1.]
Γ_ϵ = [0.1 0.; 0 0.1]
O = [0., 1.]'
x_true = [5., 0.]
z_true = O * x_true

# G = [2. 1; 0 2.]
G = [3. 10.; 1. 5.]
y = G * x_true + rand(MvNormal(zeros(n), Γ_ϵ))

μ_z = O * μ_x 
Γ_z = O * Γ_x * O'
invΓ_x = inv(Γ_x)
invΓ_z = inv(Γ_z)
invΓ_ϵ = inv(Γ_ϵ)

Γ_xgy = inv(G' * invΓ_ϵ * G + invΓ_x)
μ_xgy = Γ_xgy * (invΓ_x * μ_x + G' * invΓ_ϵ * y)

prsamp_x = rand(MvNormal(μ_x, Γ_x), m)
prsamp_z = O * prsamp_x
prsamp_y = G * prsamp_x + rand(MvNormal(zeros(n), Γ_ϵ), m)

# scatter(prsamp_x[1,1:10:end], prsamp_x[2,1:10:end], alpha=0.2, label="", title="Prior Samples X")
# scatter!([x_true[1]], [x_true[2]], markersize=20, label="Truth")

# scatter(prsamp_y[1,1:10:end], prsamp_y[2,1:10:end], alpha=0.2, label="", title="Prior Predictive Y")
# scatter!([y[1]], [y[2]], markersize=20, label="Observation")

# density(prsamp_z, title="Prior Density Z")

# O_comp = ones(2)' - O
# H = G * O_comp' * O_comp * G'
# H = G * (I - O' * O) * G'

sqrtcovy = sqrt(cov(prsamp_y,dims=2))


H = inv(sqrtcovy) * G * Γ_x * O' * invΓ_z * O * Γ_x * G' * inv(sqrtcovy)
# H = G * Γ_x * O' * invΓ_z * O * Γ_x * G'

eigs, V = eigvals(H)[end:-1:1], eigvecs(H)[:,end:-1:1]
display(V)

Vr = inv(sqrtcovy) * V[:,1]
# Vr = V[:,1]

# Vr = [0.79, 0.61]

# m_test = 1000
# cov_zgvry = zeros(m_test)
# mu_zgvry = zeros(m_test)
# sgn_num = 1

# for i in 1:m_test #0:0.01:1.

#     val = LinRange(0, 1, m_test)[i]
#     Vr = [val; sgn_num * sqrt(1-val^2)]

#     cov_vry_z = Vr' * G * Γ_x * O' 
#     cov_vry = Vr' * (G * Γ_x * G' + Γ_ϵ) * Vr
#     μ_z_g_vry = μ_z .- cov_vry_z' * inv(cov_vry) * (Vr' * G * μ_x - Vr' * y)
#     Γ_z_g_vry = Γ_z - cov_vry_z' * inv(cov_vry) * cov_vry_z
#     cov_zgvry[i] = Γ_z_g_vry
#     mu_zgvry[i] = μ_z_g_vry

# end
# alpha = LinRange(0, 1, m_test)[argmin(abs.(cov_zgvry .- O * Γ_xgy * O'))]
# println("Optimal Eigenvector: $([alpha; -sqrt(1-alpha^2)])")

# sgn = sgn_num > 0 ? "" : "-"

# plot(LinRange(0, 1, m_test), cov_zgvry, linewidth=2, label="Test Vr", title="Visualizing Variance Reduction using " * L" V_r = [\alpha, " * sgn * L"\sqrt{1-\alpha^2}]^\top", xlabel=L"\alpha", ylabel=L"\Gamma_{Z|V_r Y}", dpi=300)
# plot!([O * Γ_xgy * O'], seriestype=:hline, linewidth=2, label="True Cov")
# plot!([O * Γ_x * O'], seriestype=:hline, linewidth=2, label="Prior Cov")

# plot(LinRange(0, 1, m_test), mu_zgvry, linewidth=2, label="Test Vr", title="Visualizing Mean Shift using " * L" V_r = [\alpha, " * sgn * L"\sqrt{1-\alpha^2}]^\top", xlabel=L"\alpha", ylabel=L"\mu_{Z|V_r Y}", dpi=300)
# plot!([O * μ_xgy], seriestype=:hline, linewidth=2, label="True Mean")
# plot!([O * μ_x], seriestype=:hline, linewidth=2, label="Prior Mean")

# # # savefig("varredpos.png")

# Vr = [0.2502502502502503, -0.9681811877173028]

## analytic solution
cov_vry_z = Vr' * G * Γ_x * O' 
cov_vry = Vr' * (G * Γ_x * G' + Γ_ϵ) * Vr
# μ_z_g_vry = μ_z .- cov_vry_z' * inv(cov_vry) * (Vr' * G * μ_x - Vr' * y)
μ_z_g_vry = μ_z .+ O * Γ_x * G' * Vr * Vr' * (y - G * μ_x)

Γ_z_g_vry = Γ_z - cov_vry_z' * inv(cov_vry) * cov_vry_z


X = vcat(Vr' * prsamp_y, prsamp_z)
# X = vcat(prsamp_x, prsamp_z)


display(scatter(X[2,1:10:end], X[1,1:10:end], alpha=0.5))
# scatter(prsamp_z[1:10:end], ( prsamp_y)[2,1:10:end], alpha=0.5)

using PrettyTables
tabledata = hcat(["True Posterior", "Goal-Oriented", "Difference"], [O * μ_xgy, μ_z_g_vry, O * μ_xgy - μ_z_g_vry], [O * Γ_xgy * O', Γ_z_g_vry, O * Γ_xgy * O'- Γ_z_g_vry])
headers = ["Quantity", "Pos Mean", "Pos Var"]
pretty_table(tabledata; header = headers)


jointmean = mean(X, dims=2)
jointcov = cov(X, dims=2)
μ_y = @view jointmean[1:end-1]
μ_z = @view jointmean[end]
Σ_zz = @view jointcov[end, end]
Σ_yz = @view jointcov[1:end-1, end]
invΣ_yy = inv(jointcov[1:end-1, 1:end-1])
μ_zgy = μ_z .- Σ_yz' * (invΣ_yy * Vr' * (G * μ_x - y))
Σ_zgy = Σ_zz .- Σ_yz' * invΣ_yy * Σ_yz


zgy_true = O * rand(MvNormal(μ_xgy, (Γ_xgy +  Γ_xgy')/2), m)
zgy_analytic = rand(Normal(μ_z_g_vry, sqrt(Γ_z_g_vry)), m)
zgy_samp = rand(Normal(μ_zgy[1,1], sqrt(Σ_zgy)), m)

density(zgy_true[1:10:end], linewidth=2, label="True Posterior")
# density!(zgy_analytic[1:10:end], linewidth=2, label="Goal-Oriented - Analytic")
density!(zgy_samp[1:10:end], linewidth=2, label="Goal-Oriented - Samples")