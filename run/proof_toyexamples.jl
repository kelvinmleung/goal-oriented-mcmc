using LinearAlgebra
using Statistics
using Plots
using LaTeXStrings
using Random

m = 200 # data dimension
d = 200 # parameter dimension
r = 150 # reduced parameter dimension
# s = 5

Random.seed!(1234)

# G = rand(m,d)
G = diagm(randn(m))
U_r = rand(d,r)

A = randn(m, m)
Γ_ϵ = diagm(abs.(randn(m))) / 100 # diagm(ones(m)) #* 100000
# Γ_ϵ = (A*A' + 1e-10 *I) 
# Γ_ϵ = G * Γ_x * G'

Γ_x = diagm(ones(d))
Γ_y = G * Γ_x * G' + Γ_ϵ





# display(eigvals(G * G'))
# display(eigvals(inv(sqrt(Γ_ϵ)) * G * G' * inv(sqrt(Γ_ϵ))))
# display(eigvals(inv(sqrt(Γ_y) * G * G' * inv(sqrt(Γ_y)))))

# println("\n")

H = G * U_r * U_r' * G' 
H_ϵ = inv(sqrt(Γ_ϵ)) * G * U_r * U_r' * G' * inv(sqrt(Γ_ϵ)) 
H_ϵ = tril(H_ϵ) + tril(H_ϵ,-1)'
H_y = inv(sqrt(Γ_y)) * G * U_r * U_r' * G' * inv(sqrt(Γ_y)) 
H_y = tril(H_y) + tril(H_y,-1)'

# evals, evecs = eigen(H)
display(eigvals(H))
display(eigvals(H_ϵ))
display(eigvals(H_y))



plt = plot(xlabel=L"$s$", ylabel=L"$\lambda_{max}(\Gamma, \Gamma_\epsilon) \, \sum_{i=1}^s \lambda_i(H,\Gamma)$", yaxis=:log, dpi=300, size=(600,400))
plot!(1:r, eigvals(H)[end:-1:1][1:r] * eigvals(inv(Γ_ϵ))[end], linewidth=2, label="No Whitening")
# plot!(1:100, eigvals(H_ϵ)[end:-1:1][1:100] * eigvals(Γ_ϵ)[end], label=L"$\Gamma_\epsilon$")
plot!(1:r, eigvals(H_ϵ)[end:-1:1][1:r] , linewidth=2, label=L"Whiten with $\Gamma_\epsilon$")
plot!(1:r, eigvals(H_y)[end:-1:1][1:r] * eigvals(inv(Γ_ϵ)* Γ_y)[end],linewidth=2,  label=L"Whiten with $\Gamma_y$")
display(plt)

# V_s = eigvecs(H)[:,end:-1:end-s+1]
# V_s_ϵ = eigvecs(H_ϵ)[:,end:-1:end-s+1]
# V_s_y = eigvecs(H_y)[:,end:-1:end-s+1]

# tr(V_s' * H * V_s)
# tr(V_s_ϵ' * H_ϵ * V_s_ϵ)
# tr(V_s_y' * H_y * V_s_y)



# maxeig_ϵ = eigvals(Γ_ϵ)[end]
maxeig_y = eigvals(inv(Γ_ϵ) * Γ_y)[end]

display(sum(eigvals(H)[end:-1:end-s+1]))
display(sum(eigvals(H_ϵ)[end:-1:end-s+1]) )
display(sum(eigvals(H_y)[end:-1:end-s+1]) * maxeig_y)

# display(sum(eigvals(H_y)[end:-1:end-s+1]) * maxeig_y > maximum([sum(eigvals(H_ϵ)[end:-1:end-s+1]) * maxeig_ϵ, sum(eigvals(H)[end:-1:end-s+1])]))



# error = sum(eigvals(G * G')) - sum(eigvals(H)[end-s:end])
# error_ϵ = sum(eigvals(inv(sqrt(Γ_ϵ) * G * G' * inv(sqrt(Γ_ϵ))))) - sum(eigvals(H_ϵ)[end-s:end])
# error_y = sum(eigvals(inv(sqrt(Γ_y) * G * G' * inv(sqrt(Γ_y))))) - sum(eigvals(H_y)[end-s:end])
# display(error)
# display(error_ϵ)
# display(error_y)