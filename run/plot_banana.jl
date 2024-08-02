using Plots
using Random, Distributions

# Function to create a banana-shaped distribution
function banana_distribution(n, b=0.2, skew_factor=0.3)
    x = randn(n)
    y = randn(n) * 0.2 + b * (x .^ 2 .- 1)
    return x, y
end

# Generate the banana-shaped distribution
n = 10000
x, y = banana_distribution(n)

y_fixed = 0.25
indices = findall(abs.(y .- y_fixed) .< 0.05)
x_conditional = x[indices]
kde_x_conditional = kde(x_conditional)

# Create a scatter plot
scatter(x, y, alpha=0.05, color=:skyblue, xlabel=L"\Theta", ylabel=L"\mathbf{y}", title="", label="", xlims=[-4,4],ylims=[-1, 1.6], dpi=300)
plot!(kde_x_conditional.x, kde_x_conditional.density * 2 .+0.5, legend=false, color=:red, linewidth=2, label="Marginal density of true distribution")
plot!([0.5], seriestype=:hline, linestyle=:dash, color=:black, linewidth=2, label=false)
# plot!([-1], seriestype=:hline, color=:black, linewidth=2, label=false)
savefig("conditionalslice.png")

#####

function gaussian_mixture_2d_distribution(n, μ1, Σ1, μ2, Σ2, w1)
    # Define the two Gaussian components
    comp1 = MvNormal(μ1, Σ1)
    comp2 = MvNormal(μ2, Σ2)
    
    # Define the mixture model
    mixture = MixtureModel([comp1, comp2], [w1, 1 - w1])
    
    # Generate data points
    data = rand(mixture, n)
    display(data)
    
    return data
end

# Parameters for the Gaussian components
μ1 = [-2.0, 2.0]
Σ1 = [1.0 0.5; 0.5 1.0]
μ2 = [3.0, 3.0]
Σ2 = [1.0 -0.5; -0.5 1.0]
w1 = 0.8  # Weight for the first component

# Generate data from the Gaussian mixture model
n = 10000
data = gaussian_mixture_2d_distribution(n, μ1, Σ1, μ2, Σ2, w1)

# Extract x and y coordinates
x = data[1,:]
y = data[2,:]

# Plot the Gaussian mixture distribution
Random.seed!(123)
scatter(x, y, alpha=0.05, color=:lightblue, label=false, xlims=[-7,7], ylims=[-2,7], dpi=300)

xgrid = range(minimum(x), stop=maximum(x), length=100)
ygrid = range(minimum(y), stop=maximum(y), length=100)
z = [pdf(MvNormal(μ1, Σ1), [xi, yi]) for xi in xgrid, yi in ygrid]'

# contour!(xgrid, ygrid, z1, color=:red, levels=3)

savefig("OEfailing1.png")

scatter!([μ1[1]], [μ1[2]], color=:red, alpha=0.8, label="MAP Estimate")
contour!(xgrid, ygrid, z, levels=3, linewidth=1.5, color=:red, colorbar=false, label=false, alpha=1)
plot!([], [], label="Laplace Approx", linecolor=:red, linewidth=1.5)

savefig("OEfailing2.png")


marginal_dist = Normal(μ1[1], sqrt(Σ1[1,1]))
x_values = range(-5, stop=2, length=100)
marginal_density = pdf.(marginal_dist, x_values)
plot!(x_values, marginal_density * 5 .- 2, label=false, legend=true, color=:red, linewidth=2, linestyle=:dash)

kde_x_marginal = kde(x)
plot!(kde_x_marginal.x, kde_x_marginal.density * 5 .- 2, color=:skyblue, linewidth=3, label=false)
plot!([-2], seriestype=:hline, linewidth=2, color=:black, label=false)

savefig("OEfailing3.png")


#########
# Function to create a banana-shaped distribution
function banana_distribution(n, b=0.2, skew_factor=0.3)
    x = randn(n)
    # y = randn(n) * 0.2 + b * (x .^ 2 .- 1)
    y = randn(n) * 0.2 + b * (x .^ 2 .- 1) .* exp.(-skew_factor * x)
    return x, y
end

# Generate the banana-shaped distribution
n = 10000
x, y = banana_distribution(n)

# Create a scatter plot
# scatter(x, y, alpha=0.1, xlabel=L"\Theta", ylabel=L"\mathbf{y}", title="", label="", xlims=[-4,4],ylims=[-1, 1.6])
scatter(x, y, alpha=0.02, xlabel=L"x_1", ylabel=L"x_2", title="", label="", xlims=[-4,4],ylims=[-1, 2.0], color=:blue2, dpi=300)



# Laplace approximation at [-2, 0.5]
mu = [-1, -0.1]
hessian =0.3 * [1.0 -0.75; -0.75 1.0]  # Assume identity covariance for simplicity

# Create a grid for contour plotting
xgrid = range(-4, stop=2, length=100)
ygrid = range(-2, stop=3, length=100)
xygrid = [collect(xgrid) collect(ygrid)]
pdf_laplace(x, y) = pdf(MvNormal(mu, hessian), [x, y])

# Evaluate the pdf on the grid
z = [pdf_laplace(x, y) for x in xgrid, y in ygrid]

# Plot the contours
contour!(xgrid, ygrid, z, levels=3, linewidth=1.5, color=:red, colorbar=false, label=false, alpha=0.4)
# plot!([], [], label="Laplace Approximation", linecolor=:red, linewidth=1.5)
scatter!([-1.72], [0.5], color=:red, alpha=0.5, label=false)


# y_fixed = 0.25
# indices = findall(abs.(y .- y_fixed) .< 0.05)
# x_conditional = x[indices]

# Density estimation for conditional distribution
kde_x_marginal = kde(x)

# Plot density of conditional distribution
plot!(kde_x_marginal.x, kde_x_marginal.density * 2 .- 1, legend=false, color=:skyblue, linewidth=2, label="Marginal density of true distribution")




# Conditional distribution x | y = 0.5
mu = [-1.72, 0.5]
# mu_x_given_y = mu[1] + hessian[1, 2] / hessian[2, 2] * (y_fixed - mu[2])
# var_x_given_y = hessian[1, 1] - hessian[1, 2]^2 / hessian[2, 2]
# conditional_dist = Normal(mu_x_given_y, sqrt(var_x_given_y))

marginal_dist = Normal(mu[1], sqrt(hessian[1,1]))

# Generate the x values for the conditional distribution
x_values = range(-4, stop=2, length=100)
marginal_density = pdf.(marginal_dist, x_values)

# Plot conditional density of the Gaussian contour
plot!(x_values, marginal_density .- 1, label="Marginal density of Laplace approximation", legend=true, color=:red, linewidth=2, linestyle=:dash)

plot!([-1], seriestype=:hline, linewidth=2, color=:black, label=false)

savefig("bananamarginal.png")
# Show the plot
# display(plot())