using NPZ

sampPr = npzread("data_canopy/priorSamples_8_unscaled.npy")
sampCwt = npzread("data_canopy/cwt_priorSamples_8_unscaled.npy")


nλ, Nsamp = size(sampPr)

Z = sampCwt
X = vcat(ones((1,Nsamp)), sampPr)'

# cut out negative values of truth that have been truncated to 0
Z = Z[sampCwt.>1e-8]
X = X[sampCwt.>1e-8, :]

# X_small = X[1:80000,:];

# β = inv(X_small'*X_small) * X_small' * Z[1:80000]

β = inv(X'*X) * X' * Z
O = β[2:end]
O_offset = β[1]

## write test function to compare Ox and z
npzwrite("data_canopy/goal_op_8_unscaled.npy", O)
npzwrite("data_canopy/goal_op_const_8_unscaled.npy",O_offset)


plot(Z[1:1000],(β'*X')[1:1000], seriestype=:scatter, xlabel="Truth", ylabel="Predicted", legend=false)#, ylims=[0,0.3], xlims=[0,0.4])
plot!([0;0.5], [0;0.5], color=:red)
