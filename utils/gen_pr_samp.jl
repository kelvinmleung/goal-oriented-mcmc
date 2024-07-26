using Random, LinearAlgebra, Distributions
using AOE
λ_ranges = [400.0 1300.0; 1450.0 1780.0; 2051.0 2451.0]
priormodel, wls = get_priormodel(:standard; λ_ranges) # PriorModel instance
rtmodel = AOE.get_radiative_transfer(:modtran; λ_ranges);

m = 10000


prsamps = zeros(8, m, 326)
for i in 1:8
    prsamps[i,:,:] = rand(MvNormal(priormodel[i].mean, priormodel[i].cov), m)'
end

npzwrite("data/data_canopy/prsamp_all_unscaled.npy", prsamps)
