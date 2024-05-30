import numpy as np
import sys
import time
sys.path.append("/Users/kmleung/Documents/Github/isofit")


from isofit.inversion import inverse_simple as inv

# from isofit.utils import ewt_from_reflectance as efr

# rfl_file = "/Users/kmleung/Documents/Github/isofit/data/reflectance/surface_model_ucsb"
# output_cwc_file = "/Users/kmleung/Documents/Github/goal-oriented-mcmc/data_canopy/ucsb_canopy"
# wl = np.array(range(350, 2501))
# startstop = (5000,5050)
# efr.run_lines(rfl_file, output_cwc_file, wl, startstop, log)

# rfl = np.load("/Users/kmleung/Documents/Github/goal-oriented-mcmc/data_refl/x_177.npy")
# wl = np.load("/Users/kmleung/Documents/Github/goal-oriented-mcmc/data_refl/wls.npy")
# print(np.shape(wl))
# cwt = inv.invert_liquid_water(rfl, wl)[0]
# print(cwt)

sampPr = np.load("data_canopy/priorSamples_8_unscaled.npy")
wl = np.load("/Users/kmleung/Documents/Github/goal-oriented-mcmc/data_refl/wls.npy")
Nsamp = np.size(sampPr,1)
cwt = np.zeros(Nsamp)

for i in range(Nsamp):
    if (i+1) % 100 == 0:
        print("Iteration:" , i+1)
    cwt[i] = inv.invert_liquid_water(sampPr[:,i], wl)[0]

print(cwt[:10])
np.save("data_canopy/cwt_priorSamples_8_unscaled.npy", cwt)


