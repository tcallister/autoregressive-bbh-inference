import numpyro
import sys
nChains = 1
numpyro.set_host_device_count(nChains)
from numpyro.infer import NUTS,MCMC,init_to_median,init_to_value
from jax import random
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
import arviz as az
from autoregressive_spin_models import ar_spinMagTilt
from getData import *

# Run over several chains to check convergence

# Get dictionaries holding injections and posterior samples
injectionDict = getInjections(reweight=False)#sample_limit=10000,reweight=True,weighting_function=reweighting_function_archi)
sampleDict = getSamples(sample_limit=2000,reweight=True,weighting_function=reweighting_function_archi)

# Instantiate array to hold all combined primary mass samples
total_Samples = 0
all_chi_samples = np.array([])
all_cost_samples = np.array([])

# Loop across individual events
for key in sampleDict:

    # Append samples together and compute range of indices holding this event's samples
    nSamples = len(sampleDict[key]['a1'])
    all_chi_samples = np.append(all_chi_samples,sampleDict[key]['a1'])
    all_cost_samples = np.append(all_cost_samples,sampleDict[key]['cost1'])
    sampleDict[key]['a1_ar_indices'] = np.arange(total_Samples,total_Samples+nSamples)
    total_Samples += nSamples

    nSamples = len(sampleDict[key]['a2'])
    all_chi_samples = np.append(all_chi_samples,sampleDict[key]['a2'])
    all_cost_samples = np.append(all_cost_samples,sampleDict[key]['cost2'])
    sampleDict[key]['a2_ar_indices'] = np.arange(total_Samples,total_Samples+nSamples)
    total_Samples += nSamples

# Similarly, get data from injection set
nInjections = len(injectionDict['a1'])
all_chi_samples = np.append(all_chi_samples,injectionDict['a1'])
all_cost_samples = np.append(all_cost_samples,injectionDict['cost1'])
a1_injection_indices = np.arange(total_Samples,total_Samples+nInjections)
total_Samples += nInjections

nInjections = len(injectionDict['a2'])
all_chi_samples = np.append(all_chi_samples,injectionDict['a2'])
all_cost_samples = np.append(all_cost_samples,injectionDict['cost2'])
a2_injection_indices = np.arange(total_Samples,total_Samples+nInjections)
total_Samples += nInjections

# Jitter as needed
chi_sorting = np.argsort(all_chi_samples)
toJitter_sortedInds = np.where(np.abs(np.diff(all_chi_samples[chi_sorting]))<1e-12)[0]
toJitter_inds = chi_sorting[toJitter_sortedInds]
all_chi_samples[toJitter_inds] += np.random.normal(loc=0,scale=1e-10,size=len(toJitter_inds))

# Get sorting information and differences between adjacent samples
chi_sorting = np.argsort(all_chi_samples)
chi_sorting_into_events = np.argsort(chi_sorting)
chi_deltas = np.diff(all_chi_samples[chi_sorting])
print(jnp.min(chi_deltas))

# Next jitter cost values
cost_sorting = np.argsort(all_cost_samples)
toJitter_sortedInds = np.where(np.abs(np.diff(all_cost_samples[cost_sorting]))<1e-12)[0]
toJitter_inds = cost_sorting[toJitter_sortedInds]
all_cost_samples[toJitter_inds] += np.random.normal(loc=0,scale=1e-10,size=len(toJitter_inds))

# Sorting and deltas
cost_sorting = np.argsort(all_cost_samples)
cost_sorting_into_events = np.argsort(cost_sorting)
cost_deltas = np.diff(all_cost_samples[cost_sorting])

# Package into dictionary
full_chi_data = {'chi_allSamples':all_chi_samples[chi_sorting],
                'chi_deltas':chi_deltas,
                'chi_reverseSorting':chi_sorting_into_events,
                'ind_a01':np.argmin((all_chi_samples[chi_sorting]-0.1)**2.),
                'cost_allSamples':all_cost_samples[cost_sorting],
                'cost_deltas':cost_deltas,
                'cost_reverseSorting':cost_sorting_into_events,
                'a1_injections_from_allSamples':a1_injection_indices,
                'a2_injections_from_allSamples':a2_injection_indices}

# Set up NUTS sampler over our likelihood
init_values = {
            'ar_chi_std':1.,
            'log_ar_chi_tau':0.,
            'ar_cost_std':0.5,
            'log_ar_cost_tau':0.,
            }
kernel = NUTS(ar_spinMagTilt,\
                dense_mass=[("ar_chi_std","log_ar_chi_tau"),("ar_cost_std","log_ar_cost_tau")],
                init_strategy=init_to_value(values=init_values))
mcmc = MCMC(kernel,num_warmup=100,num_samples=100,num_chains=nChains)

# Choose a random key and run over our model
rng_key = random.PRNGKey(170817)
rng_key,rng_key_ = random.split(rng_key)
mcmc.run(rng_key_,sampleDict,injectionDict,full_chi_data)
mcmc.print_summary()

# Save out data
data = az.from_numpyro(mcmc)
#az.to_netcdf(data,"/mnt/ceph/users/tcallister/autoregressive-pop-modeling-data/alt_ar_chi.cdf")
#np.save('/mnt/ceph/users/tcallister/autoregressive-pop-modeling-data/alt_ar_chi_data.npy',full_chi_data)
az.to_netcdf(data,"../data/ar_chi_cost.cdf")
np.save('../data/ar_chi_cost_data.npy',full_chi_data)
