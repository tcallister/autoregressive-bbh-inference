import numpyro
import sys
nChains = 3
numpyro.set_host_device_count(nChains)
from numpyro.infer import NUTS,MCMC,init_to_value
from jax import random
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
import arviz as az
import numpy as np
np.random.seed(977)
from autoregressive_mass_models import ar_lnm_q
from getData import *

# Run over several chains to check convergence

# Get dictionaries holding injections and posterior samples
injectionDict = getInjections(reweight=False)
sampleDict = getSamples(sample_limit=2000,reweight=False)

# Instantiate array to hold all combined primary mass samples
total_lnm_Samples = 0
all_lnm_samples = np.array([])

total_q_Samples = 0
all_q_samples = np.array([])

# Loop across individual events
for key in sampleDict:

    # Append samples together and compute range of indices holding this event's samples
    nSamples = len(sampleDict[key]['m1'])
    all_lnm_samples = np.append(all_lnm_samples,np.log(sampleDict[key]['m1']))
    sampleDict[key]['m1_ar_indices'] = np.arange(total_lnm_Samples,total_lnm_Samples+nSamples)
    total_lnm_Samples += nSamples

    nSamples = len(sampleDict[key]['m2'])
    all_lnm_samples = np.append(all_lnm_samples,np.log(sampleDict[key]['m2']))
    sampleDict[key]['m2_ar_indices'] = np.arange(total_lnm_Samples,total_lnm_Samples+nSamples)
    total_lnm_Samples += nSamples

    nSamples = len(sampleDict[key]['m1'])
    all_q_samples = np.append(all_q_samples,sampleDict[key]['m2']/sampleDict[key]['m1'])
    sampleDict[key]['q_ar_indices'] = np.arange(total_q_Samples,total_q_Samples+nSamples)
    total_q_Samples += nSamples

# Similarly, get data from injection set
nInjections = len(injectionDict['m1'])
all_lnm_samples = np.append(all_lnm_samples,np.log(injectionDict['m1']))
m1_injection_indices = np.arange(total_lnm_Samples,total_lnm_Samples+nInjections)
total_lnm_Samples += nInjections

nInjections = len(injectionDict['m2'])
all_lnm_samples = np.append(all_lnm_samples,np.log(injectionDict['m2']))
m2_injection_indices = np.arange(total_lnm_Samples,total_lnm_Samples+nInjections)
total_lnm_Samples += nInjections

nInjections = len(injectionDict['m1'])
all_q_samples = np.append(all_q_samples,injectionDict['m2']/injectionDict['m1'])
q_injection_indices = np.arange(total_q_Samples,total_q_Samples+nInjections)
total_q_Samples += nInjections

# Jitter as needed
lnm_sorting = np.argsort(all_lnm_samples)
toJitter_sortedInds = np.where(np.abs(np.diff(all_lnm_samples[lnm_sorting]))<1e-12)[0]
toJitter_inds = lnm_sorting[toJitter_sortedInds]
all_lnm_samples[toJitter_inds] += np.random.normal(loc=0,scale=1e-10,size=len(toJitter_inds))

# Get sorting information and differences between adjacent samples
lnm_sorting = np.argsort(all_lnm_samples)
lnm_sorting_into_events = np.argsort(lnm_sorting)
lnm_deltas = np.diff(all_lnm_samples[lnm_sorting])

# Repeat with mass ratio
q_sorting = np.argsort(all_q_samples)
toJitter_sortedInds = np.where(np.abs(np.diff(all_q_samples[q_sorting]))<1e-12)[0]
toJitter_inds = q_sorting[toJitter_sortedInds]
all_q_samples[toJitter_inds] += np.random.normal(loc=0,scale=1e-10,size=len(toJitter_inds))

q_sorting = np.argsort(all_q_samples)
q_sorting_into_events = np.argsort(q_sorting)
q_deltas = np.diff(all_q_samples[q_sorting])

# Package into dictionary
full_lnm_q_data = {'lnm_allSamples':all_lnm_samples[lnm_sorting],
                'lnm_deltas':lnm_deltas,
                'lnm_reverseSorting':lnm_sorting_into_events,
                'ind_m20':np.argmin((all_lnm_samples[lnm_sorting]-np.log(20.))**2.),
                'q_allSamples':all_q_samples[q_sorting],
                'q_deltas':q_deltas,
                'q_reverseSorting':q_sorting_into_events,
                'm1_injections_from_allSamples':m1_injection_indices,
                'm2_injections_from_allSamples':m2_injection_indices,
                'q_injections_from_allSamples':q_injection_indices}

# Set up NUTS sampler over our likelihood
init_values = {
            'ar_lnm_std':1.,
            'log_ar_lnm_tau':0.,
            'ar_q_std':1.,
            'log_ar_q_tau':0.
            }
kernel = NUTS(ar_lnm_q,init_strategy=init_to_value(values=init_values),dense_mass=[("ar_lnm_std","log_ar_lnm_tau"),("ar_q_std","log_ar_q_tau")])
mcmc = MCMC(kernel,num_warmup=1000,num_samples=1500,num_chains=nChains)

# Choose a random key and run over our model
rng_key = random.PRNGKey(170729)
rng_key,rng_key_ = random.split(rng_key)
mcmc.run(rng_key_,sampleDict,injectionDict,full_lnm_q_data)
mcmc.print_summary()

# Save out data
data = az.from_numpyro(mcmc)
az.to_netcdf(data,"/mnt/ceph/users/tcallister/autoregressive-bbh-inference-data/ar_lnm_q.cdf")
np.save('/mnt/ceph/users/tcallister/autoregressive-bbh-inference-data/ar_lnm_q_data.npy',full_lnm_q_data)
#az.to_netcdf(data,"../data/ar_lnm_q.cdf")
#np.save('../data/ar_lnm_q_data.npy',full_lnm_q_data)

