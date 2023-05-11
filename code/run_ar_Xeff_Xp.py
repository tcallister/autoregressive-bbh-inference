import numpyro
nChains = 3
numpyro.set_host_device_count(nChains)
from numpyro.infer import NUTS,MCMC,init_to_value
from jax import random
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
import arviz as az
import numpy as np
np.random.seed(111)
from autoregressive_spin_models import ar_Xeff_Xp
from getData import *

# Run over several chains to check convergence

# Get dictionaries holding injections and posterior samples
injectionDict = getInjections(reweight=False)
sampleDict = getSamples(sample_limit=2500,reweight=False)

# Instantiate array to hold all combined primary mass samples
total_Xeff_Samples = 0
all_Xeff_samples = np.array([])
all_Xp_samples = np.array([])

# Loop across individual events
for key in sampleDict:

    # Append samples together and compute range of indices holding this event's samples
    nSamples = len(sampleDict[key]['Xeff'])
    all_Xeff_samples = np.append(all_Xeff_samples,sampleDict[key]['Xeff'])
    all_Xp_samples = np.append(all_Xp_samples,sampleDict[key]['Xp'])
    sampleDict[key]['ar_indices'] = np.arange(total_Xeff_Samples,total_Xeff_Samples+nSamples)
    total_Xeff_Samples += nSamples

# Similarly, get data from injection set
nInjections = len(injectionDict['Xeff'])
all_Xeff_samples = np.append(all_Xeff_samples,injectionDict['Xeff'])
all_Xp_samples = np.append(all_Xp_samples,injectionDict['Xp'])
Xeff_injection_indices = np.arange(total_Xeff_Samples,total_Xeff_Samples+nInjections)
total_Xeff_Samples += nInjections

# Jitter as needed
Xeff_sorting = np.argsort(all_Xeff_samples)
toJitter_sortedInds = np.where(np.abs(np.diff(all_Xeff_samples[Xeff_sorting]))<1e-12)[0]
toJitter_inds = Xeff_sorting[toJitter_sortedInds]
all_Xeff_samples[toJitter_inds] += np.random.normal(loc=0,scale=1e-10,size=len(toJitter_inds))

# Get sorting information and differences between adjacent samples
Xeff_sorting = np.argsort(all_Xeff_samples)
Xeff_sorting_into_events = np.argsort(Xeff_sorting)
Xeff_deltas = np.diff(all_Xeff_samples[Xeff_sorting])

# Jitter as needed
Xp_sorting = np.argsort(all_Xp_samples)
toJitter_sortedInds = np.where(np.abs(np.diff(all_Xp_samples[Xp_sorting]))<1e-12)[0]
toJitter_inds = Xp_sorting[toJitter_sortedInds]
all_Xp_samples[toJitter_inds] += np.random.normal(loc=0,scale=1e-10,size=len(toJitter_inds))

# Get sorting information and differences between adjacent samples
Xp_sorting = np.argsort(all_Xp_samples)
Xp_sorting_into_events = np.argsort(Xp_sorting)
Xp_deltas = np.diff(all_Xp_samples[Xp_sorting])

# Package into dictionary
full_Xeff_Xp_data = {'Xeff_allSamples':all_Xeff_samples[Xeff_sorting],
                'Xeff_deltas':Xeff_deltas,
                'Xeff_reverseSorting':Xeff_sorting_into_events,
                'ind_Xeff0':np.argmin((all_Xeff_samples[Xeff_sorting])**2.),
                'Xp_allSamples':all_Xp_samples[Xp_sorting],
                'Xp_deltas':Xp_deltas,
                'Xp_reverseSorting':Xp_sorting_into_events,
                'ind_Xp02':np.argmin((all_Xp_samples[Xp_sorting]-0.2)**2.),
                'injections_from_allSamples':Xeff_injection_indices}

# Set up NUTS sampler over our likelihood
"""
init_values = {
            'ar_Xeff_std':1.8,
            'log_ar_Xeff_tau':0.,
            'ar_q_std':1.2,
            'log_ar_q_tau':0.,
            'mu_chi':0.1
            }
"""
kernel = NUTS(ar_Xeff_Xp)#,init_strategy=init_to_value(values=init_values),dense_mass=[("ar_z_std","log_ar_z_tau"),("ar_q_std","log_ar_q_tau")])
#mcmc = MCMC(kernel,num_warmup=1000,num_samples=1500,num_chains=nChains)
mcmc = MCMC(kernel,num_warmup=500,num_samples=500,num_chains=nChains)

# Choose a random key and run over our model
rng_key = random.PRNGKey(202)
rng_key,rng_key_ = random.split(rng_key)
mcmc.run(rng_key_,sampleDict,injectionDict,full_Xeff_Xp_data)
mcmc.print_summary()

# Save out data
data = az.from_numpyro(mcmc)
#az.to_netcdf(data,"/mnt/ceph/users/tcallister/autoregressive-bbh-inference-data/final-ar_Xeff_Xp.cdf")
#np.save('/mnt/ceph/users/tcallister/autoregressive-bbh-inference-data/final-ar_Xeff_Xp_data.npy',full_Xeff_Xp_data)
az.to_netcdf(data,"/project2/kicp/tcallister/autoregressive-bbh-inference-data/ar_Xeff_Xp.cdf")
np.save('/project2/kicp/tcallister/autoregressive-bbh-inference-data/ar_Xeff_Xp_data.npy',full_Xeff_Xp_data)

