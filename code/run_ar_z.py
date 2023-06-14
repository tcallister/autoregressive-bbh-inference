import numpyro
nChains = 1
numpyro.set_host_device_count(nChains)
from numpyro.infer import NUTS,MCMC,init_to_value
from jax import random
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
import arviz as az
import numpy as np
np.random.seed(138)
from autoregressive_redshift_models import ar_mergerRate
from getData import *
from utilities import compute_prior_params

# Run over several chains to check convergence

# Get dictionaries holding injections and posterior samples
injectionDict = getInjections(reweight=False)
sampleDict = getSamples(sample_limit=3000,reweight=False)

# Instantiate array to hold all combined primary mass samples
total_z_Samples = 0
all_z_samples = np.array([])

# Loop across individual events
for key in sampleDict:

    # Append samples together and compute range of indices holding this event's samples
    nSamples = len(sampleDict[key]['z'])
    all_z_samples = np.append(all_z_samples,sampleDict[key]['z'])
    sampleDict[key]['ar_indices'] = np.arange(total_z_Samples,total_z_Samples+nSamples)
    total_z_Samples += nSamples

# Similarly, get data from injection set
nInjections = len(injectionDict['z'])
all_z_samples = np.append(all_z_samples,injectionDict['z'])
z_injection_indices = np.arange(total_z_Samples,total_z_Samples+nInjections)
total_z_Samples += nInjections

# Jitter as needed
z_sorting = np.argsort(all_z_samples)
toJitter_sortedInds = np.where(np.abs(np.diff(all_z_samples[z_sorting]))<1e-12)[0]
toJitter_inds = z_sorting[toJitter_sortedInds]
all_z_samples[toJitter_inds] += np.random.normal(loc=0,scale=1e-10,size=len(toJitter_inds))

# Get sorting information and differences between adjacent samples
z_sorting = np.argsort(all_z_samples)
z_sorting_into_events = np.argsort(z_sorting)
z_deltas = np.diff(all_z_samples[z_sorting])

# Package into dictionary
full_z_data = {'z_allSamples':all_z_samples[z_sorting],
                'z_deltas':z_deltas,
                'z_reverseSorting':z_sorting_into_events,
                'ind_z02':np.argmin((all_z_samples[z_sorting]-0.2)**2.),
                'injections_from_allSamples':z_injection_indices}

# Compute hyperparameter constraints
dR_max = 100
dR_event = 2
N = 69
Delta_z = 2
z_std_std,z_ln_tau_mu,z_ln_tau_std,z_regularization_std = compute_prior_params(dR_max,dR_event,Delta_z,N)

# Set up NUTS sampler over our likelihood
#init_values = {'ar_z_std':0.8,'ar_z_tau':0.2}
kernel = NUTS(ar_mergerRate,dense_mass=[("ar_z_std","ar_ratio","logR20")])#,init_strategy=init_to_value(values=init_values),target_accept_prob=0.9)
mcmc = MCMC(kernel,num_warmup=500,num_samples=500,num_chains=nChains)

# Choose a random key and run over our model
rng_key = random.PRNGKey(139)
rng_key,rng_key_ = random.split(rng_key)
mcmc.run(rng_key_,sampleDict,injectionDict,full_z_data,\
    z_std_std=z_std_std,z_ln_tau_mu=z_ln_tau_mu,z_ln_tau_std=z_ln_tau_std,z_regularization_std=z_regularization_std)
mcmc.print_summary()

# Save out data
data = az.from_numpyro(mcmc)
az.to_netcdf(data,"./../data/ar_z_test.cdf")
np.save('./../data/ar_z_data_test.npy',full_z_data)
#az.to_netcdf(data,"/mnt/ceph/users/tcallister/autoregressive-bbh-inference-data/final-ar_z.cdf")
#np.save('/mnt/ceph/users/tcallister/autoregressive-bbh-inference-data/final-ar_z_data.npy',full_z_data)

