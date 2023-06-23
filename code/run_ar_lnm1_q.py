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
np.random.seed(1207)
from autoregressive_mass_models import ar_lnm1_q
from getData import *
from utilities import compute_prior_params

# Get dictionaries holding injections and posterior samples
injectionDict = getInjections(reweight=False)
sampleDict = getSamples(sample_limit=2000,reweight=True,weighting_function=reweighting_function_arlnm1_q)

# Instantiate array to hold all combined primary mass samples
total_lnm1_Samples = 0
all_lnm1_samples = np.array([])

total_q_Samples = 0
all_q_samples = np.array([])

# Loop across individual events
for key in sampleDict:

    # Append samples together and compute range of indices holding this event's samples
    nSamples = len(sampleDict[key]['m1'])
    all_lnm1_samples = np.append(all_lnm1_samples,np.log(sampleDict[key]['m1']))
    all_q_samples = np.append(all_q_samples,sampleDict[key]['m2']/sampleDict[key]['m1'])
    sampleDict[key]['ar_indices'] = np.arange(total_lnm1_Samples,total_lnm1_Samples+nSamples)
    total_lnm1_Samples += nSamples

# Similarly, get data from injection set
nInjections = len(injectionDict['m1'])
all_lnm1_samples = np.append(all_lnm1_samples,np.log(injectionDict['m1']))
m1_injection_indices = np.arange(total_lnm1_Samples,total_lnm1_Samples+nInjections)
total_lnm1_Samples += nInjections

all_q_samples = np.append(all_q_samples,injectionDict['m2']/injectionDict['m1'])
q_injection_indices = np.arange(total_q_Samples,total_q_Samples+nInjections)
total_q_Samples += nInjections

# Jitter as needed
lnm1_sorting = np.argsort(all_lnm1_samples)
toJitter_sortedInds = np.where(np.abs(np.diff(all_lnm1_samples[lnm1_sorting]))<1e-12)[0]
toJitter_inds = lnm1_sorting[toJitter_sortedInds]
all_lnm1_samples[toJitter_inds] += np.random.normal(loc=0,scale=1e-10,size=len(toJitter_inds))

# Get sorting information and differences between adjacent samples
lnm1_sorting = np.argsort(all_lnm1_samples)
lnm1_sorting_into_events = np.argsort(lnm1_sorting)
lnm1_deltas = np.diff(all_lnm1_samples[lnm1_sorting])

# Repeat with mass ratio
q_sorting = np.argsort(all_q_samples)
toJitter_sortedInds = np.where(np.abs(np.diff(all_q_samples[q_sorting]))<1e-12)[0]
toJitter_inds = q_sorting[toJitter_sortedInds]
all_q_samples[toJitter_inds] += np.random.normal(loc=0,scale=1e-10,size=len(toJitter_inds))

q_sorting = np.argsort(all_q_samples)
q_sorting_into_events = np.argsort(q_sorting)
q_deltas = np.diff(all_q_samples[q_sorting])

# Package into dictionary
full_lnm1_q_data = {'all_lnm1_samples':all_lnm1_samples[lnm1_sorting],
                'lnm1_deltas':lnm1_deltas,
                'lnm1_reverseSorting':lnm1_sorting_into_events,
                'ind_m20':np.argmin((all_lnm1_samples[lnm1_sorting]-np.log(20.))**2.),
                'all_q_samples':all_q_samples[q_sorting],
                'q_deltas':q_deltas,
                'q_reverseSorting':q_sorting_into_events,
                'injections_from_allSamples':m1_injection_indices}

# Compute hyperparameter constraints
dR_max = 100
dR_event = 2
N = 69
Delta_lnm1 = 4.
Delta_q = 1.
lnm1_std_std,lnm1_ln_tau_mu,lnm1_ln_tau_std,lnm1_regularization_std = compute_prior_params(dR_max,dR_event,Delta_lnm1,N)
q_std_std,q_ln_tau_mu,q_ln_tau_std,q_regularization_std = compute_prior_params(dR_max,dR_event,Delta_q,N)

# Set up NUTS sampler over our likelihood
init_values = {
            'ar_lnm1_std':1.,
            'log_ar_lnm1_tau':0.5,
            'ar_q_std':1.,
            'log_ar_q_tau':0.,
            'mu_chi':0.1
            }
kernel = NUTS(ar_lnm1_q,init_strategy=init_to_value(values=init_values),dense_mass=[("ar_lnm1_std","log_ar_lnm1_tau"),("ar_q_std","log_ar_q_tau")])
mcmc = MCMC(kernel,num_warmup=1000,num_samples=1500,num_chains=nChains)

# Choose a random key and run over our model
rng_key = random.PRNGKey(1206)
rng_key,rng_key_ = random.split(rng_key)
mcmc.run(rng_key_,sampleDict,injectionDict,full_lnm1_q_data,\
    lnm1_std_std=lnm1_std_std,lnm1_ln_tau_mu=lnm1_ln_tau_mu,lnm1_ln_tau_std=lnm1_ln_tau_std,lnm1_regularization_std=lnm1_regularization_std,\
    q_std_std=q_std_std,q_ln_tau_mu=q_ln_tau_mu,q_ln_tau_std=q_ln_tau_std,q_regularization_std=q_regularization_std)

# Save out data
data = az.from_numpyro(mcmc)
az.to_netcdf(data,"/project2/kicp/tcallister/autoregressive-bbh-inference-data/ar_lnm1_q.cdf")
np.save('/project2/kicp/tcallister/autoregressive-bbh-inference-data/ar_lnm1_q_data.npy',full_lnm1_q_data)
