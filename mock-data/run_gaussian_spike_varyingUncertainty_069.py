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
np.random.seed(111)
from autoregressive_model import ar_model
import sys
sys.path.append("./../code/")
from utilities import compute_prior_params

"""
This script runs an autoregressive population inference on the mock data in `gaussian_spike_samples_varyingUncertainty.npy`,
which is itself created by the jupyter notebook `gen_fake_samples.ipynb`.
In particular, it will analyze the first 69 mock posteriors in this data set, to mirror the size of our actual GWTC-3 catalog.
"""

# Get dictionaries holding posterior samples
# Restrict to the first 69 mock events
sampleDict = np.load('gaussian_spike_samples_varyingUncertainty.npy',allow_pickle=True)[()]
sampleDict = {i:sampleDict[i] for i in range(69)}

# Instantiate array to hold all combined posterior samples
total_Samples = 0
all_samples = np.array([])

# Loop across individual events
for key in sampleDict:

    # Append samples together and compute range of indices holding this event's samples
    nSamples = len(sampleDict[key]['samps'])
    all_samples = np.append(all_samples,sampleDict[key]['samps'])
    sampleDict[key]['ar_indices'] = np.arange(total_Samples,total_Samples+nSamples)
    total_Samples += nSamples

# Assume no selection effects.
# At the same time, we want to realistically mirror how found injections influence the AR(1) inference,
# so we will still characterize selection effects by randomly "detecting" a set of mock injections
# spread uniformly throughout parameter space
injectionDict = {'inj':np.random.random(10000)*2.-1.,'p_draw':np.ones(10000)*(1./2.),'nTrials':10000.}

# Read out values from injection set
nInjections = len(injectionDict['inj'])
all_samples = np.append(all_samples,injectionDict['inj'])
injection_indices = np.arange(total_Samples,total_Samples+nInjections)
total_Samples += nInjections

# Get sorting information and differences between adjacent samples
sorting = np.argsort(all_samples)
sorting_into_events = np.argsort(sorting)
deltas = np.diff(all_samples[sorting])

# Package into dictionary
full_data = {'all_samples':all_samples[sorting],
                'deltas':deltas,
                'reverseSorting':sorting_into_events,
                'ind_ref':np.argmin((all_samples[sorting])**2.),
                'injections_from_allSamples':injection_indices}

# Compute hyperparameter prior constraints
dR_max = 100
dR_event = 2
N = 69
DeltaX = 2.
std_std,ln_tau_mu,ln_tau_std,regularization_std = compute_prior_params(dR_max,dR_event,DeltaX,N)

# Set up NUTS sampler over our likelihood
kernel = NUTS(ar_model)
mcmc = MCMC(kernel,num_warmup=400,num_samples=750,num_chains=nChains)

# Choose a random key and run over our model
rng_key = random.PRNGKey(211)
rng_key,rng_key_ = random.split(rng_key)
mcmc.run(rng_key_,sampleDict,injectionDict,full_data,std_std=std_std,ln_tau_mu=ln_tau_mu,ln_tau_std=ln_tau_std,regularization_std=regularization_std)
mcmc.print_summary()

# Save out data
data = az.from_numpyro(mcmc)
az.to_netcdf(data,"ar_gaussian_spike_varyingUncertainty_069.cdf")
np.save('ar_gaussian_spike_varyingUncertainty_069_data.npy',full_data)

