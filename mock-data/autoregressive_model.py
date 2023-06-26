import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
import jax.scipy
from jax import vmap,lax
import numpy as np
import sys
sys.path.append('./../code/')
from utilities import *

def ar_model(sampleDict,injectionDict,full_data,std_std=1.66,ln_tau_mu=0,ln_tau_std=4.7,regularization_std=0.83,empty_obs=False):

    """
    Likelihood model performing inference with an AR(1) processes, for use within `numpyro`.
    Used to analyze mock data to produce results in Appendix C of paper text

    Parameters
    ----------
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections
    full_data : dict
        Precomputed dictionary containing accumulated list of *all* samples, among both event posteriors and injections.
        Dictionary also contains information needed to sort list into numerical order, or into event-based order.
        Prepared in e.g. `run_gaussian_varyingUncertainty_069.py`.
    std_std : float
        Standard deviation of prior on `ar_std`
    ln_tau_mu : float
        Mean of prior on `log_ar_tau`
    ln_tau_std : float
        Standard deviation of prior on `log_ar_tau`
    regularization_std : float
        Standard deviation of regularizing prior on the ratio `ar_std/jnp.sqrt(ar_tau)`
    empty_obs : bool
        If `True`, will neglect likelihood contributions from observed events (default `False`)
    """

    # Read complete list of sorted samples and deltas between them
    # Additionally split deltas into those below and above our reference mass
    all_samples = full_data['all_samples']
    deltas = full_data['deltas']
    ind_ref = full_data['ind_ref']
    deltas_low = deltas[:ind_ref][::-1]
    deltas_high = deltas[ind_ref:]

    ###################################
    # Constructing our AR1 process
    ###################################

    # First get variance of the process
    # We will sample from a half normal distribution, but override this with a quadratic prior
    # on the processes' standard deviation; see Eq. B1
    ar_std = numpyro.sample("ar_std",dist.HalfNormal(std_std))

    # Next, the autocorrelation length
    log_ar_tau = numpyro.sample("log_ar_tau",dist.Normal(ln_tau_mu,ln_tau_std))
    ar_tau = numpyro.deterministic("ar_tau",jnp.exp(log_ar_tau))

    # Sample an initial rate density at the reference mass point
    # First draw un unscaled variable from N(0,1), then rescale by the standard deviation
    ln_f_ref_unscaled = numpyro.sample("ln_f_ref_unscaled",dist.Normal(0,1))
    ln_f_ref = ln_f_ref_unscaled*ar_std

    # Generate forward steps and join to reference value, following the procedure outlined in Appendix A
    # First generate a sequence of unnormalized steps from N(0,1), then rescale to compute weights and innovations
    steps_forward = numpyro.sample("steps_forward",dist.Normal(0,1),sample_shape=(deltas_high.size,))
    phis_forward = jnp.exp(-deltas_high/ar_tau)
    ws_forward = jnp.sqrt(-jnp.expm1(-2.*deltas_high/ar_tau))*(ar_std*steps_forward)
    final,ln_f_high = lax.scan(build_ar1,ln_f_ref,jnp.transpose(jnp.array([phis_forward,ws_forward]))) 
    ln_fs = jnp.append(ln_f_ref,ln_f_high)

    # Generate backward steps and prepend to forward steps above following an analogous procedure
    steps_backward = numpyro.sample("steps_backward",dist.Normal(0,1),sample_shape=(deltas_low.size,))
    phis_backward = jnp.exp(-deltas_low/ar_tau)
    ws_backward = jnp.sqrt(-jnp.expm1(-2.*deltas_low/ar_tau))*(ar_std*steps_backward)
    final,ln_f_low = lax.scan(build_ar1,ln_f_ref,jnp.transpose(jnp.array([phis_backward,ws_backward])))
    ln_fs = jnp.append(ln_f_low[::-1],ln_fs)

    # Exponentiate and save
    fs = jnp.exp(ln_fs)
    numpyro.deterministic("fs",fs)

    # Reverse sort our AR process back into an array in which injections and each event's PE samples are grouped
    fs_eventSorted = fs[full_data['reverseSorting']]

    ##############################
    # Remaining degrees of freedom
    ##############################

    # Sample the merger rate at our reference mass and redshift values
    logR20 = numpyro.sample("logR20",dist.Uniform(-6,6))
    R20 = numpyro.deterministic("R20",10.**logR20)

    ###############################
    # Expected number of detections
    ###############################

    # Compute proposed population weights
    f_det = fs_eventSorted[full_data['injections_from_allSamples']]
    p_draw = injectionDict['p_draw']

    # All together, the quantity below is the detection rate dN/dm1*dm2*da1*da2*dcost1*dcost2*dz*dt_det
    R_pop_det = R20*f_det

    # Form ratio of proposed weights over draw weights
    # The division by 2 corresponds to the fact that injections are uniformly placed over the 2 year observation period
    inj_weights = R_pop_det/(p_draw/2.)
    
    # As a fit diagnostic, compute effective number of injections.
    nEff_inj = jnp.sum(inj_weights)**2/jnp.sum(inj_weights**2)
    nObs = 1.0*len(sampleDict)
    nEff_inj_per_event = numpyro.deterministic("nEff_inj_per_event",nEff_inj/nObs)

    # Compute total expected number of detections and add to log-likelihood
    Nexp = jnp.sum(inj_weights)/injectionDict['nTrials']
    numpyro.factor("rate",-Nexp)

    # As discussed in Appendix B, we need a regularizing log-likelihood factor to help stabilize our inference; see Eq. B3
    # Also prevent effective injection counts from becoming pathologically low
    numpyro.factor("regularization",-(ar_std/jnp.sqrt(ar_tau))**2/(2.*regularization_std**2))
    numpyro.factor("Neff_inj_penalty",jnp.log(1./(1.+(nEff_inj_per_event/4.)**(-30.))))

    ###############################
    # Compute per-event likelihoods
    ###############################

    # This function defines the per-event log-likelihood. It expects the following arguments:
    # `sample` : Arrays of posterior samples for the given event
    # `ar_indices` : Indices used to retrieve the correct AR1 rates corresponding to this event's samples
    def logp(sample,ar_indices):

        # Compute proposed population weights, analogous to calculation for injections done above
        # Use `ar_indices` to extract the correct values of `fs_eventSorted`
        # corresponding to each of this event's posterior samples
        f = fs_eventSorted[ar_indices]

        # From full rate
        R_pop = R20*f
        mc_weights = R_pop

        # Compute effective number of samples and return log-likelihood
        n_eff = jnp.sum(mc_weights)**2/jnp.sum(mc_weights**2)     
        return jnp.log(jnp.mean(mc_weights)),n_eff
    
    # Map the log-likelihood function over each event in our catalog
    if not empty_obs:

        log_ps,n_effs = vmap(logp)(
                            jnp.array([sampleDict[k]['samps'] for k in sampleDict]),
                            jnp.array([sampleDict[k]['ar_indices'] for k in sampleDict]))
            
        # As a diagnostic, save minimum number of effective samples across all events
        min_log_neff = numpyro.deterministic('min_log_neff',jnp.min(jnp.log10(n_effs)))

        # Penalize
        numpyro.factor("Neff_penalty",jnp.log(1./(1.+(min_log_neff/0.6)**(-30.))))

        # Tally log-likelihoods across our catalog
        numpyro.factor("logp",jnp.sum(log_ps))

