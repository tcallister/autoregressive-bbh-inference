import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from jax.scipy.special import erf
from jax import soft_pmap,vmap,lax
import numpy as np
from utilities import *

def ar_mergerRate(sampleDict,injectionDict,full_z_data):

    """
    Implementation of a Gaussian effective spin distribution for inference within `numpyro`

    Parameters
    ----------
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections
    """

    all_z_samples = full_z_data['z_allSamples']
    ind_z02 = full_z_data['ind_z02']
    z_deltas = full_z_data['z_deltas']
    z_deltas_low = z_deltas[:ind_z02][::-1]
    z_deltas_high = z_deltas[ind_z02:]

    ############################################################
    # First sample the properties of our autoregressive process
    # First get variance of the process
    # We are imposing a steep power-law prior on this parameter
    ar_z_std = numpyro.sample("ar_z_std",dist.HalfNormal(1.))
    numpyro.factor("ar_z_std_prior",ar_z_std**2/2. - (ar_z_std/1.177)**4)

    # Finally the autocorrelation length
    # Since the posterior for this parameter runs up against prior boundaries, sample in logit space
    logit_ar_z_tau = numpyro.sample("logit_ar_z_tau",dist.Normal(0,logit_std))
    ar_z_tau,jac_ar_z_tau = get_value_from_logit(logit_ar_z_tau,0.1,2.)
    numpyro.factor("p_ar_z_tau",logit_ar_z_tau**2/(2.*logit_std**2)-jnp.log(jac_ar_z_tau))
    numpyro.deterministic("ar_z_tau",ar_z_tau)
    numpyro.factor("z_regularization",-(ar_z_std/jnp.sqrt(ar_z_tau))**2)

    # Sample an initial rate density at reference point
    ln_f_z_ref_unscaled = numpyro.sample("ln_f_z_ref_unscaled",dist.Normal(0,1))
    ln_f_z_ref = ln_f_z_ref_unscaled*ar_z_std

    # Generate forward steps
    z_steps_forward = numpyro.sample("z_steps_forward",dist.Normal(0,1),sample_shape=(z_deltas_high.size,))
    z_phis_forward = jnp.exp(-z_deltas_high/ar_z_tau)
    z_ws_forward = jnp.sqrt(1.-jnp.exp(-2.*z_deltas_high/ar_z_tau))*(ar_z_std*z_steps_forward)
    final,ln_f_zs_high = lax.scan(build_ar1,ln_f_z_ref,jnp.transpose(jnp.array([z_phis_forward,z_ws_forward]))) 
    ln_f_zs = jnp.append(ln_f_z_ref,ln_f_zs_high)

    # Generate backward steps
    z_steps_backward = numpyro.sample("z_steps_backward",dist.Normal(0,1),sample_shape=(z_deltas_low.size,))
    z_phis_backward = jnp.exp(-z_deltas_low/ar_z_tau)
    z_ws_backward = jnp.sqrt(1.-jnp.exp(-2.*z_deltas_low/ar_z_tau))*(ar_z_std*z_steps_backward)
    final,ln_f_zs_low = lax.scan(build_ar1,ln_f_z_ref,jnp.transpose(jnp.array([z_phis_backward,z_ws_backward])))
    ln_f_zs = jnp.append(ln_f_zs_low[::-1],ln_f_zs)

    # Exponentiate and save
    f_zs = jnp.exp(ln_f_zs)
    f_zs_eventSorted = f_zs[full_z_data['z_reverseSorting']]
    numpyro.deterministic("f_zs",f_zs)

    ##############################
    # Remaining degrees of freedom
    ##############################
    
    # Sample our hyperparameters
    # alpha: Power-law index on primary mass distribution
    # mu_m1: Location of gaussian peak in primary mass distribution
    # sig_m1: Width of gaussian peak
    # f_peak: Fraction of events comprising gaussian peak
    # mMax: Location at which BBH mass distribution tapers off
    # mMin: Lower boundary at which BBH mass distribution tapers off
    # dmMax: Taper width above maximum mass
    # dmMin: Taper width below minimum mass
    # bq: Power-law index on the conditional secondary mass distribution p(m2|m1)
    # mu: Mean of the chi-effective distribution
    # logsig_chi: Log10 of the chi-effective distribution's standard deviation

    logR20 = numpyro.sample("logR20",dist.Uniform(-6,3))
    R20 = numpyro.deterministic("R20",10.**logR20)

    alpha = numpyro.sample("alpha",dist.Normal(0,6))
    mu_m1 = numpyro.sample("mu_m1",dist.Uniform(20,50))
    mMin = numpyro.sample("mMin",dist.Uniform(5,15))
    bq = numpyro.sample("bq",dist.Normal(0,4))
    mu_chi = numpyro.sample("mu_chi",dist.Uniform(0.,1.))
    logsig_chi = numpyro.sample("logsig_chi",dist.Uniform(-1.,0.))
    mu_cost=1

    logit_sig_m1 = numpyro.sample("logit_sig_m1",dist.Normal(0,logit_std))
    logit_log_f_peak = numpyro.sample("logit_log_f_peak",dist.Normal(0,logit_std))
    logit_mMax = numpyro.sample("logit_mMax",dist.Normal(0,logit_std))
    logit_log_dmMin = numpyro.sample("logit_log_dmMin",dist.Normal(0,logit_std))
    logit_log_dmMax = numpyro.sample("logit_log_dmMax",dist.Normal(0,logit_std))
    logit_sig_cost = numpyro.sample("logit_sig_cost",dist.Normal(0,logit_std))

    sig_m1,jac_sig_m1 = get_value_from_logit(logit_sig_m1,2.,15.)
    log_f_peak,jac_log_f_peak = get_value_from_logit(logit_log_f_peak,-3,0.)
    mMax,jac_mMax = get_value_from_logit(logit_mMax,50.,100.)
    log_dmMin,jac_log_dmMin = get_value_from_logit(logit_log_dmMin,-1,1)
    log_dmMax,jac_log_dmMax = get_value_from_logit(logit_log_dmMax,0.5,1.5)
    sig_cost,jac_sig_cost = get_value_from_logit(logit_sig_cost,0.5,1.5)

    numpyro.deterministic("sig_m1",sig_m1)
    numpyro.deterministic("log_f_peak",log_f_peak)
    numpyro.deterministic("mMax",mMax)
    numpyro.deterministic("log_dmMin",log_dmMin)
    numpyro.deterministic("log_dmMax",log_dmMax)
    numpyro.deterministic("sig_cost",sig_cost)

    numpyro.factor("p_sig_m1",logit_sig_m1**2/(2.*logit_std**2)-jnp.log(jac_sig_m1))
    numpyro.factor("p_log_f_peak",logit_log_f_peak**2/(2.*logit_std**2)-jnp.log(jac_log_f_peak))
    numpyro.factor("p_mMax",logit_mMax**2/(2.*logit_std**2)-jnp.log(jac_mMax))
    numpyro.factor("p_log_dmMin",logit_log_dmMin**2/(2.*logit_std**2)-jnp.log(jac_log_dmMin))
    numpyro.factor("p_log_dmMax",logit_log_dmMax**2/(2.*logit_std**2)-jnp.log(jac_log_dmMax))
    numpyro.factor("p_sig_cost",logit_sig_cost**2/(2.*logit_std**2)-jnp.log(jac_sig_cost))

    # Normalization
    p_m1_norm = massModel(20.,alpha,mu_m1,sig_m1,10.**log_f_peak,mMax,mMin,10.**log_dmMax,10.**log_dmMin)

    # Read out found injections
    # Note that `pop_reweight` is the inverse of the draw weights for each event
    m1_det = injectionDict['m1']
    m2_det = injectionDict['m2']
    a1_det = injectionDict['a1']
    a2_det = injectionDict['a2']
    cost1_det = injectionDict['cost1']
    cost2_det = injectionDict['cost2']
    z_det = injectionDict['z']
    dVdz_det = injectionDict['dVdz']
    p_draw = injectionDict['p_draw_m1m2z']*injectionDict['p_draw_a1a2cost1cost2']

    # Compute proposed population weights
    f_m1_det = massModel(m1_det,alpha,mu_m1,sig_m1,10.**log_f_peak,mMax,mMin,10.**log_dmMax,10.**log_dmMin)/p_m1_norm
    p_m2_det = (1.+bq)*m2_det**bq/(m1_det**(1.+bq)-tmp_min**(1.+bq))
    p_a1_det = truncatedNormal(a1_det,mu_chi,10.**logsig_chi,0,1)
    p_a2_det = truncatedNormal(a2_det,mu_chi,10.**logsig_chi,0,1)
    p_cost1_det = truncatedNormal(cost1_det,mu_cost,sig_cost,-1,1)
    p_cost2_det = truncatedNormal(cost2_det,mu_cost,sig_cost,-1,1)
    f_z_det = dVdz_det*f_zs_eventSorted[full_z_data['injections_from_allSamples']]/(1.+z_det)
    R_pop_det = R20*f_m1_det*p_m2_det*f_z_det*p_a1_det*p_a2_det*p_cost1_det*p_cost2_det

    # Form ratio of proposed weights over draw weights
    inj_weights = R_pop_det/(p_draw/2.)
    
    # As a fit diagnostic, compute effective number of injections
    nEff_inj = jnp.sum(inj_weights)**2/jnp.sum(inj_weights**2)
    nObs = 1.0*len(sampleDict)
    numpyro.deterministic("nEff_inj_per_event",nEff_inj/nObs)

    # Compute net detection efficiency and add to log-likelihood
    Nexp = jnp.sum(inj_weights)/injectionDict['nTrials']
    numpyro.factor("rate",-Nexp)
    
    # This function defines the per-event log-likelihood
    # m1_sample: Primary mass posterior samples
    # m2_sample: Secondary mass posterior samples
    # z_sample: Redshift posterior samples
    # dVdz_sample: Differential comoving volume at each sample location
    # Xeff_sample: Effective spin posterior samples
    # priors: PE priors on each sample
    def logp(m1_sample,m2_sample,a1_sample,a2_sample,cost1_sample,cost2_sample,z_sample,dVdz_sample,priors,z_ar_indices):

        # Compute proposed population weights
        f_m1 = massModel(m1_sample,alpha,mu_m1,sig_m1,10.**log_f_peak,mMax,mMin,10.**log_dmMax,10.**log_dmMin)/p_m1_norm
        p_m2 = (1.+bq)*m2_sample**bq/(m1_sample**(1.+bq)-tmp_min**(1.+bq))
        p_a1 = truncatedNormal(a1_sample,mu_chi,10.**logsig_chi,0,1)
        p_a2 = truncatedNormal(a2_sample,mu_chi,10.**logsig_chi,0,1)
        p_cost1 = truncatedNormal(cost1_sample,mu_cost,sig_cost,-1,1)
        p_cost2 = truncatedNormal(cost2_sample,mu_cost,sig_cost,-1,1)
        f_z = dVdz_sample*f_zs_eventSorted[z_ar_indices]/(1.+z_sample)
        R_pop = R20*f_m1*p_m2*f_z*p_a1*p_a2*p_cost1*p_cost2

        mc_weights = R_pop/priors
        
        # Compute effective number of samples and return log-likelihood
        n_eff = jnp.sum(mc_weights)**2/jnp.sum(mc_weights**2)     
        return jnp.log(jnp.mean(mc_weights)),n_eff
    
    # Map the log-likelihood function over each event in our catalog
    log_ps,n_effs = vmap(logp)(
                        jnp.array([sampleDict[k]['m1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['m2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['a1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['a2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['cost1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['cost2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['z'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['dVc_dz'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['z_prior'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['ar_indices'] for k in sampleDict]))
        
    # As a diagnostic, save minimum number of effective samples across all events
    numpyro.deterministic('min_log_neff',jnp.min(jnp.log10(n_effs)))

    # Tally log-likelihoods across our catalog
    numpyro.factor("logp",jnp.sum(log_ps))