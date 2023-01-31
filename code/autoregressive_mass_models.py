import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from jax import vmap,lax
import numpy as np
from custom_distributions import *
from utilities import *

def ar_lnm1_q(sampleDict,injectionDict,full_lnm1_q_data):

    """
    Likelihood model in which the BBH ln(m1) and q distributions are described as AR(1) processes, for use within `numpyro`.
    The distributions redshifts and spins are simultaneously fit.
    The BBH merger rate is assumed to grow as a power law in `(1+z)`.
    Spin magnitudes are described via a truncated normal distribution, and spin cosine tilts are modeled as another
    truncated normal, whose mean is fixed to `cos(theta)=1`.

    Parameters
    ----------
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog, as prepared by `getData.getSamples`
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections, as prepared by `getData.getInjections`
    full_lnm1_q_data : dict
        Precomputed dictionary containing accumulated list of *all* `lnm1` samples and `q`, among both event posteriors and injections.
        Dictionary also contains information needed to sort list into numerical order, or into event-based order.
        Prepared in `run_ar_lnm1_q.py`.
    """

    # Read complete list of sorted lnm1 samples and deltas between them
    # Additionally split deltas into those below and above our reference mass
    all_lnm1_samples = full_lnm1_q_data['all_lnm1_samples']
    lnm1_deltas = full_lnm1_q_data['lnm1_deltas']
    ind_m20 = full_lnm1_q_data['ind_m20']
    lnm1_deltas_low = lnm1_deltas[:ind_m20][::-1]
    lnm1_deltas_high = lnm1_deltas[ind_m20:]

    # Similarly read out list of sorted mass ratios and deltas
    all_q_samples = full_lnm1_q_data['all_q_samples']
    q_deltas = full_lnm1_q_data['q_deltas'][::-1]

    ###################################
    # Constructing our lnm1 AR1 process
    ###################################

    # First get variance of the process
    # We will sample from a half normal distribution, but override this with a quadratic prior
    # on the processes' standard deviation; see Eq. B1
    ar_lnm1_std = numpyro.sample("ar_lnm1_std",dist.HalfNormal(1))
    numpyro.factor("ar_lnm1_std_prior",ar_lnm1_std**2/2. - (ar_lnm1_std/2.)**4/8.75)

    # Next, the autocorrelation length
    log_ar_lnm1_tau = numpyro.sample("log_ar_lnm1_tau",dist.Normal(0,0.75))
    ar_lnm1_tau = numpyro.deterministic("ar_lnm1_tau",10.**log_ar_lnm1_tau)

    # As discussed in Appendix B, we need a regularizing log-likelihood factor to help stabilize our inference; see Eq. B3
    numpyro.factor("lnm1_regularization",-(ar_lnm1_std/jnp.sqrt(ar_lnm1_tau))**2/(2.*0.5**2))

    # Sample an initial rate density at the reference mass point
    # First draw un unscaled variable from N(0,1), then rescale by the standard deviation
    ln_f_lnm1_ref_unscaled = numpyro.sample("ln_f_lnm1_ref_unscaled",dist.Normal(0,1))
    ln_f_lnm1_ref = ln_f_lnm1_ref_unscaled*ar_lnm1_std

    # Generate forward steps and join to reference value, following the procedure outlined in Appendix A
    # First generate a sequence of unnormalized steps from N(0,1), then rescale to compute weights and innovations
    lnm1_steps_forward = numpyro.sample("lnm1_steps_forward",dist.Normal(0,1),sample_shape=(lnm1_deltas_high.size,))
    lnm1_phis_forward = jnp.exp(-lnm1_deltas_high/ar_lnm1_tau)
    lnm1_ws_forward = jnp.sqrt(-jnp.expm1(-2.*lnm1_deltas_high/ar_lnm1_tau))*(ar_lnm1_std*lnm1_steps_forward)
    final,ln_f_lnm1s_high = lax.scan(build_ar1,ln_f_lnm1_ref,jnp.transpose(jnp.array([lnm1_phis_forward,lnm1_ws_forward]))) 
    ln_f_lnm1s = jnp.append(ln_f_lnm1_ref,ln_f_lnm1s_high)

    # Generate backward steps and prepend to forward steps above following an analogous procedure
    lnm1_steps_backward = numpyro.sample("lnm1_steps_backward",dist.Normal(0,1),sample_shape=(lnm1_deltas_low.size,))
    lnm1_phis_backward = jnp.exp(-lnm1_deltas_low/ar_lnm1_tau)
    lnm1_ws_backward = jnp.sqrt(-jnp.expm1(-2.*lnm1_deltas_low/ar_lnm1_tau))*(ar_lnm1_std*lnm1_steps_backward)
    final,ln_f_lnm1s_low = lax.scan(build_ar1,ln_f_lnm1_ref,jnp.transpose(jnp.array([lnm1_phis_backward,lnm1_ws_backward])))
    ln_f_lnm1s = jnp.append(ln_f_lnm1s_low[::-1],ln_f_lnm1s)

    # Exponentiate and save
    f_lnm1s = jnp.exp(ln_f_lnm1s)
    numpyro.deterministic("f_lnm1s",f_lnm1s)

    # Reverse sort our AR process back into an array in which injections and each event's PE samples are grouped
    f_lnm1s_eventSorted = f_lnm1s[full_lnm1_q_data['lnm1_reverseSorting']]

    ############################
    # Construct AR1 process in q
    ############################

    # Follow the same strategies to construct an AR1 process over mass ratio
    # First get the process' standard deviation
    ar_q_std = numpyro.sample("ar_q_std",dist.HalfNormal(1))
    numpyro.factor("ar_q_std_prior",ar_q_std**2/2. - (ar_q_std/2.)**4/8.75)

    # Next the autocorrelation length
    log_ar_q_tau = numpyro.sample("log_ar_q_tau",dist.Normal(-0.3,0.5))
    ar_q_tau = numpyro.deterministic("ar_q_tau",10.**log_ar_q_tau)
    numpyro.factor("q_regularization",-(ar_q_std/jnp.sqrt(ar_q_tau))**2/(2.*0.5**2))

    # Choose an initial reference value
    ln_f_q_ref_unscaled = numpyro.sample("ln_f_q_ref_unscaled",dist.Normal(0,1))
    ln_f_q_ref = ln_f_q_ref_unscaled*ar_q_std

    # Generate backward steps and prepend to reference value
    q_steps_backward = numpyro.sample("q_steps_backward",dist.Normal(0,1),sample_shape=(q_deltas.size,))
    q_phis_backward = jnp.exp(-q_deltas/ar_q_tau)
    q_ws_backward = jnp.sqrt(-jnp.expm1(-2.*q_deltas/ar_q_tau))*(ar_q_std*q_steps_backward)
    final,ln_f_qs = lax.scan(build_ar1,ln_f_q_ref,jnp.transpose(jnp.array([q_phis_backward,q_ws_backward])))
    ln_f_qs = jnp.append(ln_f_qs[::-1],ln_f_q_ref)

    # Exponentiate and save
    f_qs = jnp.exp(ln_f_qs)
    numpyro.deterministic("f_qs",f_qs)
    f_qs_eventSorted = f_qs[full_lnm1_q_data['q_reverseSorting']]

    ##############################
    # Remaining degrees of freedom
    ##############################

    # Sample our hyperparameters
    # R20: Differential merger rate (dR/dlnm1) at reference m1 and z values
    # mu_chi: Mean of spin magnitude distribution; see Eq. C4
    # logsig_chi: Log10 spin magnitude standard deviation; see Eq. C4
    # sig_cost: Standard deviation of cosine spin tilts; see Eq. C5
    # kappa: Power-law index on redshift evolution of the merger rate; see Eq. C6

    # Sample the merger rate at our reference mass and redshift values
    logR20 = numpyro.sample("logR20",dist.Uniform(-6,3))
    R20 = numpyro.deterministic("R20",10.**logR20)

    # Sample our baseline hyperparameters for redshift and component spins
    # Draw some parameters directly
    mu_chi = numpyro.sample("mu_chi",dist.Uniform(0.,1.))  
    logsig_chi = numpyro.sample("logsig_chi",dist.Uniform(-1.,0.))  
    kappa = numpyro.sample("kappa",dist.Normal(0,5))

    # Some parameters have posteriors that encounter their prior boundaries.
    # In this case it is easier to sample in logit space over the (-inf,inf) interval,
    # then transform back to the actual parameter of interest.

    # First draw a logit quantity from a normal distribution, then override this normal
    # prior to impose a uniform prior on the physical parameter of interest
    logit_sig_cost = numpyro.sample("logit_sig_cost",dist.Normal(0,logit_std))  
    sig_cost,jac_sig_cost = get_value_from_logit(logit_sig_cost,0.3,2.)
    numpyro.deterministic("sig_cost",sig_cost)
    numpyro.factor("p_sig_cost",logit_sig_cost**2/(2.*logit_std**2)-jnp.log(jac_sig_cost))

    # Fixed params: mean of cosine spin tilts; Eq. C5
    mu_cost = 1.

    # Compute normalization factors necessary to ensure that `R20` is correctly defined as the
    # merger rate at the desired reference mass and redshift values
    f_z_norm = (1+0.2)**kappa

    # Entropy penalization
    #p_lnm1 = f_lnm1s/jnp.trapz(f_lnm1s,all_lnm1_samples)
    #p_q = f_qs/jnp.trapz(f_qs,all_q_samples)
    #S_lnm1 = -jnp.trapz(p_lnm1*jnp.log(p_lnm1),all_lnm1_samples)
    #S_q = -jnp.trapz(p_q*jnp.log(p_q),all_q_samples)
    #numpyro.factor("entropy",S_lnm1+S_q)

    ###############################
    # Expected number of detections
    ###############################

    # Read out found injections and draw probabilities
    m1_det = injectionDict['m1']
    a1_det = injectionDict['a1']
    a2_det = injectionDict['a2']
    cost1_det = injectionDict['cost1']
    cost2_det = injectionDict['cost2']
    z_det = injectionDict['z']
    dVdz_det = injectionDict['dVdz']
    p_draw = injectionDict['p_draw_m1m2z']*injectionDict['p_draw_a1a2cost1cost2']

    # Compute proposed population weights
    # Note that injection draw weights are defined as dP/dm1*dm2, rather than dP/dlnm1*dq (as we are using here)
    # This requires an additional factor of 1/m1 on both f_m1_det and f_m2_det below
    # Additionally, draw weights are defined as probability densities over redshift and detector frame time,
    # necessitating us to multiply by dVdz*(1.+z)**(-1) to convert from a source-frame rate density
    f_m1_det = f_lnm1s_eventSorted[full_lnm1_q_data['injections_from_allSamples']]/m1_det
    f_m2_det = f_qs_eventSorted[full_lnm1_q_data['injections_from_allSamples']]/m1_det
    f_z_det = dVdz_det*(1.+z_det)**(kappa-1.)/f_z_norm
    p_a1_det = truncatedNormal(a1_det,mu_chi,10.**logsig_chi,0,1)
    p_a2_det = truncatedNormal(a2_det,mu_chi,10.**logsig_chi,0,1)
    p_cost1_det = truncatedNormal(cost1_det,mu_cost,sig_cost,-1,1)
    p_cost2_det = truncatedNormal(cost2_det,mu_cost,sig_cost,-1,1)

    # All together, the quantity below is the detection rate dN/dm1*dm2*da1*da2*dcost1*dcost2*dz*dt_det
    R_pop_det = R20*f_m1_det*f_m2_det*f_z_det*p_a1_det*p_a2_det*p_cost1_det*p_cost2_det

    # Form ratio of proposed weights over draw weights
    # The division by 2 corresponds to the fact that injections are uniformly placed over the 2 year observation period
    inj_weights = R_pop_det/(p_draw/2.)
    
    # As a fit diagnostic, compute effective number of injections.
    nEff_inj = jnp.sum(inj_weights)**2/jnp.sum(inj_weights**2)
    nObs = 1.0*len(sampleDict)
    numpyro.deterministic("nEff_inj_per_event",nEff_inj/nObs)

    # Compute total expected number of detections and add to log-likelihood
    Nexp = jnp.sum(inj_weights)/injectionDict['nTrials']
    numpyro.factor("rate",-Nexp)

    ###############################
    # Compute per-event likelihoods
    ###############################

    # This function defines the per-event log-likelihood. It expects the following arguments:
    # m1_sample, z_sample...: Arrays of posterior samples for the given event
    # priors: Corresponding array of prior probabilities assigned to each sample
    # ar_indices: Indices used to retrieve the correct AR1 rates corresponding to this event's samples
    def logp(m1_sample,z_sample,dVdz_sample,a1_sample,a2_sample,cost1_sample,cost2_sample,priors,ar_indices):

        # Compute proposed population weights, analogous to calculation for injections done above
        # Use `ar_indices` to extract the correct values of `f_lnm1s_eventSorted` and `f_qs_eventSorted`
        # corresponding to each of this event's posterior samples
        f_m1 = f_lnm1s_eventSorted[ar_indices]/m1_sample
        f_m2 = f_qs_eventSorted[ar_indices]/m1_sample
        f_z = dVdz_sample*(1.+z_sample)**(kappa-1.)/f_z_norm
        p_a1 = truncatedNormal(a1_sample,mu_chi,10.**logsig_chi,0,1)
        p_a2 = truncatedNormal(a2_sample,mu_chi,10.**logsig_chi,0,1)
        p_cost1 = truncatedNormal(cost1_sample,mu_cost,sig_cost,-1,1)
        p_cost2 = truncatedNormal(cost2_sample,mu_cost,sig_cost,-1,1)

        # All together, the quantity below is the detection rate dN/dm1*dm2*da1*da2*dcost1*dcost2*dz*dt_det
        R_pop = R20*f_m1*f_m2*f_z*p_a1*p_a2*p_cost1*p_cost2

        # Form ratio of proposed population weights to PE priors
        mc_weights = R_pop/priors

        # Compute effective number of samples and return log-likelihood
        n_eff = jnp.sum(mc_weights)**2/jnp.sum(mc_weights**2)     
        return jnp.log(jnp.mean(mc_weights)),n_eff
    
    # Map the log-likelihood function over each event in our catalog
    log_ps,n_effs = vmap(logp)(
                        jnp.array([sampleDict[k]['m1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['z'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['dVc_dz'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['a1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['a2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['cost1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['cost2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['z_prior'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['ar_indices'] for k in sampleDict]))
        
    # As a diagnostic, save minimum number of effective samples across all events
    numpyro.deterministic('min_log_neff',jnp.min(jnp.log10(n_effs)))

    # Tally log-likelihoods across our catalog
    numpyro.factor("logp",jnp.sum(log_ps))

def ar_lnm_q(sampleDict,injectionDict,full_lnm_q_data):

    """
    Likelihood model in which the BBH ln(m) and q distributions are described as AR(1) processes, for use within `numpyro`.
    Specifically, the merger rate is assumed to take the form dR/dlnm1*dlnm2 \propto f(lnm1)*f(lnm2)*g(q).
    The distributions redshifts and spins are simultaneously fit.
    The BBH merger rate is assumed to grow as a power law in `(1+z)`.
    Spin magnitudes are described via a truncated normal distribution, and spin cosine tilts are modeled as another
    truncated normal, whose mean is fixed to `cos(theta)=1`.

    Parameters
    ----------
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog, as prepared by `getData.getSamples`
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections, as prepared by `getData.getInjections`
    full_lnm_q_data : dict
        Precomputed dictionary containing accumulated list of *all* `lnm` samples and `q`, among both event posteriors and injections.
        Dictionary also contains information needed to sort list into numerical order, or into event-based order.
    """

    # Read complete list of sorted lnm1 samples and deltas between them
    # Additionally split deltas into those below and above our reference mass
    all_lnm_samples = full_lnm_q_data['lnm_allSamples']
    ind_m20 = full_lnm_q_data['ind_m20']
    lnm_deltas = full_lnm_q_data['lnm_deltas']
    lnm_deltas_low = lnm_deltas[:ind_m20][::-1]
    lnm_deltas_high = lnm_deltas[ind_m20:]

    # Similarly read out list of sorted mass ratios and deltas
    all_q_samples =  full_lnm_q_data['q_allSamples']
    q_deltas = full_lnm_q_data['q_deltas'][::-1]

    ###################################
    # Constructing our lnm AR1 process
    ###################################

    # First get variance of the process
    # We will sample from a half normal distribution, but override this with a quadratic prior
    # on the processes' standard deviation; see Eq. B1
    ar_lnm_std = numpyro.sample("ar_lnm_std",dist.HalfNormal(1))
    numpyro.factor("ar_lnm_std_prior",ar_lnm_std**2/2. - (ar_lnm_std/1.177)**4)

    # Next the autocorrelation length
    log_ar_lnm_tau = numpyro.sample("log_ar_lnm_tau",dist.Normal(0,0.75))
    ar_lnm_tau = numpyro.deterministic("ar_lnm_tau",10.**log_ar_lnm_tau)

    # As discussed in Appendix B, we need a regularizing log-likelihood factor to help stabilize our inference; see Eq. B3
    numpyro.factor("lnm_regularization",-(ar_lnm_std/jnp.sqrt(ar_lnm_tau))**2/(2.*0.5**2))

    # Sample an initial rate density at reference point
    # First draw un unscaled variable from N(0,1), then rescale by the standard deviation
    ln_f_lnm_ref_unscaled = numpyro.sample("ln_f_lnm_ref_unscaled",dist.Normal(0,1))
    ln_f_lnm_ref = ln_f_lnm_ref_unscaled*ar_lnm_std

    # Generate forward steps and join to reference value, following the procedure outlined in Appendix A
    # First generate a sequence of unnormalized steps from N(0,1), then rescale to compute weights and innovations
    lnm_steps_forward = numpyro.sample("lnm_steps_forward",dist.Normal(0,1),sample_shape=(lnm_deltas_high.size,))
    lnm_phis_forward = jnp.exp(-lnm_deltas_high/ar_lnm_tau)
    lnm_ws_forward = jnp.sqrt(1.-jnp.exp(-2.*lnm_deltas_high/ar_lnm_tau))*ar_lnm_std*lnm_steps_forward
    final,ln_f_lnms_high = lax.scan(build_ar1,ln_f_lnm_ref,jnp.transpose(jnp.array([lnm_phis_forward,lnm_ws_forward]))) 
    ln_f_lnms = jnp.append(ln_f_lnm_ref,ln_f_lnms_high)

    # Generate backward steps and prepend to forward steps above following an analogous procedure
    lnm_steps_backward = numpyro.sample("lnm_steps_backward",dist.Normal(0,1),sample_shape=(lnm_deltas_low.size,))
    lnm_phis_backward = jnp.exp(-lnm_deltas_low/ar_lnm_tau)
    lnm_ws_backward = jnp.sqrt(1.-jnp.exp(-2.*lnm_deltas_low/ar_lnm_tau))*ar_lnm_std*lnm_steps_backward
    final,ln_f_lnms_low = lax.scan(build_ar1,ln_f_lnm_ref,jnp.transpose(jnp.array([lnm_phis_backward,lnm_ws_backward])))
    ln_f_lnms = jnp.append(ln_f_lnms_low[::-1],ln_f_lnms)

    # Exponentiate and save
    f_lnms = jnp.exp(ln_f_lnms)
    numpyro.deterministic("f_lnms",f_lnms)

    # Reverse sort our AR process back into an array in which injections and each event's PE samples are grouped
    f_lnms_eventSorted = f_lnms[full_lnm_q_data['lnm_reverseSorting']]

    #######################################
    # Next, build AR1 process in mass ratio
    #######################################

    # Follow the same strategies to construct an AR1 process over mass ratio
    # First get the process' standard deviation
    ar_q_std = numpyro.sample("ar_q_std",dist.HalfNormal(1))
    numpyro.factor("ar_q_std_prior",ar_q_std**2/2. - (ar_q_std/1.177)**4)

    # Next the autocorrelation length
    log_ar_q_tau = numpyro.sample("log_ar_q_tau",dist.Normal(0,0.75))
    ar_q_tau = numpyro.deterministic("ar_q_tau",10.**log_ar_q_tau)
    numpyro.factor("q_regularization",-(ar_q_std/jnp.sqrt(ar_q_tau))**2/(2.*0.5**2))

    # Sample an initial rate density at reference point
    ln_f_q_ref_unscaled = numpyro.sample("ln_f_q_ref_unscaled",dist.Normal(0,1))
    ln_f_q_ref = ln_f_q_ref_unscaled*ar_q_std

    # Generate backward steps
    q_steps_backward = numpyro.sample("q_steps_backward",dist.Normal(0,1),sample_shape=(q_deltas.size,))
    q_phis_backward = jnp.exp(-q_deltas/ar_q_tau)
    q_ws_backward = jnp.sqrt(1.-jnp.exp(-2.*q_deltas/ar_q_tau))*ar_q_std*q_steps_backward
    final,ln_f_qs_low = lax.scan(build_ar1,ln_f_q_ref,jnp.transpose(jnp.array([q_phis_backward,q_ws_backward])))
    ln_f_qs = jnp.append(ln_f_qs_low[::-1],ln_f_q_ref)

    # Exponentiate and save
    f_qs = jnp.exp(ln_f_qs)
    numpyro.deterministic("f_qs",f_qs)
    f_qs_eventSorted = f_qs[full_lnm_q_data['q_reverseSorting']]

    ####################################################################################
    # Sample our baseline hyperparameters for mass ratio, redshift, and component spins
    ####################################################################################

    # Sample our hyperparameters
    # R20: Differential merger rate (dR/dlnm1) at reference m1 and z values
    # mu_chi: Mean of spin magnitude distribution; see Eq. C4
    # logsig_chi: Log10 spin magnitude standard deviation; see Eq. C4
    # sig_cost: Standard deviation of cosine spin tilts; see Eq. C5
    # kappa: Power-law index on redshift evolution of the merger rate; see Eq. C6

    # Sample the merger rate at our reference mass and redshift values
    logR20 = numpyro.sample("logR20",dist.Uniform(-6,3))
    R20 = numpyro.deterministic("R20",10.**logR20)

    # Sample our baseline hyperparameters for redshift and component spins
    mu_chi = numpyro.sample("mu_chi",dist.Uniform(0.,1.))  
    kappa = numpyro.sample("kappa",dist.Normal(0,5))  
    logsig_chi = numpyro.sample("logsig_chi",dist.Uniform(-1.,0.)) 

    # Some parameters have posteriors that encounter their prior boundaries.
    # In this case it is easier to sample in logit space over the (-inf,inf) interval,
    # then transform back to the actual parameter of interest.

    # First draw a logit quantity from a normal distribution, then override this normal
    # prior to impose a uniform prior on the physical parameter of interest
    logit_sig_cost = numpyro.sample("logit_sig_cost",dist.Normal(0,logit_std))  # Standard deviation of cosine spin tilts; Eq. C5
    sig_cost,jac_sig_cost = get_value_from_logit(logit_sig_cost,0.3,2.)
    numpyro.deterministic("sig_cost",sig_cost)
    numpyro.factor("p_sig_cost",logit_sig_cost**2/(2.*logit_std**2)-jnp.log(jac_sig_cost))

    # Fixed params: mean of cosine spin tilts; Eq. C5
    mu_cost = 1.

    # Compute normalization factors necessary to ensure that `R20` is correctly defined as the
    # merger rate at the desired reference mass and redshift values
    f_z_norm = (1+0.2)**kappa

    # Read out found injections
    a1_det = injectionDict['a1']
    a2_det = injectionDict['a2']
    cost1_det = injectionDict['cost1']
    cost2_det = injectionDict['cost2']
    m1_det = injectionDict['m1']
    m2_det = injectionDict['m2']
    z_det = injectionDict['z']
    dVdz_det = injectionDict['dVdz']
    p_draw = injectionDict['p_draw_m1m2z']*injectionDict['p_draw_a1a2cost1cost2']

    # Compute proposed population weights
    # Note that injection draw weights are defined as dP/dm1*dm2, rather than dP/dlnm1*dq (as we are using here)
    # This requires an additional factor of 1/m1 on both f_m1_det and f_m2_det below
    f_m1_det = f_lnms_eventSorted[full_lnm_q_data['m1_injections_from_allSamples']]/m1_det
    f_m2_det = f_lnms_eventSorted[full_lnm_q_data['m2_injections_from_allSamples']]/m2_det
    f_q_det = f_qs_eventSorted[full_lnm_q_data['q_injections_from_allSamples']]
    f_z_det = dVdz_det*(1.+z_det)**(kappa-1.)/f_z_norm
    p_a1_det = truncatedNormal(a1_det,mu_chi,10.**logsig_chi,0,1)
    p_a2_det = truncatedNormal(a2_det,mu_chi,10.**logsig_chi,0,1)
    p_cost1_det = truncatedNormal(cost1_det,mu_cost,sig_cost,-1,1)
    p_cost2_det = truncatedNormal(cost2_det,mu_cost,sig_cost,-1,1)

    # All together, the quantity below is the detection rate dN/dm1*dm2*da1*da2*dcost1*dcost2*dz*dt_det
    R_pop_det = R20*f_m1_det*f_m2_det*f_q_det*f_z_det*p_a1_det*p_a2_det*p_cost1_det*p_cost2_det

    # Form ratio of proposed weights over draw weights
    # The division by 2 corresponds to the fact that injections are uniformly placed over the 2 year observation period
    inj_weights = R_pop_det/(p_draw/2.)
    
    # As a fit diagnostic, compute effective number of injections
    nEff_inj = jnp.sum(inj_weights)**2/jnp.sum(inj_weights**2)
    nObs = 1.0*len(sampleDict)
    numpyro.deterministic("nEff_inj_per_event",nEff_inj/nObs)

    # Compute net detection efficiency and add to log-likelihood
    Nexp = jnp.sum(inj_weights)/injectionDict['nTrials']
    numpyro.factor("rate",-Nexp)

    # This function defines the per-event log-likelihood. It expects the following arguments:
    # m1_sample, z_sample...: Arrays of posterior samples for the given event
    # priors: Corresponding array of prior probabilities assigned to each sample
    # ar_indices: Indices used to retrieve the correct AR1 rates corresponding to this event's samples
    def logp(m1_sample,m2_sample,z_sample,dVdz_sample,a1_sample,a2_sample,cost1_sample,cost2_sample,priors,m1_ar_indices,m2_ar_indices,q_ar_indices):

        # Compute proposed population weights
        # Use `ar_indices` to extract the correct values of `f_lnm1s_eventSorted` and `f_qs_eventSorted`
        # correspond to each of this event's posterior samples
        f_m1 = f_lnms_eventSorted[m1_ar_indices]/m1_sample
        f_m2 = f_lnms_eventSorted[m2_ar_indices]/m2_sample
        f_q = f_qs_eventSorted[q_ar_indices]
        f_z = dVdz_sample*(1.+z_sample)**(kappa-1.)/f_z_norm
        p_a1 = truncatedNormal(a1_sample,mu_chi,10.**logsig_chi,0,1)
        p_a2 = truncatedNormal(a2_sample,mu_chi,10.**logsig_chi,0,1)
        p_cost1 = truncatedNormal(cost1_sample,mu_cost,sig_cost,-1,1)
        p_cost2 = truncatedNormal(cost2_sample,mu_cost,sig_cost,-1,1)
        R_pop = R20*f_m1*f_m2*f_q*f_z*p_a1*p_a2*p_cost1*p_cost2

        # Form ratio of proposed population weights to PE priors
        mc_weights = R_pop/priors

        # Compute effective number of samples and return log-likelihood
        n_eff = jnp.sum(mc_weights)**2/jnp.sum(mc_weights**2)     
        return jnp.log(jnp.mean(mc_weights)),n_eff
    
    # Map the log-likelihood function over each event in our catalog
    log_ps,n_effs = vmap(logp)(
                        jnp.array([sampleDict[k]['m1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['m2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['z'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['dVc_dz'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['a1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['a2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['cost1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['cost2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['z_prior'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['m1_ar_indices'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['m2_ar_indices'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['q_ar_indices'] for k in sampleDict]))
        
    # As a diagnostic, save minimum number of effective samples across all events
    numpyro.deterministic('min_log_neff',jnp.min(jnp.log10(n_effs)))

    # Tally log-likelihoods across our catalog
    numpyro.factor("logp",jnp.sum(log_ps))
