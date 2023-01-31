import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from jax import vmap,lax
import numpy as np
from utilities import *

def ar_spinMagTilt(sampleDict,injectionDict,full_chi_data):

    """
    Likelihood model in which the BBH spin magnitude and cosine tilt distributions are described as AR(1) processes, for use within `numpyro`.
    The distributions redshifts and masses are simultaneously fit.
    The BBH merger rate is assumed to grow as a power law in `(1+z)`.
    Meanwhile, primary masses are assumed to follow a mixture between a power-law and Gaussian peak, while mass ratios are distributed as a power law.

    Parameters
    ----------
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog, as prepared by `getData.getSamples`
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections, as prepared by `getData.getInjections`
    full_chi_data : dict
        Precomputed dictionary containing accumulated list of *all* component spin magnitude and tilts, among both event posteriors and injections.
        Dictionary also contains information needed to sort list into numerical order, or into event-based order.
        Prepared in `run_ar_chi_cost.py`.
    """

    # Read complete list of sorted chi samples and deltas between them
    # Additionally split deltas into those below and above our reference spin magnitude
    all_chi_samples = full_chi_data['chi_allSamples']
    ind_a01 = full_chi_data['ind_a01']
    chi_deltas = full_chi_data['chi_deltas']
    chi_deltas_low = chi_deltas[:ind_a01][::-1]
    chi_deltas_high = chi_deltas[ind_a01:]

    # Similarly read out list of sorted mass ratios and deltas
    all_cost_samples = full_chi_data['cost_allSamples']
    cost_deltas = full_chi_data['cost_deltas'][::-1]

    ###################################
    # Constructing our chi AR1 process
    ###################################

    # First get variance of the process
    # We will sample from a half normal distribution, but override this with a quadratic prior
    # on the processes' standard deviation; see Eq. B1
    ar_chi_std = numpyro.sample("ar_chi_std",dist.HalfNormal(1.))
    numpyro.factor("ar_chi_std_prior",ar_chi_std**2/2. - (ar_chi_std/0.75)**4/8.75)

    # Next the autocorrelation length
    # Since the posterior for this parameter runs up against prior boundaries, sample in logit space
    logit_ar_chi_tau = numpyro.sample("logit_ar_chi_tau",dist.Normal(0,logit_std))
    ar_chi_tau,jac_ar_chi_tau = get_value_from_logit(logit_ar_chi_tau,0.2,2.)
    numpyro.factor("p_ar_chi_tau",logit_ar_chi_tau**2/(2.*logit_std**2)-jnp.log(jac_ar_chi_tau))
    numpyro.deterministic("ar_chi_tau",ar_chi_tau)

    # As discussed in Appendix B, we need a regularizing log-likelihood factor to help stabilize our inference; see Eq. B3
    numpyro.factor("chi_regularization",-(ar_chi_std/jnp.sqrt(ar_chi_tau))**2/(2.*0.4**2))

    # Sample an initial rate density at reference point
    ln_f_chi_ref_unscaled = numpyro.sample("ln_f_chi_ref_unscaled",dist.Normal(0,1))
    ln_f_chi_ref = ln_f_chi_ref_unscaled*ar_chi_std

    # Generate forward steps
    # First generate a sequence of unnormalized steps from N(0,1), then rescale to compute weights and innovations
    chi_steps_forward = numpyro.sample("chi_steps_forward",dist.Normal(0,1),sample_shape=(chi_deltas_high.size,))
    chi_phis_forward = jnp.exp(-chi_deltas_high/ar_chi_tau)
    chi_ws_forward = jnp.sqrt(1.-jnp.exp(-2.*chi_deltas_high/ar_chi_tau))*(ar_chi_std*chi_steps_forward)
    final,ln_f_chis_high = lax.scan(build_ar1,ln_f_chi_ref,jnp.transpose(jnp.array([chi_phis_forward,chi_ws_forward]))) 
    ln_f_chis = jnp.append(ln_f_chi_ref,ln_f_chis_high)

    # Generate backward steps and prepend to forward steps above following an analogous procedure
    chi_steps_backward = numpyro.sample("chi_steps_backward",dist.Normal(0,1),sample_shape=(chi_deltas_low.size,))
    chi_phis_backward = jnp.exp(-chi_deltas_low/ar_chi_tau)
    chi_ws_backward = jnp.sqrt(1.-jnp.exp(-2.*chi_deltas_low/ar_chi_tau))*(ar_chi_std*chi_steps_backward)
    final,ln_f_chis_low = lax.scan(build_ar1,ln_f_chi_ref,jnp.transpose(jnp.array([chi_phis_backward,chi_ws_backward])))
    ln_f_chis = jnp.append(ln_f_chis_low[::-1],ln_f_chis)

    # Exponentiate and save
    f_chis = jnp.exp(ln_f_chis)
    numpyro.deterministic("f_chis",f_chis)

    # Reverse sort our AR process back into an array in which injections and each event's PE samples are grouped
    f_chi_eventSorted = f_chis[full_chi_data['chi_reverseSorting']]

    ################################
    # Construct AR1 process in cost
    ################################

    # Follow the same strategies to construct an AR1 process over cost
    # First get the process' standard deviation
    ar_cost_std = numpyro.sample("ar_cost_std",dist.HalfNormal(1.))
    numpyro.factor("ar_cost_std_prior",ar_cost_std**2/2. - (ar_cost_std/0.75)**4/8.75)

    # Next the autocorrelation length
    logit_ar_cost_tau = numpyro.sample("logit_ar_cost_tau",dist.Normal(0,logit_std))
    ar_cost_tau,jac_ar_cost_tau = get_value_from_logit(logit_ar_cost_tau,0.3,4.)
    numpyro.factor("p_ar_cost_tau",logit_ar_cost_tau**2/(2.*logit_std**2)-jnp.log(jac_ar_cost_tau))
    numpyro.deterministic("ar_cost_tau",ar_cost_tau)
    numpyro.factor("cost_regularization",-(ar_cost_std/jnp.sqrt(ar_cost_tau))**2/(2.*0.4**2))

    # Choose an initial reference value
    ln_f_cost_ref_unscaled = numpyro.sample("ln_f_cost_ref_unscaled",dist.Normal(0,1))
    ln_f_cost_ref = ln_f_cost_ref_unscaled*ar_cost_std

    # Generate backward steps and prepend to reference value
    cost_steps_backward = numpyro.sample("cost_steps_backward",dist.Normal(0,1),sample_shape=(cost_deltas.size,))
    cost_phis_backward = jnp.exp(-cost_deltas/ar_cost_tau)
    cost_ws_backward = jnp.sqrt(-jnp.expm1(-2.*cost_deltas/ar_cost_tau))*(ar_cost_std*cost_steps_backward)
    final,ln_f_costs = lax.scan(build_ar1,ln_f_cost_ref,jnp.transpose(jnp.array([cost_phis_backward,cost_ws_backward])))
    ln_f_costs = jnp.append(ln_f_costs[::-1],ln_f_cost_ref)

    # Exponentiate and save
    f_cost = jnp.exp(ln_f_costs)
    numpyro.deterministic("f_cost",f_cost)
    f_cost_eventSorted = f_cost[full_chi_data['cost_reverseSorting']]

    ##############################
    # Remaining degrees of freedom
    ##############################
    
    # Sample our hyperparameters
    # R20: Differential merger rate (dR/dlnm1) at reference m1, z, and spin values
    # alpha: Power-law index on primary mass distribution
    # mu_m1: Location of gaussian peak in primary mass distribution
    # sig_m1: Width of gaussian peak
    # f_peak: Fraction of events comprising gaussian peak
    # mMax: Location at which BBH mass distribution tapers off
    # mMin: Lower boundary at which BBH mass distribution tapers off
    # dmMax: Taper width above maximum mass
    # dmMin: Taper width below minimum mass
    # bq: Power-law index on the conditional secondary mass distribution p(m2|m1)
    # kappa: Power-law index on redshift evolution of the merger rate; see Eq. C6

    # Sample the merger rate at our reference mass and redshift values
    logR20 = numpyro.sample("logR20",dist.Uniform(-6,3))
    R20 = numpyro.deterministic("R20",10.**logR20)

    # Draw some parameters directly
    alpha = numpyro.sample("alpha",dist.Normal(0,6))
    mu_m1 = numpyro.sample("mu_m1",dist.Uniform(20,50))
    mMin = numpyro.sample("mMin",dist.Uniform(5,15))
    bq = numpyro.sample("bq",dist.Normal(0,4))
    kappa = numpyro.sample("kappa",dist.Normal(0,5))

    # Some parameters have posteriors that encounter their prior boundaries.
    # In this case it is easier to sample in logit space over the (-inf,inf) interval,
    # then transform back to the actual parameter of interest.

    # First draw logit quantities on the unbounded (-inf,inf)  interval
    logit_sig_m1 = numpyro.sample("logit_sig_m1",dist.Normal(0,logit_std))
    logit_log_f_peak = numpyro.sample("logit_log_f_peak",dist.Normal(0,logit_std))
    logit_mMax = numpyro.sample("logit_mMax",dist.Normal(0,logit_std))
    logit_log_dmMin = numpyro.sample("logit_log_dmMin",dist.Normal(0,logit_std))
    logit_log_dmMax = numpyro.sample("logit_log_dmMax",dist.Normal(0,logit_std))

    # Inverse transform back to the physical parameters of interest
    sig_m1,jac_sig_m1 = get_value_from_logit(logit_sig_m1,3.,15.)
    log_f_peak,jac_log_f_peak = get_value_from_logit(logit_log_f_peak,-3,0.)
    mMax,jac_mMax = get_value_from_logit(logit_mMax,50.,100.)
    log_dmMin,jac_log_dmMin = get_value_from_logit(logit_log_dmMin,-1,1)
    log_dmMax,jac_log_dmMax = get_value_from_logit(logit_log_dmMax,0.5,1.5)

    numpyro.deterministic("sig_m1",sig_m1)
    numpyro.deterministic("log_f_peak",log_f_peak)
    numpyro.deterministic("mMax",mMax)
    numpyro.deterministic("log_dmMin",log_dmMin)
    numpyro.deterministic("log_dmMax",log_dmMax)

    # Override prior factors of logit quantities, and impose a uniform prior in the physical space
    numpyro.factor("p_sig_m1",logit_sig_m1**2/(2.*logit_std**2)-jnp.log(jac_sig_m1))
    numpyro.factor("p_log_f_peak",logit_log_f_peak**2/(2.*logit_std**2)-jnp.log(jac_log_f_peak))
    numpyro.factor("p_mMax",logit_mMax**2/(2.*logit_std**2)-jnp.log(jac_mMax))
    numpyro.factor("p_log_dmMin",logit_log_dmMin**2/(2.*logit_std**2)-jnp.log(jac_log_dmMin))
    numpyro.factor("p_log_dmMax",logit_log_dmMax**2/(2.*logit_std**2)-jnp.log(jac_log_dmMax))

    # Compute normalization factors necessary to ensure that `R20` is correctly defined as the
    # merger rate at the desired reference mass and redshift values
    f_m1_norm = massModel(20.,alpha,mu_m1,sig_m1,10.**log_f_peak,mMax,mMin,10.**log_dmMax,10.**log_dmMin)
    f_z_norm = (1.+0.2)**kappa

    # Read out found injections
    m1_det = injectionDict['m1']
    m2_det = injectionDict['m2']
    z_det = injectionDict['z']
    dVdz_det = injectionDict['dVdz']
    p_draw = injectionDict['p_draw_m1m2z']*injectionDict['p_draw_a1a2cost1cost2']

    # Compute proposed population weights
    # Note that draw weights are defined as a probability density over redshift and detector frame time
    # We therefore need to multiply by dVdz*(1+z)**(-1) to convert from a source-frame merger rate density
    f_m1_det = massModel(m1_det,alpha,mu_m1,sig_m1,10.**log_f_peak,mMax,mMin,10.**log_dmMax,10.**log_dmMin)/f_m1_norm
    p_m2_det = (1.+bq)*m2_det**bq/(m1_det**(1.+bq)-tmp_min**(1.+bq))
    f_a1_det = f_chi_eventSorted[full_chi_data['a1_injections_from_allSamples']]
    f_a2_det = f_chi_eventSorted[full_chi_data['a2_injections_from_allSamples']]
    f_cost1_det = f_cost_eventSorted[full_chi_data['a1_injections_from_allSamples']]
    f_cost2_det = f_cost_eventSorted[full_chi_data['a2_injections_from_allSamples']]
    f_z_det = dVdz_det*(1.+z_det)**(kappa-1.)/f_z_norm 

    # All together, the quantity below is the detection rate dN/dm1*dm2*da1*da2*dcost1*dcost2*dz*dt_det
    R_pop_det = R20*f_m1_det*p_m2_det*f_z_det*f_a1_det*f_a2_det*f_cost1_det*f_cost2_det

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

    ###############################
    # Compute per-event likelihoods
    ###############################
    
    # This function defines the per-event log-likelihood. It expects the following arguments:
    # m1_sample, z_sample...: Arrays of posterior samples for the given event
    # priors: Corresponding array of prior probabilities assigned to each sample
    # ar_indices: Indices used to retrieve the correct AR1 rates corresponding to this event's samples
    def logp(m1_sample,m2_sample,z_sample,dVdz_sample,priors,a1_ar_indices,a2_ar_indices):

        # Compute proposed population weights, analogous to calculation for injections done above
        # Use `ar_indices` to extract the correct values of `f_chi_eventSorted` and `f_cost_eventSorted`
        # corresponding to each of this event's posterior samples
        f_m1 = massModel(m1_sample,alpha,mu_m1,sig_m1,10.**log_f_peak,mMax,mMin,10.**log_dmMax,10.**log_dmMin)/f_m1_norm
        p_m2 = (1.+bq)*m2_sample**bq/(m1_sample**(1.+bq)-tmp_min**(1.+bq))
        f_a1 = f_chi_eventSorted[a1_ar_indices]
        f_a2 = f_chi_eventSorted[a2_ar_indices]
        f_cost1 = f_cost_eventSorted[a1_ar_indices]
        f_cost2 = f_cost_eventSorted[a2_ar_indices]
        f_z = dVdz_sample*(1.+z_sample)**(kappa-1.)/f_z_norm

        # All together, the quantity below is the detection rate dN/dm1*dm2*da1*da2*dcost1*dcost2*dz*dt_det
        R_pop = R20*f_m1*p_m2*f_z*f_a1*f_a2*f_cost1*f_cost2

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
                        jnp.array([sampleDict[k]['z_prior'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['a1_ar_indices'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['a2_ar_indices'] for k in sampleDict]))
        
    # As a diagnostic, save minimum number of effective samples across all events
    numpyro.deterministic('min_log_neff',jnp.min(jnp.log10(n_effs)))

    # Tally log-likelihoods across our catalog
    numpyro.factor("logp",jnp.sum(log_ps))

def ar_Xeff_Xp(sampleDict,injectionDict,full_chi_data):

    """
    Likelihood model in which the BBH Xeff and Xp distributions are described as AR(1) processes, for use within `numpyro`.
    The distributions redshifts and masses are simultaneously fit.
    The BBH merger rate is assumed to grow as a power law in `(1+z)`.
    Meanwhile, primary masses are assumed to follow a mixture between a power-law and Gaussian peak, while mass ratios are distributed as a power law.

    Parameters
    ----------
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog, as prepared by `getData.getSamples`
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections, as prepared by `getData.getInjections`
    full_lnm1_q_data : dict
        Precomputed dictionary containing accumulated list of *all* Xeff and Xp samples, among both event posteriors and injections.
        Dictionary also contains information needed to sort list into numerical order, or into event-based order.
        Prepared in `run_ar_Xeff_Xp.py`.
    """

    # Read complete list of sorted Xeff samples and deltas between them
    # Additionally split deltas into those below and above our reference mass
    all_Xeff_samples = full_chi_data['Xeff_allSamples']
    ind_Xeff0 = full_chi_data['ind_Xeff0']
    Xeff_deltas = full_chi_data['Xeff_deltas']
    Xeff_deltas_low = Xeff_deltas[:ind_Xeff0][::-1]
    Xeff_deltas_high = Xeff_deltas[ind_Xeff0:]

    # Similarly read out list of sorted Xp values and deltas
    all_Xp_samples = full_chi_data['Xp_allSamples']
    ind_Xp02 = full_chi_data['ind_Xp02']
    Xp_deltas = full_chi_data['Xp_deltas']
    Xp_deltas_low = Xp_deltas[:ind_Xp02][::-1]
    Xp_deltas_high = Xp_deltas[ind_Xp02:]

    ###################################
    # Constructing our Xeff AR1 process
    ###################################

    # First get variance of the process
    # We will sample from a half normal distribution, but override this with a quadratic prior
    # on the processes' standard deviation; see Eq. B1
    ar_Xeff_std = numpyro.sample("ar_Xeff_std",dist.HalfNormal(1.))
    numpyro.factor("ar_Xeff_std_prior",ar_Xeff_std**2/2. - (ar_Xeff_std/1.75)**4/8.75)

    # Next the autocorrelation length
    # Since the posterior for this parameter runs up against prior boundaries, sample in logit space
    logit_ar_Xeff_tau = numpyro.sample("logit_ar_Xeff_tau",dist.Normal(0,logit_std))
    ar_Xeff_tau,jac_ar_Xeff_tau = get_value_from_logit(logit_ar_Xeff_tau,0.2,2.)
    numpyro.factor("p_ar_Xeff_tau",logit_ar_Xeff_tau**2/(2.*logit_std**2)-jnp.log(jac_ar_Xeff_tau))
    numpyro.deterministic("ar_Xeff_tau",ar_Xeff_tau)

    # As discussed in Appendix B, we need a regularizing log-likelihood factor to help stabilize our inference; see Eq. B3
    numpyro.factor("Xeff_regularization",-(ar_Xeff_std/jnp.sqrt(ar_Xeff_tau))**2/(2.*0.4**2))

    # Sample an initial rate density at reference point
    ln_f_Xeff_ref_unscaled = numpyro.sample("ln_f_Xeff_ref_unscaled",dist.Normal(0,1))
    ln_f_Xeff_ref = ln_f_Xeff_ref_unscaled*ar_Xeff_std

    # Generate forward steps and join to reference value, following the procedure outlined in Appendix A
    # First generate a sequence of unnormalized steps from N(0,1), then rescale to compute weights and innovations
    Xeff_steps_forward = numpyro.sample("Xeff_steps_forward",dist.Normal(0,1),sample_shape=(Xeff_deltas_high.size,))
    Xeff_phis_forward = jnp.exp(-Xeff_deltas_high/ar_Xeff_tau)
    Xeff_ws_forward = jnp.sqrt(1.-jnp.exp(-2.*Xeff_deltas_high/ar_Xeff_tau))*(ar_Xeff_std*Xeff_steps_forward)
    final,ln_f_Xeffs_high = lax.scan(build_ar1,ln_f_Xeff_ref,jnp.transpose(jnp.array([Xeff_phis_forward,Xeff_ws_forward]))) 
    ln_f_Xeffs = jnp.append(ln_f_Xeff_ref,ln_f_Xeffs_high)

    # Generate backward steps and prepend to forward steps above following an analogous procedure
    Xeff_steps_backward = numpyro.sample("Xeff_steps_backward",dist.Normal(0,1),sample_shape=(Xeff_deltas_low.size,))
    Xeff_phis_backward = jnp.exp(-Xeff_deltas_low/ar_Xeff_tau)
    Xeff_ws_backward = jnp.sqrt(1.-jnp.exp(-2.*Xeff_deltas_low/ar_Xeff_tau))*(ar_Xeff_std*Xeff_steps_backward)
    final,ln_f_Xeffs_low = lax.scan(build_ar1,ln_f_Xeff_ref,jnp.transpose(jnp.array([Xeff_phis_backward,Xeff_ws_backward])))
    ln_f_Xeffs = jnp.append(ln_f_Xeffs_low[::-1],ln_f_Xeffs)

    # Exponentiate and save
    f_Xeffs = jnp.exp(ln_f_Xeffs)
    numpyro.deterministic("f_Xeffs",f_Xeffs)

    # Reverse sort our AR process back into an array in which injections and each event's PE samples are grouped
    f_Xeff_eventSorted = f_Xeffs[full_chi_data['Xeff_reverseSorting']]

    #############################
    # Construct AR1 process in Xp
    #############################

    # Follow the same strategies to construct an AR1 process over Xp
    # First get the process' standard deviation
    ar_Xp_std = numpyro.sample("ar_Xp_std",dist.HalfNormal(1.))
    numpyro.factor("ar_Xp_std_prior",ar_Xp_std**2/2. - (ar_Xp_std/1.75)**4/8.75)

    # Next the autocorrelation length
    # Since the posterior for this parameter runs up against prior boundaries, sample in logit space
    logit_ar_Xp_tau = numpyro.sample("logit_ar_Xp_tau",dist.Normal(0,logit_std))
    ar_Xp_tau,jac_ar_Xp_tau = get_value_from_logit(logit_ar_Xp_tau,0.2,2.)
    numpyro.factor("p_ar_Xp_tau",logit_ar_Xp_tau**2/(2.*logit_std**2)-jnp.log(jac_ar_Xp_tau))
    numpyro.deterministic("ar_Xp_tau",ar_Xp_tau)
    numpyro.factor("Xp_regularization",-(ar_Xp_std/jnp.sqrt(ar_Xp_tau))**2/(2.*0.4**2))

    # Sample an initial rate density at reference point
    ln_f_Xp_ref_unscaled = numpyro.sample("ln_f_Xp_ref_unscaled",dist.Normal(0,1))
    ln_f_Xp_ref = ln_f_Xp_ref_unscaled*ar_Xp_std

    # Generate forward steps
    Xp_steps_forward = numpyro.sample("Xp_steps_forward",dist.Normal(0,1),sample_shape=(Xp_deltas_high.size,))
    Xp_phis_forward = jnp.exp(-Xp_deltas_high/ar_Xp_tau)
    Xp_ws_forward = jnp.sqrt(1.-jnp.exp(-2.*Xp_deltas_high/ar_Xp_tau))*(ar_Xp_std*Xp_steps_forward)
    final,ln_f_Xps_high = lax.scan(build_ar1,ln_f_Xp_ref,jnp.transpose(jnp.array([Xp_phis_forward,Xp_ws_forward]))) 
    ln_f_Xps = jnp.append(ln_f_Xp_ref,ln_f_Xps_high)

    # Generate backward steps and prepend
    Xp_steps_backward = numpyro.sample("Xp_steps_backward",dist.Normal(0,1),sample_shape=(Xp_deltas_low.size,))
    Xp_phis_backward = jnp.exp(-Xp_deltas_low/ar_Xp_tau)
    Xp_ws_backward = jnp.sqrt(1.-jnp.exp(-2.*Xp_deltas_low/ar_Xp_tau))*(ar_Xp_std*Xp_steps_backward)
    final,ln_f_Xps_low = lax.scan(build_ar1,ln_f_Xp_ref,jnp.transpose(jnp.array([Xp_phis_backward,Xp_ws_backward])))
    ln_f_Xps = jnp.append(ln_f_Xps_low[::-1],ln_f_Xps)

    # Exponentiate and save
    f_Xps = jnp.exp(ln_f_Xps)
    numpyro.deterministic("f_Xps",f_Xps)
    f_Xp_eventSorted = f_Xps[full_chi_data['Xp_reverseSorting']]

    ##############################
    # Remaining degrees of freedom
    ##############################
    
    # Sample our hyperparameters
    # R20: Differential merger rate (dR/dlnm1) at reference m1, z, and spin values
    # alpha: Power-law index on primary mass distribution
    # mu_m1: Location of gaussian peak in primary mass distribution
    # sig_m1: Width of gaussian peak
    # f_peak: Fraction of events comprising gaussian peak
    # mMax: Location at which BBH mass distribution tapers off
    # mMin: Lower boundary at which BBH mass distribution tapers off
    # dmMax: Taper width above maximum mass
    # dmMin: Taper width below minimum mass
    # bq: Power-law index on the conditional secondary mass distribution p(m2|m1)
    # kappa: Power-law index on redshift evolution of the merger rate; see Eq. C6

    # Sample the merger rate at our reference mass and redshift values
    logR20 = numpyro.sample("logR20",dist.Uniform(-6,3))
    R20 = numpyro.deterministic("R20",10.**logR20)

    # Sample our baseline hyperparameters for masses and component spins
    # Draw some parameters directly
    alpha = numpyro.sample("alpha",dist.Normal(0,10))
    mu_m1 = numpyro.sample("mu_m1",dist.Uniform(20,50))
    mMin = numpyro.sample("mMin",dist.Uniform(5,15))
    bq = numpyro.sample("bq",dist.Normal(0,4))
    kappa = numpyro.sample("kappa",dist.Normal(0,5))

    # Some parameters have posteriors that encounter their prior boundaries.
    # In this case it is easier to sample in logit space over the (-inf,inf) interval,
    # then transform back to the actual parameter of interest.

    # First draw logit quantities on the unbounded (-inf,inf)  interval
    logit_sig_m1 = numpyro.sample("logit_sig_m1",dist.Normal(0,logit_std))
    logit_log_f_peak = numpyro.sample("logit_log_f_peak",dist.Normal(0,logit_std))
    logit_mMax = numpyro.sample("logit_mMax",dist.Normal(0,logit_std))
    logit_log_dmMin = numpyro.sample("logit_log_dmMin",dist.Normal(0,logit_std))
    logit_log_dmMax = numpyro.sample("logit_log_dmMax",dist.Normal(0,logit_std))

    # Inverse transform back to the physical parameters of interest
    sig_m1,jac_sig_m1 = get_value_from_logit(logit_sig_m1,2.,15.)
    log_f_peak,jac_log_f_peak = get_value_from_logit(logit_log_f_peak,-3,0.)
    mMax,jac_mMax = get_value_from_logit(logit_mMax,50.,100.)
    log_dmMin,jac_log_dmMin = get_value_from_logit(logit_log_dmMin,-1,1)
    log_dmMax,jac_log_dmMax = get_value_from_logit(logit_log_dmMax,0.5,1.5)

    numpyro.deterministic("sig_m1",sig_m1)
    numpyro.deterministic("log_f_peak",log_f_peak)
    numpyro.deterministic("mMax",mMax)
    numpyro.deterministic("log_dmMin",log_dmMin)
    numpyro.deterministic("log_dmMax",log_dmMax)

    # Override prior factors of logit quantities, and impose a uniform prior in the physical space
    numpyro.factor("p_sig_m1",logit_sig_m1**2/(2.*logit_std**2)-jnp.log(jac_sig_m1))
    numpyro.factor("p_log_f_peak",logit_log_f_peak**2/(2.*logit_std**2)-jnp.log(jac_log_f_peak))
    numpyro.factor("p_mMax",logit_mMax**2/(2.*logit_std**2)-jnp.log(jac_mMax))
    numpyro.factor("p_log_dmMin",logit_log_dmMin**2/(2.*logit_std**2)-jnp.log(jac_log_dmMin))
    numpyro.factor("p_log_dmMax",logit_log_dmMax**2/(2.*logit_std**2)-jnp.log(jac_log_dmMax))

    # Compute normalization factors necessary to ensure that `R20` is correctly defined as the
    # merger rate at the desired reference mass and redshift values
    f_m1_norm = massModel(20.,alpha,mu_m1,sig_m1,10.**log_f_peak,mMax,mMin,10.**log_dmMax,10.**log_dmMin)
    f_z_norm = (1.+0.2)**kappa

    # Read out found injections
    m1_det = injectionDict['m1']
    m2_det = injectionDict['m2']
    z_det = injectionDict['z']
    dVdz_det = injectionDict['dVdz']
    p_draw = injectionDict['p_draw_m1m2z']*injectionDict['p_draw_chiEff_chiP']

    # Compute proposed population weights
    # Note that draw weights are defined as a probability density over redshift and detector frame time
    # We therefore need to multiply by dVdz*(1+z)**(-1) to convert from a source-frame merger rate density
    f_m1_det = massModel(m1_det,alpha,mu_m1,sig_m1,10.**log_f_peak,mMax,mMin,10.**log_dmMax,10.**log_dmMin)/f_m1_norm
    p_m2_det = (1.+bq)*m2_det**bq/(m1_det**(1.+bq)-tmp_min**(1.+bq))
    f_Xeff_det = f_Xeff_eventSorted[full_chi_data['injections_from_allSamples']]
    f_Xp_det = f_Xp_eventSorted[full_chi_data['injections_from_allSamples']]
    f_z_det = dVdz_det*(1.+z_det)**(kappa-1.)/f_z_norm 

    # All together, the quantity below is the detection rate dN/dm1*dm2*dXeff*dXp*dz*dt_det
    R_pop_det = R20*f_m1_det*p_m2_det*f_z_det*f_Xeff_det*f_Xp_det

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
    
    ###############################
    # Compute per-event likelihoods
    ###############################

    # This function defines the per-event log-likelihood. It expects the following arguments:
    # m1_sample, z_sample...: Arrays of posterior samples for the given event
    # priors: Corresponding array of prior probabilities assigned to each sample
    # ar_indices: Indices used to retrieve the correct AR1 rates corresponding to this event's samples
    def logp(m1_sample,m2_sample,z_sample,dVdz_sample,priors,ar_indices):

        # Compute proposed population weights, analogous to calculation for injections done above
        # Use `ar_indices` to extract the correct values of `f_Xeff_eventSorted` and `f_Xp_eventSorted`
        # corresponding to each of this event's posterior samples
        f_m1 = massModel(m1_sample,alpha,mu_m1,sig_m1,10.**log_f_peak,mMax,mMin,10.**log_dmMax,10.**log_dmMin)/f_m1_norm
        p_m2 = (1.+bq)*m2_sample**bq/(m1_sample**(1.+bq)-tmp_min**(1.+bq))
        f_Xeff = f_Xeff_eventSorted[ar_indices]
        f_Xp = f_Xp_eventSorted[ar_indices]
        f_z = dVdz_sample*(1.+z_sample)**(kappa-1.)/f_z_norm

        # All together, the quantity below is the detection rate dN/dm1*dm2*dXeff*dXp*dz*dt_det
        R_pop = R20*f_m1*p_m2*f_z*f_Xeff*f_Xp

        # Compute effective number of samples and return log-likelihood
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
                        jnp.array([sampleDict[k]['z_prior']*sampleDict[k]['joint_priors'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['ar_indices'] for k in sampleDict]))
        
    # As a diagnostic, save minimum number of effective samples across all events
    numpyro.deterministic('min_log_neff',jnp.min(jnp.log10(n_effs)))

    # Tally log-likelihoods across our catalog
    numpyro.factor("logp",jnp.sum(log_ps))

def ar_spinMagTilt_priorOnly(sampleDict,injectionDict,full_chi_data):

    """
    Implementation of a Gaussian effective spin distribution for inference within `numpyro`

    Parameters
    ----------
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections
    """

    all_chi_samples = full_chi_data['chi_allSamples']
    ind_a01 = full_chi_data['ind_a01']
    chi_deltas = full_chi_data['chi_deltas']
    chi_deltas_low = chi_deltas[:ind_a01][::-1]
    chi_deltas_high = chi_deltas[ind_a01:]

    all_cost_samples = full_chi_data['cost_allSamples']
    cost_deltas = full_chi_data['cost_deltas'][::-1]

    ############################################################
    # First sample the properties of our autoregressive process
    # First get variance of the process
    # We are imposing a steep power-law prior on this parameter
    ar_chi_std = numpyro.sample("ar_chi_std",dist.HalfNormal(1.))
    numpyro.factor("ar_chi_std_prior",ar_chi_std**2/2. - (ar_chi_std/0.75)**4/8.75)

    # Finally the autocorrelation length
    # Since the posterior for this parameter runs up against prior boundaries, sample in logit space
    #log_ar_chi_tau = numpyro.sample("log_ar_chi_tau",dist.Normal(0.,1))
    #ar_chi_tau = numpyro.deterministic("ar_chi_tau",10.**log_ar_chi_tau)
    logit_ar_chi_tau = numpyro.sample("logit_ar_chi_tau",dist.Normal(0,logit_std))
    ar_chi_tau,jac_ar_chi_tau = get_value_from_logit(logit_ar_chi_tau,0.2,2.)
    numpyro.factor("p_ar_chi_tau",logit_ar_chi_tau**2/(2.*logit_std**2)-jnp.log(jac_ar_chi_tau))
    numpyro.deterministic("ar_chi_tau",ar_chi_tau)
    numpyro.factor("chi_regularization",-(ar_chi_std/jnp.sqrt(ar_chi_tau))**2/(2.*0.4**2))

    # Sample an initial rate density at reference point
    ln_f_chi_ref_unscaled = numpyro.sample("ln_f_chi_ref_unscaled",dist.Normal(0,1))
    ln_f_chi_ref = ln_f_chi_ref_unscaled*ar_chi_std

    # Generate forward steps
    chi_steps_forward = numpyro.sample("chi_steps_forward",dist.Normal(0,1),sample_shape=(chi_deltas_high.size,))
    chi_phis_forward = jnp.exp(-chi_deltas_high/ar_chi_tau)
    chi_ws_forward = jnp.sqrt(1.-jnp.exp(-2.*chi_deltas_high/ar_chi_tau))*(ar_chi_std*chi_steps_forward)
    final,ln_f_chis_high = lax.scan(build_ar1,ln_f_chi_ref,jnp.transpose(jnp.array([chi_phis_forward,chi_ws_forward]))) 
    ln_f_chis = jnp.append(ln_f_chi_ref,ln_f_chis_high)

    # Generate backward steps
    chi_steps_backward = numpyro.sample("chi_steps_backward",dist.Normal(0,1),sample_shape=(chi_deltas_low.size,))
    chi_phis_backward = jnp.exp(-chi_deltas_low/ar_chi_tau)
    chi_ws_backward = jnp.sqrt(1.-jnp.exp(-2.*chi_deltas_low/ar_chi_tau))*(ar_chi_std*chi_steps_backward)
    final,ln_f_chis_low = lax.scan(build_ar1,ln_f_chi_ref,jnp.transpose(jnp.array([chi_phis_backward,chi_ws_backward])))
    ln_f_chis = jnp.append(ln_f_chis_low[::-1],ln_f_chis)

    # Exponentiate and save
    f_chis = jnp.exp(ln_f_chis)
    f_chi_eventSorted = f_chis[full_chi_data['chi_reverseSorting']]
    numpyro.deterministic("f_chis",f_chis)

    ############################
    # Construct AR1 process in cost
    ############################

    # First sample the properties of our autoregressive process
    # First get variance of the process
    ar_cost_std = numpyro.sample("ar_cost_std",dist.HalfNormal(1.))
    numpyro.factor("ar_cost_std_prior",ar_cost_std**2/2. - (ar_cost_std/0.75)**4/8.75)

    # Finally the autocorrelation length
    #log_ar_cost_tau = numpyro.sample("log_ar_cost_tau",dist.Normal(0.,1.))
    #ar_cost_tau = numpyro.deterministic("ar_cost_tau",10.**log_ar_cost_tau)
    logit_ar_cost_tau = numpyro.sample("logit_ar_cost_tau",dist.Normal(0,logit_std))
    ar_cost_tau,jac_ar_cost_tau = get_value_from_logit(logit_ar_cost_tau,0.3,4.)
    numpyro.factor("p_ar_cost_tau",logit_ar_cost_tau**2/(2.*logit_std**2)-jnp.log(jac_ar_cost_tau))
    numpyro.deterministic("ar_cost_tau",ar_cost_tau)
    numpyro.factor("cost_regularization",-(ar_cost_std/jnp.sqrt(ar_cost_tau))**2/(2.*0.4**2))

    ln_f_cost_ref_unscaled = numpyro.sample("ln_f_cost_ref_unscaled",dist.Normal(0,1))
    ln_f_cost_ref = ln_f_cost_ref_unscaled*ar_cost_std

    # Generate backward steps and prepend to reference value
    cost_steps_backward = numpyro.sample("cost_steps_backward",dist.Normal(0,1),sample_shape=(cost_deltas.size,))
    cost_phis_backward = jnp.exp(-cost_deltas/ar_cost_tau)
    cost_ws_backward = jnp.sqrt(-jnp.expm1(-2.*cost_deltas/ar_cost_tau))*(ar_cost_std*cost_steps_backward)
    final,ln_f_costs = lax.scan(build_ar1,ln_f_cost_ref,jnp.transpose(jnp.array([cost_phis_backward,cost_ws_backward])))
    ln_f_costs = jnp.append(ln_f_costs[::-1],ln_f_cost_ref)

    # Exponentiate and save
    f_cost = jnp.exp(ln_f_costs)
    f_cost_eventSorted = f_cost[full_chi_data['cost_reverseSorting']]
    numpyro.deterministic("f_cost",f_cost)

