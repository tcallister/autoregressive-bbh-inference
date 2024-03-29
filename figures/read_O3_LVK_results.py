import numpy as np
import json
from scipy.interpolate import interp1d,interp2d
from scipy.special import beta as bfunc
from scipy.special import erf
from astropy.cosmology import Planck15
import astropy.units as u
import sys
sys.path.append('./../code/')
from getData import *
from utilities import calculate_gaussian_2D
import matplotlib.pyplot as plt

def read_lvk_plpeak_data():

    """
    Function to load samples from LVK O3b population analysis using the PL+Peak mass model and Default spin model

    Returns
    -------
    lvk_results : dict
        Dictionary containing hyperposterior samples
    """

    # Retrive json file
    with open('./../input/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json','r') as jf:
        plpeak_data = json.load(jf) 

    # Read out mass parameters
    plpeak_alphas = np.array(plpeak_data['posterior']['content']['alpha'])
    plpeak_mMaxs = np.array(plpeak_data['posterior']['content']['mmax'])
    plpeak_mMins = np.array(plpeak_data['posterior']['content']['mmin'])
    plpeak_fPeaks = np.array(plpeak_data['posterior']['content']['lam'])
    plpeak_mu_m1s = np.array(plpeak_data['posterior']['content']['mpp'])
    plpeak_sig_m1s = np.array(plpeak_data['posterior']['content']['sigpp'])
    plpeak_delta_ms = np.array(plpeak_data['posterior']['content']['delta_m'])
    plpeak_betas = np.array(plpeak_data['posterior']['content']['beta'])

    # Read rate and redshift evolution parameters
    plpeak_kappas = np.array(plpeak_data['posterior']['content']['lamb'])
    plpeak_Rtot = np.array(plpeak_data['posterior']['content']['rate'])

    # Read spin parameters
    # Remember that `sigma_chi` as computed by gwpopulation is actually the *variance* of the spin magnitude distribution
    default_mu_chi = np.array(plpeak_data['posterior']['content']['mu_chi'])
    default_sig_chi = np.sqrt(np.array(plpeak_data['posterior']['content']['sigma_chi']))
    default_sig_aligned = np.array(plpeak_data['posterior']['content']['sigma_spin'])
    default_f_aligned = np.array(plpeak_data['posterior']['content']['xi_spin'])

    # Construct and return dictionary
    lvk_results = {
        'alpha':plpeak_alphas,
        'mMax':plpeak_mMaxs,
        'mMin':plpeak_mMins,
        'fPeak':plpeak_fPeaks,
        'mu_m1':plpeak_mu_m1s,
        'sig_m1':plpeak_sig_m1s,
        'delta_m':plpeak_delta_ms,
        'bq':plpeak_betas,
        'kappa':plpeak_kappas,
        'Rtot':plpeak_Rtot,
        'mu_chi':default_mu_chi,
        'sig_chi':default_sig_chi,
        'sig_cost':default_sig_aligned,
        'f_aligned':default_f_aligned
        } 

    return lvk_results

def read_lvk_gaussian_spin_data():

    """
    Function to load samples from LVK O3b population analysis using a (variant of the) PL+Peak mass model and Gaussian effective spin model

    Returns
    -------
    lvk_results : dict
        Dictionary containing hyperposterior samples
    """

    # Retrieve file
    with open('../input/posteriors_gaussian_spin_samples_FAR_1_in_1.json','r') as jf:
        gaussian_spin_data = json.load(jf)

    # Read out hyperparameters
    gaussian_bq = np.array(gaussian_spin_data['beta_q'])
    gaussian_mMax = np.array(gaussian_spin_data['mMax'])
    gaussian_kappa = np.array(gaussian_spin_data['kappa'])
    gaussian_alpha = np.array(gaussian_spin_data['lmbda'])
    gaussian_mu_m1 = np.array(gaussian_spin_data['m0'])
    gaussian_sig_m1 = np.array(gaussian_spin_data['sigM'])
    gaussian_f_peak = np.array(gaussian_spin_data['peak_fraction'])
    gaussian_chiEff_mean = np.array(gaussian_spin_data['chiEff_mean'])
    gaussian_chiEff_std = np.array(gaussian_spin_data['chiEff_std'])
    gaussian_chiP_mean = np.array(gaussian_spin_data['chiP_mean'])
    gaussian_chiP_std = np.array(gaussian_spin_data['chiP_std'])
    gaussian_rho = np.array(gaussian_spin_data['rho_chiEff_chiP'])

    # Construct and return dictionary
    lvk_results = {
        'bq':gaussian_bq,
        'mMax':gaussian_mMax,
        'kappa':gaussian_kappa,
        'alpha':gaussian_alpha,
        'mu_m1':gaussian_mu_m1,
        'sig_m1':gaussian_sig_m1,
        'f_peak':gaussian_f_peak,
        'chiEff_mean':gaussian_chiEff_mean,
        'chiEff_std':gaussian_chiEff_std,
        'chiP_mean':gaussian_chiP_mean,
        'chiP_std':gaussian_chiP_std,
        'rho':gaussian_rho 
        } 

    return lvk_results

def plpeak_m1_q(alpha,mMax,mMin,fPeak,mu_m1,sig_m1,delta_m,bq,kappa,R,npts):

    """
    Helper function that computes the merger rate over a grid of primary masses and mass ratios, according to
    a PowerLaw+Peak mass model

    Inputs
    ------
    alpha : float
        Slope on the "power law" piece of the Power-Law+Peak primary mass model
    mMax : float
        Maximum black hole mass
    mMin : float
        Minimum black hole mass
    fPeak : float
        Fraction of primaries occupying the "peak" part of the Power-Law+Peak primary mass model
    mu_m1 : float
        Mean location of the primary mass peak
    sig_m1 : float
        Standard deviation of the primary mass peak
    delta_m : float
        Smoothing length over which the primary mass distribution "turns on" above `mMin`
    bq : float
        Power-law index governing the mass ratio distribution
    R : float
        Overall merger rate (integrated across masses) at `z=0`
    npts : int
        Number of grid points to use when constructing primary mass and mass ratio grid

    Returns
    -------
    m1_grid : np.array
        An array of primary mass grid points
    q_grid : np.array
        An array of mass ratio grid points
    dR_dm1_dq : np.array
        2D array of merger rate densities `dR/(dm1*dq)` defined across `m1_grid` and `q_grid`
    """
    
    # Define primary mass and mass ratio grid
    # Make grids slightly different sizes to disambiguate dimensions
    m1_grid = np.linspace(2,100,npts)
    q_grid = np.linspace(0,1,npts+1)
    M,Q = np.meshgrid(m1_grid,q_grid)
    
    # Primary mass probability distribution
    # Start by defining normalized power-law and gaussian components
    pl = (1.-alpha)*M**(-alpha)/(mMax**(1.-alpha) - mMin**(1.-alpha))
    pl[M<mMin] = 0
    pl[M>mMax] = 0
    peak = np.exp(-(M-mu_m1)**2/(2.*sig_m1**2))/np.sqrt(2.*np.pi*sig_m1**2)
    
    # Identify masses at which smoothing will be applied
    smoothing = np.ones(M.shape)
    to_smooth = (M>mMin)*(M<mMin+delta_m)

    # Then define and apply smoothing factor
    smoothing[to_smooth] = 1./(np.exp(delta_m/(M[to_smooth]-mMin) + delta_m/(M[to_smooth]-mMin-delta_m))+1.)
    smoothing[M<mMin] = 0
    p_m1_unnormed = (fPeak*peak + (1.-fPeak)*pl)*smoothing
    
    # Similarly identify (m1,q) gridpoints for which smoothing should be applied on m2
    # Define the corresponding smoothing factor
    q_smoothing = np.ones(Q.shape)
    to_smooth = (M*Q>mMin)*(M*Q<mMin+delta_m)
    q_smoothing[to_smooth] = 1./(np.exp(delta_m/((Q*M)[to_smooth]-mMin) + delta_m/((Q*M)[to_smooth]-mMin-delta_m))+1.)
    q_smoothing[(M*Q)<mMin] = 0
    
    # Define mass ratio distribution, including smoothing
    p_q_unnormed = Q**bq/(1.-(mMin/M)**(1.+bq))*q_smoothing
    p_q_unnormed[Q<(mMin/M)] = 0.

    # Normalize the conditional p(q|m1)
    # Occasionally we run into trouble normalizing p(q|m1) when working with m1 values sufficiently small that p(m1)=p(q|m1)=0 for all q
    # In this case, overwrite and set p(q|m1)=0
    p_q = p_q_unnormed/np.trapz(p_q_unnormed,q_grid,axis=0)
    p_q[p_q!=p_q] = 0
    
    # Combine primary mass and mass ratio distributions and normalize over m1
    p_m1_q_unnormed = p_m1_unnormed*p_q
    p_m1_q = p_m1_q_unnormed/(np.sum(p_m1_q_unnormed)*(m1_grid[1]-m1_grid[0])*(q_grid[1]-q_grid[0]))

    # Scale by total rate at z=0.2 and return
    dR_dm1_dq = R*(1.+0.2)**kappa*p_m1_q
    
    return m1_grid,q_grid,dR_dm1_dq

def default_spin(mu_chi,sig_chi,sig_cost,npts):

    """
    Helper function that computes component spin probability densities over a grid of spin magnitude and tilts,
    according to the `Default` spin model (in which spin magnitudes are Beta-distributed, while tilts are described
    as a mixture between isotropic and preferentially-aligned populations)

    Inputs
    ------
    mu_chi : float
        Mean of the Beta spin-magnitude distribution
    sig_chi : float
        Standard deviation of the Beta spin-magnitude distribution
    sig_cost : float
        Standard deviation of the preferentially-aligned subpopulation
    npts : int
        Number of grid points to use when constructing spin magnitude and tilt grids

    Returns
    -------
    chi_grid : np.array
        An array of component spin magnitude grid points
    cost_grid : np.array
        An array of component spin tilt grid points
    p_chi : np.array
        Spin magnitude probability densities defined across `chi_grid`
    p_cost_peak : np.array
        Spin tilt probabilities from the preferentially-aligned subpopulation, defined across `cost_grid`.
    p_cost_iso : np.array
        Spin tilt probabilities from the isotropic subpopulation.
        The full spin-tilt probability distribution is given by `f_iso*p_cost_iso + (1-f_iso)*p_cost_peak` for some
        mixture fraction `f_iso`
    """

    # Define grid of spin magnitude and (cosine) spin tilt values
    chi_grid = np.linspace(0,1,npts)
    cost_grid = np.linspace(-1,1,npts+1)

    # Transform mu_chi and sig_chi to beta distribution "alpha" and "beta" shape parameters
    nu = mu_chi*(1.-mu_chi)/sig_chi**2 - 1.
    alpha = mu_chi*nu
    beta = (1.-mu_chi)*nu

    # Define spin magnitude probability distribution
    p_chi = chi_grid**(alpha-1.)*(1.-chi_grid)**(beta-1.)/bfunc(alpha,beta)
    
    # Preferentially-aligned probability densities
    p_cost_peak = np.exp(-(cost_grid-1.)**2/(2.*sig_cost**2))/np.sqrt(2.*np.pi*sig_cost**2)
    p_cost_peak /= erf(0.)/2. - erf(-2./np.sqrt(2.*sig_cost**2))/2.

    # Finally, define the (constant) isotropic probability densities
    p_cost_iso = np.ones(cost_grid.size)/2.

    return chi_grid,cost_grid,p_chi,p_cost_peak,p_cost_iso
    
def get_lvk_z(nTraces,m1_ref=20,nGridpoints=500):

    # Get posterior samples
    lvk_data = read_lvk_plpeak_data()

    z_grid = np.linspace(0,2,nGridpoints)
    R_zs = np.zeros((nTraces,nGridpoints))

    random_inds = np.random.choice(np.arange(lvk_data['alpha'].size),nTraces,replace=False)
    for i in range(nTraces):
        
        ind = random_inds[i]
        m1_grid,q_grid,R_m1_q = plpeak_m1_q(lvk_data['alpha'][ind],
                             lvk_data['mMax'][ind],
                             lvk_data['mMin'][ind],
                             lvk_data['fPeak'][ind],
                             lvk_data['mu_m1'][ind],
                             lvk_data['sig_m1'][ind],
                             lvk_data['delta_m'][ind],
                             lvk_data['bq'][ind],
                             lvk_data['kappa'][ind],
                             lvk_data['Rtot'][ind],
                             nGridpoints)

        # Convert to merger rate per log mass
        R_lnm1_q = R_m1_q*m1_grid[np.newaxis,:]
        #R_lnm1 = np.trapz(R_lnm1_q,q_grid,axis=0)
        R_lnm1 = R_lnm1_q[-1,:]

        # Interpolate to reference points
        R_z02_interpolator = interp1d(m1_grid,R_lnm1)
        R_z02_ref = R_z02_interpolator(m1_ref)
        #R_z02_ref = np.sum(R_m1_q)*(m1_grid[1]-m1_grid[0])*(q_grid[1]-q_grid[0])

        # Rescale over z grid
        R_zs[i,:] = R_z02_ref*(1.+z_grid)**lvk_data['kappa'][ind]/(1.+0.2)**lvk_data['kappa'][ind]

    return z_grid,R_zs

def get_lvk_componentSpin(nTraces,m1_ref=20,q_ref=1,nGridpoints=500):

    lvk_data = read_lvk_plpeak_data()

    R_chi1_chi2 = np.zeros((nTraces,nGridpoints))
    R_cost1_cost2 = np.zeros((nTraces,nGridpoints+1))
    p_chis = np.zeros((nTraces,nGridpoints))
    p_costs = np.zeros((nTraces,nGridpoints+1))

    random_inds = np.random.choice(np.arange(lvk_data['alpha'].size),nTraces,replace=False)
    for i in range(nTraces):
        
        ind = random_inds[i]
        m1_grid,q_grid,dR_dm1_dq = plpeak_m1_q(lvk_data['alpha'][ind],
                             lvk_data['mMax'][ind],
                             lvk_data['mMin'][ind],
                             lvk_data['fPeak'][ind],
                             lvk_data['mu_m1'][ind],
                             lvk_data['sig_m1'][ind],
                             lvk_data['delta_m'][ind],
                             lvk_data['bq'][ind],
                             lvk_data['kappa'][ind],
                             lvk_data['Rtot'][ind],
                             nGridpoints)

        # Convert to merger rate per log mass, store
        dR_dlnm1_dq = dR_dm1_dq*m1_grid[np.newaxis,:]

        # Extract rate at reference points
        R_interpolator = interp2d(m1_grid,q_grid,dR_dlnm1_dq)
        dR_dlnm1_dq_ref = R_interpolator(m1_ref,q_ref)

        # Get spin distribution data
        chi_grid,cost_grid,p_chi,p_cost_peak,p_cost_iso = default_spin(lvk_data['mu_chi'][ind],lvk_data['sig_chi'][ind],lvk_data['sig_cost'][ind],nGridpoints)

        # Evaluate merger rate at chi1=chi2=chi, cost1=cost2=1
        p_cost1_cost2_1 = lvk_data['f_aligned'][ind]*p_cost_peak[-1]**2 + (1.-lvk_data['f_aligned'][ind])*p_cost_iso[-1]**2
        p_chi1_chi2 = p_chi**2
        R_chi1_chi2[i,:] = dR_dlnm1_dq_ref*p_cost1_cost2_1*p_chi1_chi2
 
        # Evalute merger rate at chi1=chi2=0.1, cost1=cost2
        p_cost1_cost2 = lvk_data['f_aligned'][ind]*p_cost_peak**2 + (1.-lvk_data['f_aligned'][ind])*p_cost_iso**2
        p_chi1_chi2_01 = np.interp(0.1,chi_grid,p_chi)**2
        R_cost1_cost2[i,:] = dR_dlnm1_dq_ref*p_cost1_cost2*p_chi1_chi2_01

        # Store marginal component spin probability distributions
        p_chis[i,:] = p_chi
        p_costs[i,:] = lvk_data['f_aligned'][ind]*p_cost_peak + (1.-lvk_data['f_aligned'][ind])*p_cost_iso

    return chi_grid,cost_grid,R_chi1_chi2,R_cost1_cost2,p_chis,p_costs
    
def get_lvk_m1_q(nTraces,nGridpoints=500):

    lvk_data = read_lvk_plpeak_data()

    R_m1s_qs = np.zeros((nTraces,nGridpoints+1,nGridpoints))
    random_inds = np.random.choice(np.arange(lvk_data['alpha'].size),nTraces,replace=False)

    for i in range(nTraces):
        
        ind = random_inds[i]
        m1_grid,q_grid,R_m1_q = plpeak_m1_q(lvk_data['alpha'][ind],
                             lvk_data['mMax'][ind],
                             lvk_data['mMin'][ind],
                             lvk_data['fPeak'][ind],
                             lvk_data['mu_m1'][ind],
                             lvk_data['sig_m1'][ind],
                             lvk_data['delta_m'][ind],
                             lvk_data['bq'][ind],
                             lvk_data['kappa'][ind],
                             lvk_data['Rtot'][ind],
                             nGridpoints)
        
        R_m1s_qs[i,:,:] = R_m1_q
    
    return m1_grid,q_grid,R_m1s_qs

def get_lvk_gaussian_spin():

    # Get posterior samples
    lvk_results = read_lvk_gaussian_spin_data()

    # Load dictionary of injections and posterior samples, which we will need in order to resample rate
    injectionDict = getInjections(sample_limit=50000,reweight=False)
    sampleDict = getSamples(sample_limit=3000,reweight=False)
    nObs = len(sampleDict)*1.

    print(injectionDict['m1'].size,"!!!")

    # Grid of dVdz values, which will be needed to compute total merger rate by integrating over redshift
    z_grid = np.arange(0.01,2.31,0.01)
    dVdz_grid = 4.*np.pi*Planck15.differential_comoving_volume(z_grid).to(u.Gpc**3/u.sr).value
    z_grid = np.concatenate([[0.],z_grid]) 
    dVdz_grid = np.concatenate([[0.],dVdz_grid])

    # Grid over which we will define and normalize effective spin distribution
    Xeff_grid = np.linspace(-1,1,500)
    Xp_grid = np.linspace(0,1,499)
    XEFF,XP = np.meshgrid(Xeff_grid,Xp_grid)
    p_Xeff_Xp = np.zeros((lvk_results['bq'].size,Xeff_grid.size,Xp_grid.size))

    # Instantiate array to hold resampled rates
    R_refs = np.zeros(lvk_results['bq'].size)

    # Loop across population posterior samples
    for i in range(lvk_results['bq'].size):

        bq = lvk_results['bq'][i]
        mMax = lvk_results['mMax'][i]
        alpha = lvk_results['alpha'][i]
        mu_m1 = lvk_results['mu_m1'][i]
        sig_m1 = lvk_results['sig_m1'][i]
        f_peak = lvk_results['f_peak'][i]
        kappa = lvk_results['kappa'][i]
        mu_eff = lvk_results['chiEff_mean'][i]
        sig_eff = lvk_results['chiEff_std'][i]
        mu_p = lvk_results['chiP_mean'][i]
        sig_p = lvk_results['chiP_std'][i]
        rho = lvk_results['rho'][i]

        # Evaluate normalized probability distribution over m1 and m2 of injections
        p_inj_m1 = f_peak*np.exp(-(injectionDict['m1']-mu_m1)**2/(2.*sig_m1**2))/np.sqrt(2.*np.pi*sig_m1**2) \
            + (1.-f_peak)*(1.+alpha)*injectionDict['m1']**alpha/(mMax**(1.+alpha) - 5.**(1.+alpha))
        p_inj_m2 = (1.+bq)*injectionDict['m2']**bq/(injectionDict['m1']**(1.+bq) - 5.**(1.+bq))
        p_inj_m1[injectionDict['m1']>mMax] = 0
        p_inj_m2[injectionDict['m2']<5.] = 0

        # Probability distribution over redshift
        # Note that we need this to be correctly normalized, and so we numerically integrate to obtain the appropriate
        # normalization constant over the range of redshifts considered
        p_z_norm = np.trapz((1.+z_grid)**(kappa-1.)*dVdz_grid,z_grid)
        p_inj_z = (1.+injectionDict['z'])**(kappa-1.)*injectionDict['dVdz']/p_z_norm

        # Finally, compute spin probability distribution
        # This is internally normalized
        p_inj_chi = calculate_gaussian_2D(injectionDict['Xeff'],injectionDict['Xp'],\
                       mu_eff,sig_eff**2,mu_p,sig_p**2,rho)

        # Overall detection efficiency
        xi = np.sum(p_inj_m1*p_inj_m2*p_inj_z*p_inj_chi/(injectionDict['p_draw_m1m2z']*injectionDict['p_draw_chiEff_chiP']))/injectionDict['nTrials']

        # Next, draw an overall intrinsic number of events that occurred in our observation time
        #log_Ntot_grid = np.linspace(8,15,10000)
        log_Ntot_grid = np.linspace(np.log(nObs/xi)-3,np.log(nObs/xi)+3,10000)
        Ntot_grid = np.exp(log_Ntot_grid)
        logp_Ntot_grid = nObs*np.log(xi*Ntot_grid)-xi*Ntot_grid
        logp_Ntot_grid -= np.max(logp_Ntot_grid)
        p_Ntot_grid = np.exp(logp_Ntot_grid)
        p_Ntot_grid /= np.trapz(p_Ntot_grid,log_Ntot_grid)

        cdf_Ntot = np.cumsum(p_Ntot_grid)*(log_Ntot_grid[1]-log_Ntot_grid[0])
        cdf_draw = np.random.random()
        log_Ntot = np.interp(cdf_draw,cdf_Ntot,log_Ntot_grid)
        #print(np.log(nObs/xi),log_Ntot)
        R0 = np.exp(log_Ntot)/p_z_norm/2.

        #fig,ax = plt.subplots()
        #ax.plot(log_Ntot_grid,cdf_Ntot)
        #plt.savefig('{0}.pdf'.format(i))

        # Rescale to our reference values, at m1=20, q=1, z=0.2
        # Additionally multiply by m1=20 so that this is a rate per logarithmic mass,
        # rather than a direct rate per unit mass
        p_m20 = f_peak*np.exp(-(20.-mu_m1)**2/(2.*sig_m1**2))/np.sqrt(2.*np.pi*sig_m1**2) \
            + (1.-f_peak)*(1.+alpha)*20.**alpha/(mMax**(1.+alpha) - 5.**(1.+alpha))
        p_q1 = (1.+bq)/(1. - (5./20.)**(1.+bq))
        R_refs[i] = R0*(1.+0.2)**kappa*p_m20*p_q1*20.

        p_spins = calculate_gaussian_2D(XEFF.reshape(-1),XP.reshape(-1),\
                       mu_eff,sig_eff**2,mu_p,sig_p**2,rho)
        p_Xeff_Xp[i,:] = np.reshape(p_spins,(Xp_grid.size,Xeff_grid.size)).T

        #print(R0,p_m20,p_q1)

    return Xeff_grid,Xp_grid,R_refs,p_Xeff_Xp


if __name__=="__main__":

    #samps = read_lvk_plpeak_data()
    #fPeaks = samps['plpeak_fPeaks']

    get_lvk_gaussian_spin()
    
