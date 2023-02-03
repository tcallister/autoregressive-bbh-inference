import numpy as np
from astropy.cosmology import Planck15
import astropy.units as u
from utilities import massModel
import os
dirname = os.path.dirname(__file__)

def reweighting_function_arlnm1_q(m1,m2,a1,a2,cost1,cost2,z,dVdz):

    """
    Function that computes weights under a fixed population similar to that preferred by
    the ln(m1),q AR1 analysis. Preemptively reweighting posteriors and injections to this population
    can help increase sampling efficiency when subsequently inferring the actual population distribution of BBHs

    Parameters
    ----------
    m1 : `np.array`
        Primary BBH masses (units of Msun)
    m2 : `np.array`
        Secondary BBH masses (units of Msun)
    a1 : `np.array`
        Primary spin magnitudes
    a2 : `np.array`
        Secondary spin magnitudes
    cost1 : `np.array`
        Cosines of primary spin-orbit tilts
    cost2 : `np.array`
        Cosines of secondary spin-orbit tilts
    z : `np.array`
        Redshifts
    dVdz : `np.array`
        Differential comoving volumes correspond to each redshift in `z`

    Returns
    -------
    dP_dm1m2a1a2cost1cost2z : `np.array`
        Probability density evaluated at each `(m1,m2,a1,a2,cost1,cost2,z)` under the target population
    """

    # Hyperparameters characterizing target distribution
    kappa = 3.
    q_std = 0.5
    alpha = -2.5
    m0 = 10.
    f_pl = 0.3
    f_peak_in_peak1 = 0.9

    # Construct primary mass distribution
    p_m1 = np.zeros_like(m1)
    p_m1 += (1.-f_pl)*f_peak_in_peak1*np.exp(-(m1-10.)**2/(2.*0.8**2))/np.sqrt(2.*np.pi*0.8**2)
    p_m1 += (1.-f_pl)*(1.-f_peak_in_peak1)*np.exp(-(m1-35.)**2/(2.*2.**2.))/np.sqrt(2.*np.pi*2.**2)
    p_m1_pl = np.ones(m1.size)
    p_m1_pl[m1<m0] = 1.
    p_m1_pl[m1>=m0] = (m1[m1>=m0]/m0)**alpha
    p_m1_pl /= (alpha*m0 - 2. - alpha*2.)/(1.+alpha)
    p_m1 += f_pl*p_m1_pl 
    p_m1[m1<2.] = 0.
    p_m1[m1>100.] = 0.

    # Compute mass ratio distribution and convert to a density over m2
    q = m2/m1
    p_q = np.exp(-(q-1)**2/(2.*q_std**2))*2./np.sqrt(2.*np.pi*q_std**2)
    p_m2 = p_q/m1

    # Redshift distribution
    p_z = dVdz*(1.+z)**(kappa-1.)
    z_grid = np.linspace(0,2.3,1000)
    dVdz_grid = 4.*np.pi*Planck15.differential_comoving_volume(z_grid).to(u.Gpc**3*u.sr**(-1)).value
    norm = np.trapz(dVdz_grid*(1.+z_grid)**(kappa-1.))
    p_z /= norm

    # Isotropic component spins
    p_a1 = 1.
    p_a2 = 1.
    p_cost1 = 1./2.
    p_cost2 = 1./2.

    return (p_m1*p_m2*p_a1*p_a2*p_cost1*p_cost2*p_z)

def getInjections(sample_limit=10000,reweight=False,weighting_function=reweighting_function_arlnm1_q):

    """
    Function to load and preprocess found injections for use in numpyro likelihood functions.

    Parameters
    ----------
    sample_limit : int
        If reweighting to a different reference population, number of found injections to use. No effect otherwise. Default 1e4
    reweight : bool
        If `True`, reweight injections to the reference population defined in `weighting_function`. Default `False`
    weighting_function : func
        Function defining new reference population to reweight to. No effect if `reweight=True`. Default `reweighting_function_arlnm1_q`

    Returns
    ------- 
    injectionDict : dict
        Dictionary containing found injections and associated draw probabilities, for downstream use in hierarchical inference
    """

    # Load injections
    injectionFile = os.path.join(dirname,"./../input/injectionDict_FAR_1_in_1.pickle")
    injectionDict = np.load(injectionFile,allow_pickle=True)

    # Convert all lists to numpy arrays
    for key in injectionDict:
        if key!='nTrials':
            injectionDict[key] = np.array(injectionDict[key])

    # If reweighting injections to a new reference population...
    if reweight:

        print("REWEIGHTING")

        # Extract injection parametetrs and old draw weights
        m1 = injectionDict['m1']
        m2 = injectionDict['m2']
        z = injectionDict['z']
        dVdz = injectionDict['dVdz']
        a1 = injectionDict['a1']
        a2 = injectionDict['a2']
        cost1 = injectionDict['cost1']
        cost2 = injectionDict['cost2']
        pDraw = injectionDict['p_draw_m1m2z']*injectionDict['p_draw_a1a2cost1cost2']

        # Define updated probabilities under new reference population
        p_new = weighting_function(m1,m2,a1,a2,cost1,cost2,z,dVdz)

        # The ratio between new andn old draw weights will be used to resample found injections
        downselection_draw_weights = (p_new/pDraw)/injectionDict['nTrials']
        nEff_downselection = np.sum(downselection_draw_weights)**2/np.sum(downselection_draw_weights**2)

        # Save new draw weights, which will take the place of `pDraw` above
        # For convenience, put the weights entirely in `p_draw_m1m2z`, leaving `p_draw_a1a2cost1cost2` to be identically one
        new_pDraw = (p_new)/np.sum(downselection_draw_weights)
        injectionDict['downselection_Neff'] = nEff_downselection
        injectionDict['nTrials'] = sample_limit
        injectionDict['p_draw_m1m2z'] = new_pDraw
        injectionDict['p_draw_a1a2cost1cost2'] = np.ones(new_pDraw.size)
        
        # Randomly resample injections
        print(nEff_downselection)
        inds_to_keep = np.random.choice(np.arange(new_pDraw.size),size=sample_limit,replace=True,p=downselection_draw_weights/np.sum(downselection_draw_weights))
        for key in injectionDict:
            if key!='nTrials' and key!='downselection_Neff':
                injectionDict[key] = injectionDict[key][inds_to_keep]

    return injectionDict

def getSamples(sample_limit=2000,bbh_only=True,reweight=False,weighting_function=reweighting_function_arlnm1_q):

    """
    Function to load and preprocess BBH posterior samples for use in numpyro likelihood functions.
    
    Parameters
    ----------
    sample_limit : int
        Number of posterior samples to retain for each event, for use in population inference (default 2000)
    bbh_only : bool
        If True, will exclude samples for BNS, NSBH, and mass-gap events (default True)
    reweight : bool
        If True, reweight posteriors to the reference population defined in `weighting_function`. Default `False`
    weighting_function : func
        Function defining new reference population to reweight to. No effect if `reweight=True`. Default `reweighting_function_arlnm1_q`

    Returns
    -------
    sampleDict : dict
        Dictionary containing posterior samples, for downstream use in hierarchical inference
    """

    # Load dictionary with preprocessed posterior samples
    sampleFile = os.path.join(dirname,"./../input/sampleDict_FAR_1_in_1_yr.pickle")
    sampleDict = np.load(sampleFile,allow_pickle=True)

    # Remove non-BBH events, if desired
    non_bbh = ['GW170817','S190425z','S190426c','S190814bv','S190917u','S200105ae','S200115j']
    if bbh_only:
        for event in non_bbh:
            print("Removing ",event)
            sampleDict.pop(event)

    # Loop across events
    for event in sampleDict:

        # If we are preemptively reweighting to a new reference prior...
        if reweight:

            # Extract samples
            m1 = np.array(sampleDict[event]['m1'])
            m2 = np.array(sampleDict[event]['m2'])
            z = np.array(sampleDict[event]['z'])
            dVdz = np.array(sampleDict[event]['dVc_dz'])
            a1 = np.array(sampleDict[event]['a1'])
            a2 = np.array(sampleDict[event]['a2'])
            cost1 = np.array(sampleDict[event]['cost1'])
            cost2 = np.array(sampleDict[event]['cost2'])
            prior = np.array(sampleDict[event]['z_prior'])

            # Define new population prior, and for convenience put it into the pre-existing `z_prior` field
            p_new = weighting_function(m1,m2,a1,a2,cost1,cost2,z,dVdz)
            sampleDict[event]['z_prior'] = p_new

            # Define weights for random downselection below
            draw_weights = p_new/prior
            draw_weights /= np.sum(draw_weights)
            neff = np.sum(draw_weights)**2/np.sum(draw_weights**2)

            if neff<2.*sample_limit:
                print(event,neff)

        # If not reweighting...
        else:
            
            # Uniform draw weights
            draw_weights = np.ones(sampleDict[event]['m1'].size)/sampleDict[event]['m1'].size

        sampleDict[event]['downselection_Neff'] = np.sum(draw_weights)**2/np.sum(draw_weights**2)

        # Randomly downselect to the desired number of samples       
        inds_to_keep = np.random.choice(np.arange(sampleDict[event]['m1'].size),size=sample_limit,replace=True,p=draw_weights)
        for key in sampleDict[event].keys():
            if key!='downselection_Neff':
                sampleDict[event][key] = sampleDict[event][key][inds_to_keep]

    return sampleDict
