import numpy as np
import json
from scipy.interpolate import interp1d,interp2d
from scipy.special import beta as bfunc
from scipy.special import erf

def read_lvk_plpeak_data():

    with open('./../input/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json','r') as jf:
        plpeak_data = json.load(jf) 

    plpeak_alphas = np.array(plpeak_data['posterior']['content']['alpha'])
    plpeak_mMaxs = np.array(plpeak_data['posterior']['content']['mmax'])
    plpeak_mMins = np.array(plpeak_data['posterior']['content']['mmin'])
    plpeak_fPeaks = np.array(plpeak_data['posterior']['content']['lam'])
    plpeak_mu_m1s = np.array(plpeak_data['posterior']['content']['mpp'])
    plpeak_sig_m1s = np.array(plpeak_data['posterior']['content']['sigpp'])
    plpeak_delta_ms = np.array(plpeak_data['posterior']['content']['delta_m'])
    plpeak_betas = np.array(plpeak_data['posterior']['content']['beta'])
    plpeak_kappas = np.array(plpeak_data['posterior']['content']['lamb'])
    plpeak_Rtot = np.array(plpeak_data['posterior']['content']['rate'])

    default_mu_chi = np.array(plpeak_data['posterior']['content']['mu_chi'])
    default_sig_chi = np.sqrt(np.array(plpeak_data['posterior']['content']['sigma_chi'])) # Remember that the stored data is the variance
    default_sig_aligned = np.array(plpeak_data['posterior']['content']['sigma_spin'])
    default_f_aligned = np.array(plpeak_data['posterior']['content']['xi_spin'])

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

def plpeak_m1_q(alpha,mMax,mMin,fPeak,mu_m1,sig_m1,delta_m,bq,kappa,R,npts):
    
    m1_grid = np.linspace(2,100,npts)
    q_grid = np.linspace(0,1,npts+1)
    M,Q = np.meshgrid(m1_grid,q_grid)
    
    pl = (1.-alpha)*M**(-alpha)/(mMax**(1.-alpha) - mMin**(1.-alpha))
    pl[M<mMin] = 0
    pl[M>mMax] = 0
    peak = np.exp(-(M-mu_m1)**2/(2.*sig_m1**2))/np.sqrt(2.*np.pi*sig_m1**2)
    
    smoothing = np.ones(M.shape)
    to_smooth = (M>mMin)*(M<mMin+delta_m)
    
    smoothing[to_smooth] = 1./(np.exp(delta_m/(M[to_smooth]-mMin) + delta_m/(M[to_smooth]-mMin-delta_m))+1.)
    smoothing[M<mMin] = 0
    
    p_m1_unnormed = (fPeak*peak + (1.-fPeak)*pl)*smoothing
    
    q_smoothing = np.ones(Q.shape)
    to_smooth = (M*Q>mMin)*(M*Q<mMin+delta_m)
    q_smoothing[to_smooth] = 1./(np.exp(delta_m/((Q*M)[to_smooth]-mMin) + delta_m/((Q*M)[to_smooth]-mMin-delta_m))+1.)
    q_smoothing[(M*Q)<mMin] = 0
    
    p_q_unnormed = Q**bq/(1.-(mMin/M)**(1.+bq))*q_smoothing
    p_q_unnormed[Q<(mMin/M)] = 0.
    p_q_unnormed /= np.trapz(p_q_unnormed,q_grid,axis=0)
    p_q_unnormed[p_q_unnormed!=p_q_unnormed] = 0
    
    p_m1_q_unnormed = p_m1_unnormed*p_q_unnormed
    p_m1_q = p_m1_q_unnormed/(np.sum(p_m1_q_unnormed)*(m1_grid[1]-m1_grid[0])*(q_grid[1]-q_grid[0]))
    
    return m1_grid,q_grid,R*(1.+0.2)**kappa*p_m1_q

def default_spin(mu_chi,sig_chi,sig_cost,npts):

    chi_grid = np.linspace(0,1,npts)
    cost_grid = np.linspace(-1,1,npts+1)

    # Transform to beta distribution parameters
    nu = mu_chi*(1.-mu_chi)/sig_chi**2 - 1.
    alpha = mu_chi*nu
    beta = (1.-mu_chi)*nu
    p_chi = chi_grid**(alpha-1.)*(1.-chi_grid)**(beta-1.)/bfunc(alpha,beta)
    
    p_cost_peak = np.exp(-(cost_grid-1.)**2/(2.*sig_cost**2))/np.sqrt(2.*np.pi*sig_cost**2)
    p_cost_peak /= erf(0.)/2. - erf(-2./np.sqrt(2.*sig_cost**2))/2.
    p_cost_iso = np.ones(cost_grid.size)/2.

    return chi_grid,cost_grid,p_chi,p_cost_peak,p_cost_iso
    
def get_lvk_z(nTraces,m1_ref=20,nGridpoints=500):

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
        R_lnm1 = np.trapz(R_lnm1_q,q_grid,axis=0)

        # Interpolate to reference points
        R_z02_interpolator = interp1d(m1_grid,R_lnm1)
        R_z02_ref = R_z02_interpolator(m1_ref)

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
