import numpy as np
import json

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
        'Rtot':plpeak_Rtot
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
