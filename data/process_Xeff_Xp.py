import arviz as az
import numpy as np
import sys
import h5py
sys.path.append('./../code/')
from utilities import *

# Load inference results
inference_data = az.from_netcdf("/Volumes/LaCie/cca/autoregressive-bbh-inference-data-resubmission/ar_Xeff_Xp.cdf")
samps = az.extract(inference_data,var_names=["R20","kappa","f_Xeffs","f_Xps","ar_Xeff_std","ar_Xeff_tau","ar_Xp_std","ar_Xp_tau",\
                "alpha","mu_m1","sig_m1","log_f_peak","mMin","mMax","log_dmMin","log_dmMax","bq",\
                "nEff_inj_per_event","min_log_neff"])

R_ref = samps.R20.values
kappa = samps.kappa.values
f_Xeffs = samps.f_Xeffs.values
f_Xps = samps.f_Xps.values
ar_Xeff_std = samps.ar_Xeff_std.values
ar_Xeff_tau = samps.ar_Xeff_tau.values
ar_Xp_std = samps.ar_Xp_std.values
ar_Xp_tau = samps.ar_Xp_tau.values
alpha = samps.alpha.values
mu_m1 = samps.mu_m1.values
sig_m1 = samps.sig_m1.values
log_f_peak = samps.log_f_peak.values
mMin = samps.mMin.values
mMax = samps.mMax.values
log_dmMin = samps.log_dmMin.values
log_dmMax = samps.log_dmMax.values
bq = samps.bq.values
nEff_inj_per_event = samps.nEff_inj_per_event.values
min_log_neff = samps.min_log_neff.values

# Also extract complete set of spin magnitude and tilt values over which AR process is defined
Xeff_Xp_data = np.load('/Volumes/LaCie/cca/autoregressive-bbh-inference-data-resubmission/ar_Xeff_Xp_data.npy',allow_pickle=True)[()]
Xeff_values = Xeff_Xp_data['Xeff_allSamples']
Xp_values = Xeff_Xp_data['Xp_allSamples']

# For ease of plotting and storage, coarse-grain by a factor of 50
f_Xeffs_reduced = f_Xeffs[::50,:]
f_Xps_reduced = f_Xps[::50,:]
Xeff_values_reduced = Xeff_values[::50]
Xp_values_reduced = Xp_values[::50]

# Compute rate density over Xeff, at fixed Xp=0.1, q=1, m1=20, and z=0.2
# Note that R_ref is already defined to be the (mean) rate density at z=0.2 and m1=20
# We additionally need to multiply by m1=20 to make this a rate per *log* mass
m_ref = 20.
f_Xp_equal_01 = f_Xps[np.argmin(np.abs(Xp_values-0.1)),:]
f_q_equal_1 = (1.+bq)/(1. - (tmp_min/m_ref)**(1.+bq))
dR_dXeffs = R_ref[np.newaxis,:]*f_Xeffs_reduced*f_Xp_equal_01[np.newaxis,:]*f_q_equal_1*m_ref

# Compute rate density over Xp, now at fixed Xeff=0.05, q=1, m1=20, and z=0.2
f_Xeff_equal_005 = f_Xeffs[np.argmin(np.abs(Xeff_values-0.05)),:]
dR_dXps = R_ref[np.newaxis,:]*f_Xeff_equal_005[np.newaxis,:]*f_Xps_reduced*f_q_equal_1*m_ref

# Create hdf5 file and write posterior samples
hfile = h5py.File('/Volumes/LaCie/cca/autoregressive-bbh-inference-data-resubmission/for-data-release/ar_Xeff_Xp_summary.hdf','w')
posterior = hfile.create_group('posterior')
posterior.create_dataset('Xeffs',data=Xeff_values_reduced)
posterior.create_dataset('Xps',data=Xp_values_reduced)
posterior.create_dataset('f_Xeffs',data=f_Xeffs_reduced)
posterior.create_dataset('f_Xps',data=f_Xps_reduced)
posterior.create_dataset('dR_dXeffs',data=dR_dXeffs)
posterior.create_dataset('dR_dXps',data=dR_dXps)
posterior.create_dataset('ar_Xeff_std',data=ar_Xeff_std)
posterior.create_dataset('ar_Xeff_tau',data=ar_Xeff_tau)
posterior.create_dataset('ar_Xp_std',data=ar_Xp_std)
posterior.create_dataset('ar_Xp_tau',data=ar_Xp_tau)
posterior.create_dataset('kappa',data=kappa)
posterior.create_dataset('alpha',data=alpha)
posterior.create_dataset('mu_m1',data=mu_m1)
posterior.create_dataset('sig_m1',data=sig_m1)
posterior.create_dataset('log_f_peak',data=log_f_peak)
posterior.create_dataset('mMin',data=mMin)
posterior.create_dataset('mMax',data=mMax)
posterior.create_dataset('log_dmMin',data=log_dmMin)
posterior.create_dataset('log_dmMax',data=log_dmMax)
posterior.create_dataset('bq',data=bq)
posterior.create_dataset('R_ref',data=R_ref)
posterior.create_dataset('nEff_inj_per_event',data=nEff_inj_per_event)
posterior.create_dataset('min_log_neff',data=min_log_neff)

# Add some metadata
hfile.attrs['Created_by'] = "process_Xeff_Xp.py"
hfile.attrs['Downloadable_from'] = "10.5281/zenodo.8087858"
hfile.attrs['Source_code'] = "https://github.com/tcallister/autoregressive-bbh-inference"

hfile.close()
