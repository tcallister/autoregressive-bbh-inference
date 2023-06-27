import arviz as az
import numpy as np
import sys
import h5py
sys.path.append('./../code/')
from utilities import *

# Load inference results
inference_data = az.from_netcdf("/Volumes/LaCie/cca/autoregressive-bbh-inference-data-resubmission/ar_chi_cost.cdf")
samps = az.extract(inference_data,var_names=["R20","kappa","f_chis","f_cost","ar_chi_std","ar_chi_tau","ar_cost_std","ar_cost_tau",\
                "alpha","mu_m1","sig_m1","log_f_peak","mMin","mMax","log_dmMin","log_dmMax","bq",\
                "nEff_inj_per_event","min_log_neff"])

R_ref = samps.R20.values
kappa = samps.kappa.values
f_chis = samps.f_chis.values
f_costs = samps.f_cost.values
ar_chi_std = samps.ar_chi_std.values
ar_chi_tau = samps.ar_chi_tau.values
ar_cost_std = samps.ar_cost_std.values
ar_cost_tau = samps.ar_cost_tau.values
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
chi_cost_data = np.load('/Volumes/LaCie/cca/autoregressive-bbh-inference-data-resubmission/ar_chi_cost_data.npy',allow_pickle=True)[()]
chi_values = chi_cost_data['chi_allSamples']
cost_values = chi_cost_data['cost_allSamples']

# For ease of plotting and storage, coarse-grain by a factor of 50
f_chis_reduced = f_chis[::50,:]
f_costs_reduced = f_costs[::50,:]
chi_values_reduced = chi_values[::50]
cost_values_reduced = cost_values[::50]

# Compute rate density over spin magnitudes, at fixed cost1=cost2=1, q=1, m1=20, and z=0.2
# Specifically we will quote the rate along the chi1=chi2 line
# This necessitates *two* powers of both f_chis and f_costs, one per component spin
# Note that R_ref is already defined to be the (mean) rate density at z=0.2 and m1=20
# We additionally need to multiply by m1=20 to make this a rate per *log* mass
m_ref = 20.
f_cost_equal_1 = f_costs[-1,:]
f_q_equal_1 = (1.+bq)/(1. - (tmp_min/m_ref)**(1.+bq))
dR_dchis = R_ref[np.newaxis,:]*f_chis_reduced**2.*f_cost_equal_1[np.newaxis,:]**2.*f_q_equal_1*m_ref

# Compute rate density over cost, now at fixed chi1=chi2=0.1, q=1, m1=20, and z=0.2
# As above, specifically compute the rate along the cost1=cost2 line
# Note that R_ref is already defined to be the (mean) rate density at z=0.2 and m1=20
f_chi_equal_01 = f_chis[np.argmin(np.abs(chi_values-0.1)),:]
dR_dcosts = R_ref[np.newaxis,:]*f_chi_equal_01[np.newaxis,:]**2.*f_costs_reduced**2.*f_q_equal_1*20.

# Create hdf5 file and write posterior samples
hfile = h5py.File('/Volumes/LaCie/cca/autoregressive-bbh-inference-data-resubmission/for-data-release/ar_chi_cost_summary.hdf','w')
posterior = hfile.create_group('posterior')
posterior.create_dataset('chis',data=chi_values_reduced)
posterior.create_dataset('costs',data=cost_values_reduced)
posterior.create_dataset('f_chis',data=f_chis_reduced)
posterior.create_dataset('f_costs',data=f_costs_reduced)
posterior.create_dataset('dR_dchis',data=dR_dchis)
posterior.create_dataset('dR_dcosts',data=dR_dcosts)
posterior.create_dataset('ar_chi_std',data=ar_chi_std)
posterior.create_dataset('ar_chi_tau',data=ar_chi_tau)
posterior.create_dataset('ar_cost_std',data=ar_cost_std)
posterior.create_dataset('ar_cost_tau',data=ar_cost_tau)
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
hfile.attrs['Created_by'] = "process_chi_cost.py"
hfile.attrs['Downloadable_from'] = "10.5281/zenodo.8087858"
hfile.attrs['Source_code'] = "https://github.com/tcallister/autoregressive-bbh-inference"

hfile.close()
