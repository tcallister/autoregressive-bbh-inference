import arviz as az
import numpy as np
import h5py
import sys
sys.path.append('./../code/')
from utilities import *

# Load inference results
inference_data = az.from_netcdf("ar_z_test.cdf")
samps = az.extract(inference_data,var_names=["R20","f_zs","ar_z_std","ar_z_tau",\
                "alpha","mu_m1","sig_m1","log_f_peak","mMin","mMax","log_dmMin","log_dmMax","bq",\
                "mu_chi","logsig_chi","sig_cost","nEff_inj_per_event","min_log_neff"])
R_ref = samps.R20.values
f_zs = samps.f_zs.values
ar_z_std = samps.ar_z_std.values
ar_z_tau = samps.ar_z_tau.values
alpha = samps.alpha.values
mu_m1 = samps.mu_m1.values
sig_m1 = samps.sig_m1.values
log_f_peak = samps.log_f_peak.values
mMin = samps.mMin.values
mMax = samps.mMax.values
log_dmMin = samps.log_dmMin.values
log_dmMax = samps.log_dmMax.values
bq = samps.bq.values
mu_chi = samps.mu_chi.values
logsig_chi = samps.logsig_chi.values
sig_cost = samps.sig_cost.values
nEff_inj_per_event = samps.nEff_inj_per_event.values
min_log_neff = samps.min_log_neff.values

# Also extract complete set of redshift values over which AR process is defined
z_data = np.load('ar_z_data_test.npy',allow_pickle=True)[()]
z_values = z_data['z_allSamples']

# For ease of plotting and storage, coarse-grain by a factor of 50
f_zs_reduced = f_zs[::50,:]
z_values_reduced = z_values[::50]

# Compute rate density over lnm1, at fixed q=1
# Note that R_ref is already defined as the merger rate at m1=20.
# Additionally multiply by m1 to convert to a rate per log mass
m1_ref = 20.
f_q_equal_1 = (1.+bq)/(1. - (tmp_min/m1_ref)**(1.+bq))
R_of_zs = R_ref[np.newaxis,:]*f_zs_reduced*f_q_equal_1[np.newaxis,:]*m1_ref

# Create hdf5 file and write posterior samples
hfile = h5py.File('ar_z_summary_test.hdf','w')
posterior = hfile.create_group('posterior')
posterior.create_dataset('zs',data=z_values_reduced)
posterior.create_dataset('f_zs',data=f_zs_reduced)
posterior.create_dataset('R_of_zs',data=R_of_zs)
posterior.create_dataset('ar_z_std',data=ar_z_std)
posterior.create_dataset('ar_z_tau',data=ar_z_tau)
posterior.create_dataset('mu_chi',data=mu_chi)
posterior.create_dataset('logsig_chi',data=logsig_chi)
posterior.create_dataset('sig_cost',data=sig_cost)
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
hfile.attrs['Created_by'] = "process_z.py"
hfile.attrs['Downloadable_from'] = "10.5281/zenodo.7600141"
hfile.attrs['Source_code'] = "https://github.com/tcallister/autoregressive-bbh-inference"

hfile.close()
