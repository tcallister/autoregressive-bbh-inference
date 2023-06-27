import arviz as az
import numpy as np
import h5py

# Load inference results
inference_data = az.from_netcdf("/Volumes/LaCie/cca/autoregressive-bbh-inference-data-resubmission/ar_lnm1_q.cdf")
samps = az.extract(inference_data,var_names=["R20","kappa","f_lnm1s","f_qs","ar_lnm1_std","ar_lnm1_tau","ar_q_std","ar_q_tau",\
                        "mu_chi","logsig_chi","sig_cost","nEff_inj_per_event","min_log_neff"])
R_ref = samps.R20.values
kappa = samps.kappa.values
f_lnm1s = samps.f_lnm1s.values
f_qs = samps.f_qs.values
ar_lnm1_std = samps.ar_lnm1_std.values
ar_lnm1_tau = samps.ar_lnm1_tau.values
ar_q_std = samps.ar_q_std.values
ar_q_tau = samps.ar_q_tau.values
mu_chi = samps.mu_chi.values
logsig_chi = samps.logsig_chi.values
sig_cost = samps.sig_cost.values
nEff_inj_per_event = samps.nEff_inj_per_event.values
min_log_neff = samps.min_log_neff.values

# Also extract complete set of mass and mass ratio values over which AR process is defined
lnm1_q_data = np.load('/Volumes/LaCie/cca/autoregressive-bbh-inference-data-resubmission/ar_lnm1_q_data.npy',allow_pickle=True)[()]
m1_values = np.exp(lnm1_q_data['all_lnm1_samples'])
q_values = lnm1_q_data['all_q_samples']

# For ease of plotting and storage, coarse-grain by a factor of 50
f_lnm1s_reduced = f_lnm1s[::50,:]
f_qs_reduced = f_qs[::50,:]
m1_values_reduced = m1_values[::50]
q_values_reduced = q_values[::50]

# Compute rate density over lnm1, at fixed q=1
# Note that z=0.2 is already built into the definition of R_ref
f_q_equal_1 = f_qs[-1,:]
dR_dlnm1s = R_ref[np.newaxis,:]*f_lnm1s_reduced*f_q_equal_1[np.newaxis,:]

# Compute rate density over q, now at fixed m1=20 and z=0.2
# Note that z=0.2 is already built into the definition of R_ref
f_lnm1_equal_ln20 = f_lnm1s[np.argmin(np.abs(m1_values-20.)),:]
dR_dqs = R_ref[np.newaxis,:]*f_lnm1_equal_ln20[np.newaxis,:]*f_qs_reduced

# Create hdf5 file and write posterior samples
hfile = h5py.File('/Volumes/LaCie/cca/autoregressive-bbh-inference-data-resubmission/for-data-release/ar_lnm1_q_summary.hdf','w')
posterior = hfile.create_group('posterior')
posterior.create_dataset('m1s',data=m1_values_reduced)
posterior.create_dataset('qs',data=q_values_reduced)
posterior.create_dataset('f_lnm1s',data=f_lnm1s_reduced)
posterior.create_dataset('f_qs',data=f_qs_reduced)
posterior.create_dataset('dR_dlnm1s',data=dR_dlnm1s)
posterior.create_dataset('dR_dqs',data=dR_dqs)
posterior.create_dataset('ar_lnm1_std',data=ar_lnm1_std)
posterior.create_dataset('ar_lnm1_tau',data=ar_lnm1_tau)
posterior.create_dataset('ar_q_std',data=ar_q_std)
posterior.create_dataset('ar_q_tau',data=ar_q_tau)
posterior.create_dataset('kappa',data=kappa)
posterior.create_dataset('mu_chi',data=mu_chi)
posterior.create_dataset('logsig_chi',data=logsig_chi)
posterior.create_dataset('sig_cost',data=sig_cost)
posterior.create_dataset('R_ref',data=R_ref)
posterior.create_dataset('nEff_inj_per_event',data=nEff_inj_per_event)
posterior.create_dataset('min_log_neff',data=min_log_neff)

# Add some metadata
hfile.attrs['Created_by'] = "process_lnm1_q.py"
hfile.attrs['Downloadable_from'] = "10.5281/zenodo.8087858"
hfile.attrs['Source_code'] = "https://github.com/tcallister/autoregressive-bbh-inference"

hfile.close()

