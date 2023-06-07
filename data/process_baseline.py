import arviz as az
import numpy as np
import sys
import h5py
sys.path.append('./../code/')
from utilities import *

# Load inference results
inference_data = az.from_netcdf("baseline.cdf")
samps = az.extract(inference_data,var_names=["R20","kappa",\
                "alpha","mu_m1","sig_m1","log_f_peak","mMin","mMax","log_dmMin","log_dmMax","bq",\
                "mu_chi","logsig_chi","sig_cost",\
                "nEff_inj_per_event","min_log_neff"])

R_ref = samps.R20.values
kappa = samps.kappa.values
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

# Create hdf5 file and write posterior samples
hfile = h5py.File('baseline_summary.hdf','w')
posterior = hfile.create_group('posterior')
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
posterior.create_dataset('mu_chi',data=mu_chi)
posterior.create_dataset('logsig_chi',data=logsig_chi)
posterior.create_dataset('sig_cost',data=sig_cost)
posterior.create_dataset('nEff_inj_per_event',data=nEff_inj_per_event)
posterior.create_dataset('min_log_neff',data=min_log_neff)

# Add some metadata
hfile.attrs['Created_by'] = "process_baseline.py"
hfile.attrs['Source_code'] = "https://github.com/tcallister/autoregressive-bbh-inference"

hfile.close()
