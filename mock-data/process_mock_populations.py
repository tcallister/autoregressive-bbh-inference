import arviz as az
import numpy as np
import sys
import h5py
sys.path.append('./../code/')
from utilities import *

def process_and_save(cdf_name):

    # Load inference results
    inference_data = az.from_netcdf(cdf_name)
    samps = az.extract(inference_data,var_names=["R20","fs","ar_std","ar_tau",\
                    "nEff_inj_per_event","min_log_neff"])

    R_ref = samps.R20.values
    fs = samps.fs.values
    ar_std = samps.ar_std.values
    ar_tau = samps.ar_tau.values
    nEff_inj_per_event = samps.nEff_inj_per_event.values
    min_log_neff = samps.min_log_neff.values

    # Also extract complete set of spin magnitude and tilt values over which AR process is defined
    sample_file = "{0}_data.npy".format(cdf_name.split(".")[0])
    sample_data = np.load(sample_file,allow_pickle=True)[()]
    chi_values = sample_data['all_samples']

    # For ease of plotting and storage, coarse-grain by a factor of 20
    fs_reduced = fs[::50,:]
    chi_values_reduced = chi_values[::50]

    # Compute rate density over mock spin space
    dR_dchis = R_ref[np.newaxis,:]*fs_reduced

    # Create hdf5 file and write posterior samples
    hfile_name = "/Volumes/LaCie/cca/autoregressive-bbh-inference-data-resubmission/for-data-release/{0}_summary.hdf".format(cdf_name.split(".")[0])
    hfile = h5py.File(hfile_name,'w')
    posterior = hfile.create_group('posterior')
    posterior.create_dataset('chis',data=chi_values_reduced)
    posterior.create_dataset('fs',data=fs_reduced)
    posterior.create_dataset('dR_dchis',data=dR_dchis)
    posterior.create_dataset('ar_std',data=ar_std)
    posterior.create_dataset('ar_tau',data=ar_tau)
    posterior.create_dataset('R_ref',data=R_ref)
    posterior.create_dataset('nEff_inj_per_event',data=nEff_inj_per_event)
    posterior.create_dataset('min_log_neff',data=min_log_neff)

    # Add some metadata
    hfile.attrs['Created_by'] = "process_mock_populations.py"
    hfile.attrs['Downloadable_from'] = "10.5281/zenodo.8087858"
    hfile.attrs['Source_code'] = "https://github.com/tcallister/autoregressive-bbh-inference"

    hfile.close()

process_and_save("ar_gaussian_varyingUncertainty_069.cdf")
process_and_save("ar_spike_varyingUncertainty_069.cdf")
process_and_save("ar_gaussian_spike_varyingUncertainty_069.cdf")
process_and_save("ar_half_normal_varyingUncertainty_069.cdf")
