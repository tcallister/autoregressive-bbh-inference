#!/bin/bash

# Download and unzip
curl https://zenodo.org/records/8087858/files/autoregressive-bbh-inference-data.zip --output "autoregressive-bbh-inference-data.zip"
unzip autoregressive-bbh-inference-data.zip

# Move input data to ../code/input/
mv sampleDict_FAR_1_in_1_yr.pickle ../input/
mv injectionDict_FAR_1_in_1.pickle ../input/
mv posteriors_gaussian_spin_samples_FAR_1_in_1.json ../input/
mv o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json ../input/

# Move results from mock data studies
mv ar_gaussian_varyingUncertainty_069_summary.hdf ../mock-data/
mv ar_spike_varyingUncertainty_069_summary.hdf ../mock-data/
mv ar_gaussian_spike_varyingUncertainty_069_summary.hdf ../mock-data/
mv ar_half_normal_varyingUncertainty_069_summary.hdf ../mock-data/

# Remove original zip files and annoying Mac OSX files
rm autoregressive-bbh-inference-data.zip
rmdir __MACOSX/
