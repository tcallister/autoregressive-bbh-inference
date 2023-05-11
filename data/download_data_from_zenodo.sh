#!/bin/bash

# Download and unzip
curl https://zenodo.org/record/7616096/files/autoregressive-bbh-inference-data.zip --output "autoregressive-bbh-inference-data.zip"
unzip autoregressive-bbh-inference-data.zip

# Move input data to ../code/input/
mv sampleDict_FAR_1_in_1_yr.pickle ../input/
mv injectionDict_FAR_1_in_1.pickle ../input/
mv posteriors_gaussian_spin_samples_FAR_1_in_1.json ../input/
mv o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json ../input/

# Remove original zip files and annoying Mac OSX files
rm autoregressive-bbh-inference-data.zip
rmdir __MACOSX/
