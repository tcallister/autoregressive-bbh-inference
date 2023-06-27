Running the analyses
====================

Here, we detail how to rerun our code to recreate the data stored at https://zenodo.org/record/7600140.

Autoregressive mass and mass ratio inference
--------------------------------------------

To rerun our autoregressive inference on the primary mass and mass ratio distribution of BBHs, do the following:

.. code-block:: bash

    $ conda activate gwtc3-spin-studes
    $ cd code/
    $ python run_ar_lnm1_q.py

This script will launch `numpyro`, using the likelihood model :meth:`autoregressive_mass_models.ar_lnm1_q`.
This may take several hours to complete.
The output will be two files:

.. code-block:: bash

    final-ar_lnm1_q.cdf
    final-ar_lnm1_q_data.npy

.. warning::
    Before running, open `run_ar_lnm1_q.py` and change the output filepaths to suitable locations!
    These output files will, by default, be written to a directory that likely does not exist on your system. 

    Also, be aware that `file-ar_lnm1_q.cdf` is *large*, running a little over 20 GB.

The file :code:`final-ar_lnm1_q_data.npy` contains arrays holding all the mass and mass ratios in our dataset (including posterior samples and injections);
these are the values over which our AR processes are defined.
This file also contains information used to sort or reverse sort these values, used inside the likelihood function.
The file :code:`final-ar_lnm1_q.cdf`, in turn, contains the full inference output, including posterior samples and inference diagnostics.

Finally, we distill the full inference output into the restricted set of data that will be used for downstream analyses and figure generation.
This is done via

.. code-block:: bash

    $ cd ../data/
    $ python process_lnm1_q.py

This script will load the two files above and create a single (much smaller!) output file:

.. code-block:: bash

    ar_lnm1_q_summary.hdf

.. note::

    A notebook that demonstrates how to load in, inspect, and manipulate this output file can be found `here <https://github.com/tcallister/autoregressive-bbh-inference/blob/main/data/inspect_ar_lnm1_q_results.ipynb>`__

Redshift evolution of the merger rate
-------------------------------------

To rerun our autoregressive inference on the BBH merger rate evolution with redshift, do the following: 

.. code-block:: bash

    $ conda activate gwtc3-spin-studes
    $ cd code/
    $ python run_ar_z.py

This script will launch `numpyro`, using the likelihood model :meth:`autoregressive_redshift_models.ar_mergerRate`.
This may take several hours to complete.
The output will be two files:

.. code-block:: bash

    final-ar_z.cdf
    final-ar_z_data.npy

.. warning::
    Before running, open `run_ar_z.py` and change the output filepaths to suitable locations!
    These output files will, by default, be written to a directory that likely does not exist on your system. 

    Also, be aware that `file-ar_z.cdf` is *large*, running at about 15 GB.

The file :code:`final-ar_z_data.npy` contains arrays holding all the redshifts in our dataset (including posterior samples and injections);
these are the values over which our AR process is defined.
This file also contains information used to sort or reverse sort these values, used inside the likelihood function.
The file :code:`final-ar_z.cdf`, in turn, contains the full inference output, including posterior samples and inference diagnostics.

Finally, we distill the full inference output into the restricted set of data that will be used for downstream analyses and figure generation.
This is done via

.. code-block:: bash

    $ cd ../data/
    $ python process_z.py

This script will load the two files above and create a single (much smaller!) output file:

.. code-block:: bash

    ar_z_summary.hdf

.. note::

    A notebook that demonstrates how to load in, inspect, and manipulate this output file can be found `here <https://github.com/tcallister/autoregressive-bbh-inference/blob/main/data/inspect_ar_z_results.ipynb>`__

Component spin magnitudes and tilts
-----------------------------------

To rerun our autoregressive inference on the BBH component spin distribution, do the following:

.. code-block:: bash

    $ conda activate gwtc3-spin-studes
    $ cd code/
    $ python run_ar_chi_cost.py

This script will launch `numpyro`, using the likelihood model :meth:`autoregressive_spin_models.ar_spinMagTilt`.
This may take several hours to complete.
The output will be two files:

.. code-block:: bash

    final-ar_chi_cost.cdf
    final-ar_chi_cost_data.npy

.. warning::
    Before running, open `run_ar_chi_cost.py` and change the output filepaths to suitable locations!
    These output files will, by default, be written to a directory that likely does not exist on your system. 

    Also, be aware that `file-ar_chi_cost.cdf` is *large*, running a little over 40 GB.

The file :code:`final-ar_chi_cost_data.npy` contains arrays holding all the spin magnitudes and cosine tilts in our dataset (including posterior samples and injections);
these are the values over which our AR processes are defined.
This file also contains information used to sort or reverse sort these values, used inside the likelihood function.
The file :code:`final-ar_chi_cost.cdf`, in turn, contains the full inference output, including posterior samples and inference diagnostics.

Finally, we distill the full inference output into the restricted set of data that will be used for downstream analyses and figure generation.
This is done via

.. code-block:: bash

    $ cd ../data/
    $ python process_chi_cost.py

This script will load the two files above and create a single (much smaller!) output file:

.. code-block:: bash

    ar_chi_cost_summary.hdf

.. note::

    A notebook that demonstrates how to load in, inspect, and manipulate this output file can be found `here <https://github.com/tcallister/autoregressive-bbh-inference/blob/main/data/inspect_ar_chi_cost_results.ipynb>`__

Effective inspiral and precessing spins
---------------------------------------

To rerun our autoregressive inference on the BBH effective spin distribution, do the following:

.. code-block:: bash

    $ conda activate gwtc3-spin-studes
    $ cd code/
    $ python run_ar_Xeff_Xp.py

This script will launch `numpyro`, using the likelihood model :meth:`autoregressive_spin_models.ar_Xeff_Xp`.
This may take several hours to complete.
The output will be two files:

.. code-block:: bash

    final-ar_Xeff_Xp.cdf
    final-ar_Xeff_Xp_data.npy

.. warning::
    Before running, open `run_ar_Xeff_Xp.py` and change the output filepaths to suitable locations!
    These output files will, by default, be written to a directory that likely does not exist on your system. 

    Also, be aware that `file-ar_Xeff_Xp.cdf` is *large*, running a little over 25 GB.

The file :code:`final-ar_Xeff_Xp_data.npy` contains arrays holding all the effective inspiral and precessing spins in our dataset (including posterior samples and injections);
these are the values over which our AR processes are defined.
This file also contains information used to sort or reverse sort these values, used inside the likelihood function.
The file :code:`final-ar_Xeff_Xp.cdf`, in turn, contains the full inference output, including posterior samples and inference diagnostics.

Finally, we distill the full inference output into the restricted set of data that will be used for downstream analyses and figure generation.
This is done via

.. code-block:: bash

    $ cd ../data/
    $ python process_Xeff_Xp.py

This script will load the two files above and create a single (much smaller!) output file:

.. code-block:: bash

    ar_Xeff_Xp_summary.hdf

.. note::

    A notebook that demonstrates how to load in, inspect, and manipulate this output file can be found `here <https://github.com/tcallister/autoregressive-bbh-inference/blob/main/data/inspect_ar_Xeff_Xp_results.ipynb>`__

Analysis of synthetic catalogs
------------------------------

In Appendix C of our paper, we perform a series of inference exercises on mock data sets consistent with several different populations.
To repeat this analysis, first regenerate the mock data sets by opening and running the notebook `mock-data/gen_fake_samples.ipynb <https://github.com/tcallister/autoregressive-bbh-inference/blob/main/mock-data/gen_fake_samples.ipynb>`__.
This will generate four files:

.. code-block:: bash

    mock-data/gaussian_samples_varyingUncertainty.npy
    mock-data/spike_samples_varyingUncertainty.npy
    mock-data/gaussian_spike_samples_varyingUncertainty.npy
    mock-data/half_normal_samples_varyingUncertainty

Each file contains a dictionary with 300 elements, corresponding to mock posteriors for 300 simulated observations from each respective population.
After these files are generated, inference on the first 69 events of each sample is accomplished by running each of the following scripts:

.. code-block:: bash

    mock-data/run_gaussian_varyingUncertainty_069.py
    mock-data/run_spike_varyingUncertainty_069.py
    mock-data/run_gaussian_spike_varyingUncertainty_069.py
    mock-data/run_half_normal_varyingUncertainty_069.py

The result will accordingly be four pairs of output files, each pair of the form

.. code-block:: bash

    mock-data/ar_gaussian_varyingUncertainty_069.cdf
    mock-data/ar_gaussian_varyingUncertainty_069_data.npy

As above, the `.cdf` files will contain the full inference output, including diagnostics and hyperposterior samples.
The `.npy` files, in turn, will contain the union of all posterior samples and injections comprising our dataset, and information needed to sort/unsort them to the AR(1) process posteriors.

When each analysis is complete, the results are distilled into a set of much smaller output files using

.. code-block:: bash

    $ cd mock-data/
    $ python process_mock_populations.py

This script will produce a final set of four files:

.. code-block:: bash

    mock-data/ar_gaussian_varyingUncertainty_069_summary.hdf
    mock-data/ar_spike varyingUncertainty_069_summary.hdf
    mock-data/ar_gaussian_spike_varyingUncertainty_069_summary.hdf
    mock-data/ar_half_normal_varyingUncertainty_069_summary.hdf

These are the files that will be used for subsequent analysis and figure generation.

.. note::

    A notebook that demonstrates how to load in, inspect, and manipulate these output file can be found `here <https://github.com/tcallister/autoregressive-bbh-inference/blob/main/mock-data/inspect_mock_results.ipynb>`__
