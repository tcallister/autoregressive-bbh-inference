Running the analyses
====================

Here, we detail how to rerun our code to recreate the data stored at https://zenodo.org/record/7600141.

Autoregressive mass and mass ratio inference
--------------------------------------------

To rerun our autoregressive inference on the primary mass and mass ratio distribution of BBHs, do the following:

.. code-block:: bash

    $ conda activate gwtc3-spin-studes
    $ cd code/
    $ python run_ar_lnm1_q.py

This script will launch `numpyro`, using the likelihood model :meth:`autoregressive_mass_models.ar_lnm1_q` defined in `code/autoregressive-mass-models.py`.
This may take several hours to complete.
The output will be two files:

.. code-block:: bash

    final-ar_lnm1_q.cdf
    final-ar_lnm1_q_data.npy

.. warning::
    Before running, open `run_ar_lnm1_q.py` and change the output filepaths to suitable locations!
    These output files will, by default, be written to a directory that likely does not exist on your system. 

    Also, be aware that `file-ar_lnm1_q.cdf` is *large*, running a little over 20 GB.


