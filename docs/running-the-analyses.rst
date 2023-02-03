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
