Getting started
===============

Setting up your environment
----------------------------

To make it as easy as possible to reproduce our results and/or figures, the `environment.yml` file can be used to build a conda environment containing the packages needed to run the code in this repository.
To set up this environment, do the following:

**Step 0**. Make sure you have conda installed. If not, see e.g. https://docs.conda.io/en/latest/miniconda.html

**Step 1**. Do the following:

.. code-block:: bash

    $ conda env create -f environment.yaml

This will create a new conda environment named *autoregressive-bbh-inference*

**Step 2**. To activate the new environment, do

.. code-block:: bash

    $ conda activate autoregressive-bbh-inference 

You can deactivate the environment using :code:`conda deactivate`

Downloading input files and inference results
---------------------------------------------

Datafiles containing the output of our inference codes are hosted on a few different Zenodo pages.
All data needed to regenerate figures and/or rerun our analyses can be found at https://doi.org/10.5281/zenodo.7600141.
To download this input/output data locally, you can do the following:

.. code-block:: bash

    $ cd data/
    $ . download_data_from_zenodo.sh

This script will populate the :code:`data/` directory with datafiles containing processed outputs of our analyses.
These output files can be inspected by running the jupyter notebooks also appearing in the :code:`data/` directory.
The script will also place several files in the :code:`input/` directory, which are needed to rerun analyses and/or regenerate figures.

The data downloaded via the above script is not the raw output of our inference, but has been post-processed and coarse-grained in order to minimize storage/memory requirements.
If you want access to the actual, underlying data, this can be found at the following set of Zenodo pages:
* XXXXX
* XXXXX
*Caution*: The datafiles hosted on these pages are pretty large (tens of GBs), so we recommend against downloading these unless you really need to!

