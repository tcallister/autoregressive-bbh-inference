Figure generation
=================

The :code:`figures/` directory contains jupyter notebooks that reproduce every figure appearing in our paper.
To regenerate any particular figure, first make sure you have locally downloaded data from Zenodo (or regenerated it yourself!).
See :ref:`Getting started` for more info about this. 

Once you've downloaded the necessary data, launch a jupyter session:

.. code-block:: bash
    
    $ conda activate autoregressive-bbh-inference
    $ cd figures/
    $ jupyter notebook

Then open the desired notebook from the list of .ipynb files and run all cells (*"Run All"* in the "Cell" menu). 
The notebook will generate and save the corresponding figure in .pdf format in the :code:`figures/` directory.

