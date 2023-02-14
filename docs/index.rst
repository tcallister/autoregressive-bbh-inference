.. autoregressive-bbh-inference documentation master file, created by
   sphinx-quickstart on Tue Jan 31 15:03:20 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to autoregressive-bbh-inference's documentation!
========================================================

This page details the code used to produce the results presented in *A parameter-free tour of the binary black hole population*, which can be accessed at

https://github.com/tcallister/autoregressive-bbh-inference/

The datasets comprising our results, as well as the input data necessary to reproduce our work, are hosted on Zenodo:

https://doi.org/10.5281/zenodo.7600140

In this paper, we modeled the distributions of binary black hole masses, spins, and redshifts using flexible *autoregressive processes*,
seeking to robustly identify the set of physical features appearing in these distributions without the need for strongly-parametrized population models.
Using the GWTC-3 catalog of LIGO/Virgo compact binary detections, we see the following:

**Binary black hole masses**

.. image:: images/figure_03a.pdf
    :height: 250

The BBH mass spectrum exhibits two maxima near 10 and 35 solar masses, with a continuum that likely steepens above 40 solar masses.
The result is well-described by a superposition between a broken power law and two Gaussian peaks.

**The black hole merger rate as a function of redshift**

.. image:: images/figure_06_cropped.pdf
    :height: 250

The BBH merger rate was higher in the past than it is today, although this growth occurs in a way that may not necessarily be well-described by a power law in :math:`1+z`.

**Black hole component spins**

.. image:: images/figure_08_cropped.pdf
    :height: 250

Black holes exhibit a unimodal distribution of spin magnitudes, concentrated at low values but with no special features at :math:`\chi=0`.
They also show a broad range of spin-orbit misalignment angles (including tilt angles beyond :math:`90^\circ`), although an isotropic distribution is also disfavored.

**Binary black hole effective spins**

.. image:: images/figure_12_cropped.pdf
    :height: 250

Similarly, the distribution of effective inspiral spins among merging binaries is unimodal, encompasses negative :math:`\chi_\mathrm{eff}` values, but also disfavors symmetry about zero.
Effective precessing spins are poorly measured, but favor a broad distribution in :math:`\chi_\mathrm{p}`.


Contents:

.. toctree::
    :maxdepth: 1

    getting-started
    making-figures
    running-the-analyses

* :ref:`modindex`
* :ref:`search`
