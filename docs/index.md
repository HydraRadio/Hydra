# Hydra
An ultra-high dimensional Bayesian sampler code for 21cm arrays

<img src="../hydra_logo.png" alt="Hydra logo" width="200px"/>

## What is Hydra?

Hydra is a code that implements a statistical model of complex visibility data from radio telescope 
arrays that target the 21cm line from neutral hydrogen. Once the 21cm signal, foreground emission, 
and instrumental effects like primary beams have been incorporated, hundreds of thousands of parameters 
are required to fully describe the model.

Hydra uses a technique called **Gibbs sampling** to efficiently draw samples from the joint posterior 
distribution of *all* the parameters. It is implemented in Python and is compatible with the data 
formats used by the HERA pipeline, such as _UVData_ and _UVBeam_.

## What is Gibbs sampling?

Gibbs sampling is a way of breaking up the joint posterior distribution for a set of parameters into a 
set of conditional distributions that can then be sampled from iteratively. As new samples of parameter 
values are drawn from each conditional distribution, the values of those parameters are updated, and 
then conditioned on in subsequent sub-steps of each iteration through the full set of conditional 
distributions. 

The idea is to choose conditional distributions that are tractable; multi-variate Gaussian distributions 
are particularly useful, as they can be sampled from exactly (e.g. without needing any Markov Chain steps) 
even for very large sets of parameters, by solving a particular linear system that we call the **Gaussian 
Constrained Realisation** (GCR) equation. This looks a lot like the *Wiener filter* equation for Gaussian 
data with noise and prior covariance matrices, plus some fluctuation terms that allow us to draw a 
different statistical sample each time.

A particularly powerful property of the GCR approach is that it can handle **missing data** in a natural 
way. In regions of missing data (e.g. due to RFI flagging), the prior terms takes over, permitting a 
well-defined model value to exist at all data points, not just the unflagged ones. This is extremely 
helpful in analyses that perform harmonic transforms, e.g. Fourier analysis, as there is no need to 
construct special estimators (like LSSA) or in-paint the data with an assumed model. The additional 
uncertainty associated with the missing data is naturally incorporated as we draw samples; there will 
typically be a larger variation in the predicted model values in the missing data regions. Depending on 
the form of the model, however, nearby data that are not missing can help constrain the possible values 
within the masked region (e.g. because the model has some level of smoothness). The main point here is 
that the sampler can draw plausible realisations of the data model for every data point, whether it was 
observed or not.

## What can Hydra do?

Hydra can construct models of the visibility data from generic radio interferometer arrays, given a model 
of the emission on the sky, and instrumental properties such as the primary beam patterns and array 
layout. It can then efficiently draw samples from the statistical distribution of the model parameters 
given the data and some prior assumptions, even though the parameter space can be vast.

The joint posterior distribution is broken up into several conditional distributions, which are the 
'heads' of the Hydra. Each head typically handles a different sub-model of the full data model, e.g. 
we have the following components:

* Gain sampler (using a linearised gain perturbation model for each antenna gain)
* Beam sampler (using a Fourier-Bessel model for each E-field beam)
* Point source sampler (using a simple power-law SED model)
* Diffuse emission sampler (using a spherical harmonic basis)
* Power spectrum sampler (using a standard bandpower representation of the power spectrum)

All in all, there can be several hundreds of thousands of parameters for a typical array size, sky model 
size and so on. Some of these parameters could be removed from the sampler via analytic marginalisation, 
but we avoid this so that we can inspect the full joint posterior distirbution as a way of analysing 
issues with the model fits.

Some of the 'heads' of Hydra can be used independently. For example, we use the power spectrum sampler 
with an alternative data-driven foreground model (rather than an explicit physical forward model of the 
sky) in order to analyse calibrated HERA data. This is the _hydra-pspec_ code.

At present, Hydra is parallelised in a basic way, meaning that it can only run on a single system (e.g. 
a single HPC node). There are plans to make a distributed version that can scale to much larger problem 
sizes, but for now the main limitation is the amount of RAM available on each node.

## Who is behind Hydra?

Hydra is a project based out of Phil Bull's research group at the Jodrell Bank Centre for Astrophysics 
(University of Manchester). It is funded by a European Research Council Starting Grant, 
[Illuminating the darkness with precision maps of neutral hydrogen across cosmic time](https://cordis.europa.eu/project/id/948764). 
The team currently consists of Jacob Burba, Sohini Dutta, Hugh Garsden, Katrine Glasscock, Fraser Kennedy, 
Mike Wilensky, and Zheng Zhang.

Contributions and requests for features are welcome, as are bug reports.
