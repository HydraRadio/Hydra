# Available "heads" of Hydra
Hydra is a Gibbs sampler, which draws samples from the joint posterior distribution of the data model parameters by breaking it up into a set of conditional distributions and iterating through them. From each conditional distribution, samples are drawn from a subspace of the full parameter space while all other parameters are fixed (conditioned on). As the sampler iterates through the "Gibbs scheme", made up of several such conditional sampling steps, it draws new samples from every subspace, so by the end of one full iteration, a new sample has been drawn for all free parameters. This is then repeated to draw as many samples of the full paraneter space as are required.

This structure allows us to sample the parameters of different components of the data model separately, conditioning on all the other components' parameters in each sub-step of the Gibbs scheme. In this sense, Hydra is _modular_: the conditional sampling steps don't need to know anything about one another except for what the most recent state of the data model was from the previous step. This makes it relatively straightforward to add new data model components and sampling steps as long as they handle the necessary inputs (the current state of the data model) and outputs (generally the updated data model, or a component thereof) in a standardised way.

For brevity, each conditional sampling step of the Gibbs scheme is referred to as one of the "heads" of Hydra (a reference to the many heads of the code's mythological namesake).

We now summarise some of the currently available heads of Hydra.

### Linearised gains
The per-antenna complex gains are modelled as a function of frequency and local sidereal time (LST). The gain model has a fixed prefactor (the 'fiducial' gain model for each antenna) times a small perturbation that can be sampled. The perturbation is constructed from a discrete set of 2D Fourier modes (i.e. modes in delay and fringe-rate, the Fourier duals of frequency and LST), which needs not be complete or orthonormal.

Gains always appear quadratically in the data model, i.e. all visibility data points involve the product of two antenna gains. We do not know of an efficient way to sample directly from the conditional distribution of a large number of parameters that are quadratic however. At best, we could condition on the parameters for all but one antenna, which would result in a multivariate Gaussian conditional distribution for the parameters of that antenna. We could then iterate through all antennas. This is onerous for large antenna arrays with potentially hundreds of antennas, and risks seriously increasing the correlation length of the Monte Carlo chain, as it would take a long time to explore correlated directions in the joint antenna gain parameter space

Instead, we make an approximation: that the gain perturbations are sufficiently small that the quadratic term of the antenna gain product can be neglected, i.e. in the data model, we keep only terms that are linear in the antenna gains. As long as this approximation is valid, we have a data model that is linear in all the gain parameters, and so we can write down a joint conditional distribution for all the gain parameters that is a multivariate Gaussian. Multivariate Gaussians can be sampled from directly bu solving a linear system (which looks like the Wiener filter plus some random fluctuation terms).

We do not assume redundancy of the array in our gain model. Nor do we apply a smoothing or similar; a smooth gain model is naturally obtained by including only Fourier modes with low delay/fringe rate values. Since arbitrarily complicated structure can be included in the fiducial gain model, which is fixed for each antenna, particular features of the gains at higher Fourier wavenumbers can be included in there if necessary, without greatly increasing the number of parameters. Care must be taken to understand how the small perturbations will modulate the structure of the fiducial gain model however (e.g. structure can appear at beat frequencies).

### Point source amplitudes


### Discrete regions of diffuse maps



### Spherical harmonic modes



### Spherical harmonic angular power spectrum


### The 21cm field on a Cartesian grid


### The spherically-averaged 21cm power spectrum
