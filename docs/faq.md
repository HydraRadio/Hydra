# Frequently asked questions

## Basic concepts and terminology

#### What is Gibbs sampling?
[Gibbs sampling](https://en.wikipedia.org/wiki/Gibbs_sampling) is a Monte Carlo method for drawing samples of 
parameters from a posterior probability distribution for many parameters. It works by breaking up the 
posterior distribution, which may be highly non-trivial (e.g. having large numbers of parameters or a complicated 
functional form), into a set of conditional distributions. Subsets of parameters can then be drawn from their 
corresponding conditional distributions while conditioning on (fixing) all other parameters. The conditional 
distributions can often be chosen such that they are much more tractable than the full posterior distribution. 
The Gibbs sampler then iterates through the conditional distributions, taking turns to draw one subset of 
parameter values while holding the others fixed until a new sample has been drawn for all free parameters. The 
process is then repeated to build up a Markov chain of samples of the parameter values, which can be shown to 
eventually converge to a set of samples from the true joint posterior distribution.

#### What is a Gaussian Constrained Realisation (GCR)?
This refers to drawing a sample for a set of parameters from a multivariate Gaussian distribution that is 
conditioned on (constrained by) some data and a model. There is an exact algorithm for drawing samples from 
a multivariate Gaussian distribution, even for very large (high dimensional) parameter spaces. The algorithm 
involves solving a linear system that resembles the Wiener filter, but with added random fluctuation terms. 
Each sample of the set of parameters can be used to construct a model prediction for what the data look like 
that is _statistically consistent_ with the data, i.e. the prediction is a valid "realisation" of the model 
that is consistent with the data. This includes data that have missing values, e.g. masked regions. As such, 
the constrained realisations can be used to predict plausible values of the data in the masked regions that 
are consistent with the values in the unmasked regions.

## Model assumptions

#### Linearisation of the complex gains
The gains are linearised so that all the gain parameters can be sampled from a large multi-variate Gaussian 
distribution simultaneously, instead of needing to do a round-robin of per-antenna conditional distributions. 
This has many advantages in terms of computation, but requires the validity of the linearisation to be 
respected to ensure sensible results.

## Sampler maths

#### What is 'realification'
Because complex-valued vectors can be a bit tricky to deal with numerically, e.g. due to linear solver routines 
that don't use complex conjugates where they should, we often split systems of `N` complex parameters into 
systems of `2 N` real parameters. See Appendix B of [Kennedy et al. (2023)](https://doi.org/10.3847/1538-4365/acc324) 
and Appendix A of [Glasscock et al. (2024)](https://arxiv.org/abs/2403.13766) for more information.

#### Where have the factors of 1/2 gone in the Gaussian probability distributions in the papers?
The 'missing' factor of 1/2 comes about when complex vector notation is being used. If this factor is included, the 
covariance matrix of the real and imaginary parts would be wring by a factor of 2. 

#### Why isn't there a white noise (omega) term that is scaled by the inverse square root of the foreground covariance?

## Code implementation

#### Why does the MPI CG solver require a square number of workers?
The matrix operator `A` and right-hand side vector `b` are split up into blocks. Each worker 'owns' a block of `A`. 
This scheme is explained in the docstring of `hydra.linear_solver.setup_mpi_blocks()`. It simplifies the 
calculations and inter-process communication if we can keep the blocks the same square shape. This is important in 
parts where multiplication by a transpose is needed for example. Enforcing squareness results in some zero-padding 
of the blocks at the edges, and for now we don't allow idle workers who don't own a block, so the total number of 
processes has to be square too.
