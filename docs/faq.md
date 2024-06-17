# Frequently asked questions

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
