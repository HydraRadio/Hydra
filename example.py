#!/usr/bin/env python

import numpy as np
import pylab as plt
import hydra

from scipy.sparse.linalg import cg, LinearOperator
import pyuvsim
import time

np.random.seed(10)

# Simulation settings
Nptsrc = 40
Ntimes = 30
Nfreqs = 20
Nants = 10
Niters = 10

#-------------------------------------------------------------------------------
# (1) Simulate some data
#-------------------------------------------------------------------------------

# Simulate some data
times = np.linspace(0.2, 1.8, Ntimes)
freqs = np.linspace(100., 120., Nfreqs)

ants = np.arange(Nants)
antpairs = []
for i in range(len(ants)):
    for j in range(i, len(ants)):
        if i != j:
            # Exclude autos
            antpairs.append((i,j))
ant_pos = {ant: [14.7*(ant % 5) + 0.5*14.7*(ant // 5),
                 14.7*(ant // 5),
                 0.] for ant in ants} # hexagon-like packing
ants1, ants2 = list(zip(*antpairs))

# Generate random point source locations
# RA goes from [0, 2 pi] and Dec from [-pi, +pi].
ra = np.random.uniform(low=0.4, high=1.4, size=Nptsrc)
dec = np.random.uniform(low=-0.6, high=-0.4, size=Nptsrc) # close to HERA stripe

# Generate fluxes
beta_ptsrc = -2.7
ptsrc_amps = 10.**np.random.uniform(low=-1., high=2., size=Nptsrc)
fluxes = ptsrc_amps[:,np.newaxis] * ((freqs / 100.)**beta_ptsrc)[np.newaxis,:]

# Beams
beams = [pyuvsim.analyticbeam.AnalyticBeam('gaussian', diameter=14.)
         for ant in ants]

# Run a simulation
t0 = time.time()
_model0 = hydra.vis_simulator.simulate_vis(
        ants=ant_pos,
        fluxes=fluxes,
        ra=ra,
        dec=dec,
        freqs=freqs*1e6, # MHz -> Hz
        lsts=times,
        beams=beams,
        polarized=False,
        precision=2,
        latitude=-30.7215 * np.pi / 180.0,
        use_feed="x",
    )
print("Simulation took %3.2f sec" % (time.time() - t0))

# Allocate computed visibilities to only the requested baselines (saves memory)
model0 = np.zeros((len(antpairs), Nfreqs, Ntimes), dtype=_model0.dtype)
for i, bl in enumerate(antpairs):
    idx1 = np.where(ants == bl[0])[0][0]
    idx2 = np.where(ants == bl[1])[0][0]
    model0[i,:,:] = _model0[:,:,idx1,idx2]
print(model0.shape)

# Plot antenna positions
# xyz = np.array(list(ant_pos.values()))
#
# plt.plot(xyz[:,0], xyz[:,1], 'r.')
# plt.xlim((-10., 100.))
# plt.ylim((-10., 100.))
# plt.show()

# Define gains and gain perturbations
gains = (1.1 + 0.1j) * np.ones((Nants, Nfreqs, Ntimes), dtype=model0.dtype)
delta_g = np.array([0.05j*np.sin(ant * times[np.newaxis,:] \
                                 * freqs[:,np.newaxis]/100.)
                    for ant in ants])
data = model0.copy()

# Apply gains
for k, bl in enumerate(antpairs):
    ant1, ant2 = bl
    i1 = np.where(ants == ant1)[0][0]
    i2 = np.where(ants == ant2)[0][0]
    data[k,:,:] *= (gains[i1] + delta_g[i1]) \
                 * (gains[i2] + delta_g[i2]).conj()

# Add noise
data += np.sqrt(0.5) * (  1.0 * np.random.randn(*data.shape) \
                        + 1.j * np.random.randn(*data.shape))


#-------------------------------------------------------------------------------
# (2) Iterate Gibbs sampler
#-------------------------------------------------------------------------------

# Get initial model guesses (use the actual baseline model for now)
current_model = model0.copy()
for k, bl in enumerate(antpairs):
    ant1, ant2 = bl
    i1 = np.where(ants == ant1)[0][0]
    i2 = np.where(ants == ant2)[0][0]
    current_model[k,:,:] *= (gains[i1]) * (gains[i2]).conj()

# Initial gain perturbation guesses
current_delta_gain = np.zeros(gains.shape, dtype=model0.dtype)

# Initial point source amplitude perturbation
current_ptsrc_a = np.zeros(ra.size)

# Set priors and auxiliary information
noise_var = np.ones(data.shape)
pspec_sqrt = 0.2 * np.ones((gains.shape[1], gains.shape[2]))
A_real, A_imag = hydra.gain_sampler.proj_operator(ants, antpairs)
gain_shape = gains.shape
N_gain_params = 2 * gains.shape[0] * gains.shape[1] * gains.shape[2]

# Iterate the Gibbs sampler
for n in range(Niters):

    #---------------------------------------------------------------------------
    # (A) Gain sampler
    #---------------------------------------------------------------------------
    t0 = time.time()
    resid = data - current_model
    b = hydra.gain_sampler.flatten_vector(
            hydra.gain_sampler.construct_rhs(resid,
                                             noise_var,
                                             pspec_sqrt,
                                             A_real,
                                             A_imag,
                                             current_model,
                                             realisation=True)
        )

    def gain_lhs_operator(x):
        return hydra.gain_sampler.flatten_vector(
                    hydra.gain_sampler.apply_operator(
                        hydra.gain_sampler.reconstruct_vector(x, gain_shape),
                                         noise_var,
                                         pspec_sqrt,
                                         A_real,
                                         A_imag,
                                         current_model)
                             )

    # Build linear operator object
    gain_lhs_shape = (N_gain_params, N_gain_params)
    gain_linear_op = LinearOperator(matvec=gain_lhs_operator, shape=gain_lhs_shape)
    print("Gain precomp. took %3.2f sec" % (time.time() - t0))

    # Solve using Conjugate Gradients
    t0 = time.time()
    x_soln, convergence_info = cg(gain_linear_op, b)
    #x_soln, convergence_info
    print("Gain sampler took %3.2f sec" % (time.time() - t0))
    print(x_soln[0], x_soln.shape)

    #---------------------------------------------------------------------------
    # (B) Point source amplitude sampler
    #---------------------------------------------------------------------------
    """
    # Prior on amplitudes
    amp_prior_std = 0.1*np.ones(Nptsrc)

    # Construct RHS of linear system
    b = construct_rhs(resid.flatten(), noise_var.flatten(), amp_prior_std, vis_proj_operator, realisation=False)

    lhs_shape = (Nptsrc, Nptsrc)
    def lhs_operator(x):
        return apply_operator2(x, noise_var.flatten(), amp_prior_std, vis_proj_operator, precomp_mat)

    # Build linear operator object
    linear_op = LinearOperator(matvec=lhs_operator, shape=lhs_shape)

    # Solve using Conjugate Gradients
    t0 = time.time()
    x_soln, convergence_info = cg(linear_op, b)
    print("Run took %3.2f sec" % (time.time() - t0))
    x_soln, convergence_info
    """
    
