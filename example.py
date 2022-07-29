#!/usr/bin/env python

import numpy as np
import pylab as plt
import hydra

import numpy.fft as fft
from scipy.sparse.linalg import cg, LinearOperator
import pyuvsim
import time, os

np.random.seed(11)

# Simulation settings
Nptsrc = 70
Ntimes = 30
Nfreqs = 20
Nants = 15
Niters = 10

sigma_noise = 0.5

hera_latitude = -30.7215 * np.pi / 180.0

# Check that output directory exists
if not os.path.exists("./output"):
    os.makedirs("./output")

#-------------------------------------------------------------------------------
# (1) Simulate some data
#-------------------------------------------------------------------------------

# Simulate some data
times = np.linspace(0.2, 0.5, Ntimes)
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
print("pstrc amps (input):", ptsrc_amps[:5])

# Beams
beams = [pyuvsim.analyticbeam.AnalyticBeam('gaussian', diameter=14.)
         for ant in ants]

# Run a simulation
t0 = time.time()
_sim_vis = hydra.vis_simulator.simulate_vis(
        ants=ant_pos,
        fluxes=fluxes,
        ra=ra,
        dec=dec,
        freqs=freqs*1e6, # MHz -> Hz
        lsts=times,
        beams=beams,
        polarized=False,
        precision=2,
        latitude=hera_latitude,
        use_feed="x",
    )
print("Simulation took %3.2f sec" % (time.time() - t0))

# Allocate computed visibilities to only the requested baselines (saves memory)
model0 = hydra.extract_vis_from_sim(ants, antpairs, _sim_vis)
del _sim_vis # save some memory

# Define gains and gain perturbations
gains = (1. + 0.j) * np.ones((Nants, Nfreqs, Ntimes), dtype=model0.dtype)
#delta_g = np.array([0.5*np.sin(times[np.newaxis,:] \
#                                 * freqs[:,np.newaxis]/100.)
#                    for ant in ants])

# Generate gain fluctuations from FFT basis
frate = np.fft.fftfreq(times.size, d=times[1] - times[0])
tau = np.fft.fftfreq(freqs.size, d=freqs[1] - freqs[0])
_delta_g = 100. \
         * np.fft.ifft2(np.exp(-0.5 * (((frate[np.newaxis,:]-9.)/2.)**2.
                                     + ((tau[:,np.newaxis] - 0.05)/0.03)**2.)))
delta_g = np.array([_delta_g for ant in ants])

# Apply gains to model
data = model0.copy()
hydra.apply_gains(data, gains * (1. + delta_g), ants, antpairs, inline=True)

# Plot input (simulated) gain perturbation for ant 0
vminmax = np.max(delta_g[0].real)
plt.matshow(delta_g[0].real, vmin=-vminmax, vmax=vminmax)
plt.colorbar()
plt.savefig("output/gain_000_xxxxx.png")

# Plot input (simulated) visibility model
vminmax_vis = np.max(model0[0,:,:].real)
plt.matshow(model0[0,:,:].real, vmin=-vminmax_vis, vmax=vminmax_vis)
plt.colorbar()
plt.savefig("output/data_model_000_xxxxx.png")

# Add noise
data += sigma_noise * np.sqrt(0.5) \
      * (  1.0 * np.random.randn(*data.shape) \
         + 1.j * np.random.randn(*data.shape))


#-------------------------------------------------------------------------------
# (2) Set up Gibbs sampler
#-------------------------------------------------------------------------------

# Get initial visibility model guesses (use the actual baseline model for now)
# This SHOULD NOT include gain factors of any kind
current_data_model = model0.copy()

# Initial gain perturbation guesses
current_delta_gain = np.zeros(gains.shape, dtype=model0.dtype)

# Initial point source amplitude factor
current_ptsrc_a = np.ones(ra.size)

# Precompute visibility projection operator (without gain factors) for ptsrc
# amplitude sampling step. NOTE: This has to be updated within the Gibbs loop
# if other components of the visibility model are being sampled
t0 = time.time()
vis_proj_operator0 = hydra.ptsrc_sampler.calc_proj_operator(
                                ra=ra,
                                dec=dec,
                                fluxes=fluxes,
                                ant_pos=ant_pos,
                                antpairs=antpairs,
                                freqs=freqs,
                                times=times,
                                beams=beams,
                                latitude=hera_latitude
)
print("Precomp. ptsrc proj. operator took %3.2f sec" % (time.time() - t0))

# Set priors and auxiliary information
# FIXME: amp_prior_std is a prior around amp=0 I think, so can skew things low!
noise_var = (sigma_noise)**2. * np.ones(data.shape)
inv_noise_var = 1. / noise_var
gain_pspec_sqrt = 0.1 * np.ones((gains.shape[1], gains.shape[2]))
amp_prior_std = 0.1 * np.ones(Nptsrc)
A_real, A_imag = hydra.gain_sampler.proj_operator(ants, antpairs)
gain_shape = gains.shape
N_gain_params = 2 * gains.shape[0] * gains.shape[1] * gains.shape[2]


# FIXME: Check that model0 == vis_proj_operator0 @ ones

#-------------------------------------------------------------------------------
# (3) Iterate Gibbs sampler
#-------------------------------------------------------------------------------

# Iterate the Gibbs sampler
print("="*60)
print("Starting Gibbs sampler (%d iterations)" % Niters)
print("="*60)
for n in range(Niters):
    print("-"*60)
    print(">>> Iteration %4d / %4d" % (n+1, Niters))
    print("-"*60)

    #---------------------------------------------------------------------------
    # (A) Gain sampler
    #---------------------------------------------------------------------------

    # current_data_model DOES NOT include gbar_i gbar_j^* factor, so we need
    # to apply it here to calculate the residual
    ggv = hydra.apply_gains(current_data_model,
                            gains,
                            ants,
                            antpairs,
                            inline=False)
    resid = data - ggv

    b = hydra.gain_sampler.flatten_vector(
            hydra.gain_sampler.construct_rhs(resid,
                                             inv_noise_var,
                                             gain_pspec_sqrt,
                                             A_real,
                                             A_imag,
                                             ggv,
                                             realisation=True)
        )

    def gain_lhs_operator(x):
        return hydra.gain_sampler.flatten_vector(
                    hydra.gain_sampler.apply_operator(
                        hydra.gain_sampler.reconstruct_vector(x, gain_shape),
                                         inv_noise_var,
                                         gain_pspec_sqrt,
                                         A_real,
                                         A_imag,
                                         ggv)
                             )

    # Build linear operator object
    gain_lhs_shape = (N_gain_params, N_gain_params)
    gain_linear_op = LinearOperator(matvec=gain_lhs_operator,
                                    shape=gain_lhs_shape)

    # Solve using Conjugate Gradients
    t0 = time.time()
    x_soln, convergence_info = cg(gain_linear_op, b)
    print("(A) Gain sampler took %3.2f sec" % (time.time() - t0))

    # Reshape solution into complex array and multiply by S^1/2 to get set of
    # Fourier coeffs of the actual solution for the frac. gain perturbation
    x_soln = hydra.gain_sampler.reconstruct_vector(x_soln, gain_shape)
    x_soln = hydra.gain_sampler.apply_sqrt_pspec(gain_pspec_sqrt, x_soln)

    # x_soln is a set of Fourier coefficients, so transform to real space
    # (ifft gives Fourier -> data space)
    xgain = np.zeros_like(x_soln)
    for k in range(xgain.shape[0]):
        xgain[k, :, :] = fft.ifft2(x_soln[k, :, :])

    print("    Gain sample:", xgain[0,0,0], xgain.shape)
    np.save("output/delta_gain_%05d" % n, x_soln)
    plt.matshow(xgain[0].real, vmin=-vminmax, vmax=vminmax)
    plt.colorbar()
    plt.savefig("output/gain_000_%05d.png" % n)

    # Update gain model with latest solution (in real space)
    current_delta_gain = xgain


    #---------------------------------------------------------------------------
    # (B) Point source amplitude sampler
    #---------------------------------------------------------------------------
    # Get the projection operator with most recent gains applied
    t0 = time.time()
    proj = vis_proj_operator0.copy()
    for k, bl in enumerate(antpairs):
        ant1, ant2 = bl
        i1 = np.where(ants == ant1)[0][0]
        i2 = np.where(ants == ant2)[0][0]
        proj[k,:,:,:] *= (gains * (1. + current_delta_gain))[i1,:,:,np.newaxis] \
                       * (gains * (1. + current_delta_gain))[i2,:,:,np.newaxis].conj()
    print("(B) Applying gains to ptsrc proj. operator took %3.2f sec" \
          % (time.time() - t0))

    # Precompute the point source matrix operator
    t0 = time.time()
    ptsrc_precomp_mat = hydra.ptsrc_sampler.precompute_op(proj, noise_var)
    print("(B) Precomp. ptsrc matrix operator took %3.2f sec" \
          % (time.time() - t0))

    # Construct current state of model (residual from amplitudes = 1)
    resid = data.copy() \
          - ( proj.reshape((-1, Nptsrc)) @ np.ones_like(amp_prior_std) ).reshape(current_data_model.shape)

    # Construct RHS of linear system
    bsrc = hydra.ptsrc_sampler.construct_rhs(resid.flatten(),
                                             noise_var.flatten(),
                                             amp_prior_std,
                                             proj,
                                             realisation=True)

    ptsrc_lhs_shape = (Nptsrc, Nptsrc)
    def ptsrc_lhs_operator(x):
        return hydra.ptsrc_sampler.apply_operator(x,
                                                  noise_var.flatten(),
                                                  amp_prior_std,
                                                  ptsrc_precomp_mat)

    # Build linear operator object
    ptsrc_linear_op = LinearOperator(matvec=ptsrc_lhs_operator,
                                     shape=ptsrc_lhs_shape)

    # Solve using Conjugate Gradients
    t0 = time.time()
    x_soln, convergence_info = cg(ptsrc_linear_op, bsrc)
    print("(B) Point source sampler took %3.2f sec" % (time.time() - t0))
    x_soln *= amp_prior_std # we solved for x = S^-1/2 s, so recover s
    print("    Example soln:", x_soln[:5]) # this is fractional deviation from assumed amplitude, so should be close to 0
    np.save("output/ptsrc_amp_%05d" % n, x_soln)

    # Update visibility model with latest solution (does not include any gains)
    # Applies projection operator to ptsrc amplitude vector
    current_data_model = ( vis_proj_operator0.reshape((-1, Nptsrc)) @ (1. + x_soln) ).reshape(current_data_model.shape)

    # Plot visibility waterfalls for current model
    plt.matshow(current_data_model[0,:,:].real, vmin=-vminmax_vis, vmax=vminmax_vis)
    plt.colorbar()
    plt.savefig("output/data_model_000_%05d.png" % n)

    # Close all figures made in this iteration
    plt.close('all')
