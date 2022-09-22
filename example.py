#!/usr/bin/env python

import numpy as np
import pylab as plt
import hydra

import numpy.fft as fft
from scipy.sparse.linalg import cg, LinearOperator
from scipy.signal import blackmanharris
import pyuvsim
import time, os
from hydra.vis_utils import flatten_vector, reconstruct_vector, timing_info, \
                            build_hex_array, get_flux_from_ptsrc_amp, convert_to_tops

import argparse

description = "Example Gibbs sampling of the joint posterior of several analysis " \
              "parameters in 21-cm power spectrum estimation from a simulated " \
              "visibility data set"
parser = argparse.ArgumentParser(description=description)
parser.add_argument("-s", type=int, action="store", default=12,
                    required=False, dest="seed",
                    help="Sets the random seed for the simulation.")
parser.add_argument("--no_gains", action="store_true",
                    required=False, dest="no_sample_gains",
                    help="Do not sample the gains (omit if desire is to sample gains).")
parser.add_argument("--no_vis", action="store_true",
                    required=False, dest="no_sample_vis",
                    help="Do not sample the visibilities "
                         "(omit if desire is to sample visibilities).")
parser.add_argument("--no_ptsrc", action="store_true",
                    required=False, dest="no_sample_ptsrc",
                    help="Do not sample the point source amplitudes "
                         "(omit if desire is to sample point source amplitudes).")
parser.add_argument("--no_beam", action="store_true",
                    required=False, dest="no_sample_beam",
                    help="Do not sample the beam (omit if desire is to sample the beam)")
parser.add_argument("--no_stats", action="store_true",
                    required=False, dest="no_calculate_stats",
                    help="Do not calculate statistics about the sampling results.")
parser.add_argument("--no_diagnostics", action="store_true",
                    required=False, dest="no_output_diagnostics",
                    help="Do not output diagnostics.")
parser.add_argument("--no_timing", action="store_true", required=False,
                    dest="no_save_timing_info", help="Do not save timing info.")
parser.add_argument("--no_plotting", action="store_true",
                    required=False, dest="no_plotting",
                    help="Do not make plots.")
parser.add_argument("--Nptsrc", type=int, action="store", default=100,
                    required=False, dest="Nptsrc",
                    help="Number of point sources to use in simulation (and model)")
parser.add_argument("--Ntimes", type=int, action="store", default=30,
                    required=False, dest="Ntimes",
                    help="Number of times to use in simulation")
parser.add_argument("--Nfreqs", type=int, action="store", default=60,
                    required=False, dest="Nfreqs",
                    help="Number of frequencies to use in simulation")
parser.add_argument("--Niters", type=int, action="store", default=100,
                    required=False, dest="Niters",
                    help="Number of joint samples to gather")
parser.add_argument("--sigma_noise", type=float, action="store",
                    default=0.05, required=False, dest="sigma_noise",
                    help="Strength of the noise")
parser.add_argument("--beam_nmax", type=int, action="store",
                    default=10, required=False, dest="beam_nmax",
                    help="Maximum radial degree of the Zernike basis for the beams")
args = parser.parse_args()

# eliminate the double negatives here.
SAMPLE_GAINS = not args.no_sample_gains
SAMPLE_VIS = not args.no_sample_vis
SAMPLE_PTSRC_AMPS = not args.no_sample_ptsrc
SAMPLE_BEAM = not args.no_sample_beam
CALCULATE_STATS = not args.no_calculate_stats
OUTPUT_DIAGNOSTICS = not args.no_output_diagnostics
SAVE_TIMING_INFO = not args.no_save_timing_info
PLOTTING = not args.no_plotting

# Simulation settings -- want some shorter variable names
Nptsrc = args.Nptsrc
Ntimes = args.Ntimes
Nfreqs = args.Nfreqs
#Nants = 15
Niters = args.Niters
beam_nmax = args.beam_nmax

sigma_noise = args.sigma_noise

hera_latitude = -30.7215 * np.pi / 180.0

# Check that output directory exists
if not os.path.exists("./output"):
    os.makedirs("./output")
ftime = "./output/timing.dat"

#-------------------------------------------------------------------------------
# (1) Simulate some data
#-------------------------------------------------------------------------------

# Simulate some data
times = np.linspace(0.2, 0.5, Ntimes)
freqs = np.linspace(100., 120., Nfreqs)

#ant_pos = build_hex_array(hex_spec=(3,4), d=14.6)
ant_pos = build_hex_array(hex_spec=(3,4), d=14.6)
ants = np.array(list(ant_pos.keys()))
Nants = len(ants)
print("Nants =", Nants)

#ants = np.arange(Nants)
antpairs = []
for i in range(len(ants)):
    for j in range(i, len(ants)):
        if i != j:
            # Exclude autos
            antpairs.append((i,j))
#ant_pos = {ant: [14.7*(ant % 5) + 0.5*14.7*(ant // 5),
#                 14.7*(ant // 5),
#                 0.] for ant in ants} # hexagon-like packing


ants1, ants2 = list(zip(*antpairs))

# Generate random point source locations
# RA goes from [0, 2 pi] and Dec from [-pi, +pi].
ra = np.random.uniform(low=0.0, high=1.8, size=Nptsrc)
dec = np.random.uniform(low=-0.6, high=-0.4, size=Nptsrc) # close to HERA stripe

# Generate fluxes
beta_ptsrc = -2.7
ptsrc_amps = 10.**np.random.uniform(low=-1., high=2., size=Nptsrc)
fluxes = get_flux_from_ptsrc_amp(ptsrc_amps, freqs, beta_ptsrc)
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
timing_info(ftime, 0, "(0) Simulation", time.time() - t0)

# Allocate computed visibilities to only the requested baselines (saves memory)
model0 = hydra.extract_vis_from_sim(ants, antpairs, _sim_vis)
del _sim_vis # save some memory

# Define gains and gain perturbations
gains = (1. + 0.j) * np.ones((Nants, Nfreqs, Ntimes), dtype=model0.dtype)

# Generate gain fluctuations from FFT basis
frate = np.fft.fftfreq(times.size, d=times[1] - times[0])
tau = np.fft.fftfreq(freqs.size, d=freqs[1] - freqs[0])

random_phase = np.exp(1.j*np.random.uniform(low=0., high=2.*np.pi, size=Nants))
random_amp = 1. + 0.05*np.random.randn(Nants)
_delta_g = 5. \
         * np.fft.ifft2(np.exp(-0.5 * (((frate[np.newaxis,:]-9.)/2.)**2.
                                     + ((tau[:,np.newaxis] - 0.05)/0.03)**2.)))
delta_g = np.array([random_amp[i] * _delta_g *random_phase[i] for i in range(Nants)],
                    dtype=model0.dtype)

# Apply a Blackman-Harris window to apodise the edges
#window = blackmanharris(model0.shape[1], sym=True)[np.newaxis,:,np.newaxis] \
#       * blackmanharris(model0.shape[2], sym=True)[np.newaxis,np.newaxis,:]
window = 1. # no window for now

# Apply gains to model
data = model0.copy() * window # FIXME
hydra.apply_gains(data, gains * (1. + delta_g), ants, antpairs, inline=True)

# Plot input (simulated) gain perturbation for ant 0
if PLOTTING:
    vminmax = np.max(delta_g[0].real)
    plt.subplot(121)
    plt.matshow(delta_g[0].real, vmin=-vminmax, vmax=vminmax, fignum=False, aspect='auto')
    plt.colorbar()
    plt.subplot(122)
    plt.matshow(delta_g[0].imag, vmin=-vminmax, vmax=vminmax, fignum=False, aspect='auto')
    plt.colorbar()
    plt.gcf().set_size_inches((10., 4.))
    plt.savefig("output/delta_g_true_000.png")

    # Plot input (simulated) visibility model
    vminmax_vis = np.max(model0[0,:,:].real)
    plt.matshow(model0[0,:,:].real, vmin=-vminmax_vis, vmax=vminmax_vis)
    plt.colorbar()
    plt.savefig("output/data_model_000_xxxxx.png")

# Add noise
data += sigma_noise * np.sqrt(0.5) \
      * (  1.0 * np.random.randn(*data.shape) \
         + 1.j * np.random.randn(*data.shape))

# Plot data (including noise)
if PLOTTING:
    plt.matshow(data[0,:,:].real, vmin=-vminmax_vis, vmax=vminmax_vis)
    plt.colorbar()
    plt.savefig("output/data_000.png")

#-------------------------------------------------------------------------------
# (2) Set up Gibbs sampler
#-------------------------------------------------------------------------------

# Get initial visibility model guesses (use the actual baseline model for now)
# This SHOULD NOT include gain factors of any kind
current_data_model = 1.*model0.copy() * window # FIXME

# Initial gain perturbation guesses
#current_delta_gain = np.zeros(gains.shape, dtype=model0.dtype)
# FIXME
current_delta_gain = np.zeros_like(delta_g)
# FIXME: Trying to fix the amplitude to the correct value
#current_delta_gain[:,0,0] += fft.fft2(delta_g[0])[0,0] # FIXME FIXME

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
timing_info(ftime, 0, "(0) Precomp. ptsrc proj. operator", time.time() - t0)

# Set priors and auxiliary information
# FIXME: amp_prior_std is a prior around amp=0 I think, so can skew things low!
noise_var = (sigma_noise)**2. * np.ones(data.shape)
inv_noise_var = window / noise_var

gain_pspec_sqrt = 0.1 * np.ones((gains.shape[1], gains.shape[2]))
gain_pspec_sqrt[0,0] = 1e-2 # FIXME: Try to fix the zero point?

# FIXME: Gain smoothing via priors
#ii, jj = np.meshgrid(np.arange(gains.shape[2]), np.arange(gains.shape[1]))
#gain_pspec_sqrt *= np.exp(-0.5 * np.sqrt(ii**2. + jj**2.)/1.**2.)

#plt.matshow(gain_pspec_sqrt, aspect='auto')
#plt.colorbar()
#plt.show()
#exit()

amp_prior_std = 0.1 * np.ones(Nptsrc)
amp_prior_std[19] = 1e-3 # FIXME
vis_pspec_sqrt = 0.01 * np.ones((1, Nfreqs, Ntimes)) # currently same for all visibilities
vis_group_id = np.zeros(len(antpairs), dtype=int) # index 0 for all

A_real, A_imag = hydra.gain_sampler.proj_operator(ants, antpairs)

gain_shape = gains.shape
N_gain_params = 2 * gains.shape[0] * gains.shape[1] * gains.shape[2]
N_vis_params = 2 * data.shape[0] * data.shape[1] * data.shape[2]


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
    t0iter = time.time()

    #---------------------------------------------------------------------------
    # (A) Gain sampler
    #---------------------------------------------------------------------------
    if SAMPLE_GAINS:
        # Current_data_model DOES NOT include gbar_i gbar_j^* factor, so we need
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
        timing_info(ftime, n, "(A) Gain sampler", time.time() - t0)


        # Reshape solution into complex array and multiply by S^1/2 to get set of
        # Fourier coeffs of the actual solution for the frac. gain perturbation
        x_soln = hydra.vis_utils.reconstruct_vector(x_soln, gain_shape)
        x_soln = hydra.gain_sampler.apply_sqrt_pspec(gain_pspec_sqrt, x_soln)

        # x_soln is a set of Fourier coefficients, so transform to real space
        # (ifft gives Fourier -> data space)
        xgain = np.zeros_like(x_soln)
        for k in range(xgain.shape[0]):
            xgain[k, :, :] = fft.ifft2(x_soln[k, :, :])

        print("    Gain sample:", xgain[0,0,0], xgain.shape)
        np.save("output/delta_gain_%05d" % n, x_soln)

        if PLOTTING:
            for i in range(len(ants)):
                plt.subplot(121)
                plt.matshow(xgain[i].real, vmin=-vminmax, vmax=vminmax,
                            fignum=False, aspect='auto')
                plt.colorbar()
                plt.subplot(122)
                plt.matshow(xgain[i].imag, vmin=-vminmax, vmax=vminmax,
                            fignum=False, aspect='auto')
                plt.colorbar()
                plt.gcf().set_size_inches((10., 4.))
                plt.savefig("output/delta_g_%03d_%05d.png" % (i, n))

            # Residual with true gains (abs)
            plt.matshow(np.abs(xgain[0]) - np.abs(delta_g[0]),
                        vmin=-np.max(np.abs(delta_g[0])),
                        vmax=np.max(np.abs(delta_g[0])),
                        aspect='auto')
            plt.colorbar()
            plt.title("%05d" % n)
            plt.gcf().set_size_inches((6., 4.))
            plt.savefig("output/gain_resid_amp_000_%05d.png" % n)

            # Residual with true gains (real)
            plt.matshow(np.real(xgain[0]) - np.real(delta_g[0]),
                        vmin=-np.max(np.real(delta_g[0])),
                        vmax=np.max(np.real(delta_g[0])),
                        aspect='auto')
            plt.colorbar()
            plt.title("%05d" % n)
            plt.gcf().set_size_inches((6., 4.))
            plt.savefig("output/gain_resid_real_000_%05d.png" % n)

            # Residual with true gains (imag)
            plt.matshow(np.imag(xgain[0]) - np.imag(delta_g[0]),
                        vmin=-np.max(np.imag(delta_g[0])),
                        vmax=np.max(np.imag(delta_g[0])),
                        aspect='auto')
            plt.colorbar()
            plt.title("%05d" % n)
            plt.gcf().set_size_inches((6., 4.))
            plt.savefig("output/gain_resid_imag_000_%05d.png" % n)

            # DEBUG
            # Compare imaginary parts of gains
            plt.subplot(121)
            plt.matshow(np.imag(xgain[0]),
                        vmin=-np.max(np.imag(delta_g[0])),
                        vmax=np.max(np.imag(delta_g[0])),
                        fignum=False, aspect='auto')
            plt.colorbar()

            plt.subplot(122)
            plt.matshow(np.imag(delta_g[0]),
                        vmin=-np.max(np.imag(delta_g[0])),
                        vmax=np.max(np.imag(delta_g[0])),
                        fignum=False, aspect='auto')
            plt.colorbar()
            plt.title("%05d" % n)
            plt.gcf().set_size_inches((10., 4.))
            plt.savefig("output/gain_resid_compare_imag_000_%05d.png" % n)

            # Compare real parts of gains
            plt.subplot(121)
            plt.matshow(np.imag(xgain[0]),
                        vmin=-np.max(np.imag(delta_g[0])),
                        vmax=np.max(np.imag(delta_g[0])),
                        fignum=False, aspect='auto')
            plt.colorbar()

            plt.subplot(122)
            plt.matshow(np.imag(delta_g[0]),
                        vmin=-np.max(np.imag(delta_g[0])),
                        vmax=np.max(np.imag(delta_g[0])),
                        fignum=False, aspect='auto')
            plt.colorbar()
            plt.title("%05d" % n)
            plt.gcf().set_size_inches((10., 4.))
            plt.savefig("output/gain_resid_compare_imag_000_%05d.png" % n)

        # Update gain model with latest solution (in real space)
        current_delta_gain = xgain


    #---------------------------------------------------------------------------
    # (B) Visibility sampler
    #---------------------------------------------------------------------------
    if SAMPLE_VIS:
        # Current_data_model DOES NOT include gbar_i gbar_j^* factor, so we need
        # to apply it here to calculate the residual
        ggv = hydra.apply_gains(current_data_model,
                                gains*(1.+current_delta_gain),
                                ants,
                                antpairs,
                                inline=False)
        resid = data - ggv

        # Construct RHS of linear system
        bvis = flatten_vector(
                    hydra.vis_sampler.construct_rhs(
                                               data=resid,
                                               inv_noise_var=inv_noise_var,
                                               sqrt_pspec=vis_pspec_sqrt,
                                               group_id=vis_group_id,
                                               gains=gains*(1.+current_delta_gain),
                                               ants=ants,
                                               antpairs=antpairs,
                                               realisation=True)
                                               )

        vis_lhs_shape = (N_vis_params, N_vis_params)
        def vis_lhs_operator(x):
            # Re-pack flattened x vector into complex vector of the right shape
            y = hydra.vis_sampler.apply_operator(reconstruct_vector(x, data.shape),
                                                 inv_noise_var=inv_noise_var,
                                                 sqrt_pspec=vis_pspec_sqrt,
                                                 group_id=vis_group_id,
                                                 gains=gains*(1.+current_delta_gain),
                                                 ants=ants,
                                                 antpairs=antpairs)
            return flatten_vector(y)

        # Build linear operator object
        vis_linear_op = LinearOperator(matvec=vis_lhs_operator,
                                       shape=vis_lhs_shape)

        # Solve using Conjugate Gradients
        t0 = time.time()
        x_soln, convergence_info = cg(vis_linear_op, bvis)
        timing_info(ftime, n, "(B) Visibility sampler", time.time() - t0)

        # Reshape solution into complex array and multiply by S^1/2 to get set of
        # Fourier coeffs of the actual solution for the frac. gain perturbation
        x_soln = hydra.vis_utils.reconstruct_vector(x_soln, data.shape)
        x_soln = hydra.vis_sampler.apply_sqrt_pspec(vis_pspec_sqrt,
                                                    x_soln,
                                                    vis_group_id,
                                                    ifft=True)

        print("    Vis sample:", x_soln[0,0,0], x_soln.shape)
        np.save("output/vis_%05d" % n, x_soln)

        if PLOTTING:
            plt.matshow(current_data_model[0].real + x_soln[0].real,
                        vmin=-vminmax_vis,
                        vmax=vminmax_vis)
            plt.colorbar()
            plt.title("%05d" % n)
            plt.savefig("output/vis_000_%05d.png" % n)

        # Update current state
        current_data_model = current_data_model + x_soln

        # Residual with true data model
        if PLOTTING:
            plt.subplot(121)
            plt.matshow(current_data_model[0].real - model0[0].real,
                        vmin=-0.5,
                        vmax=0.5,
                        fignum=False, aspect='auto')
            plt.colorbar()
            plt.title("%05d" % n)

            plt.subplot(122)
            plt.matshow(current_data_model[0].imag - model0[0].imag,
                        vmin=-0.5,
                        vmax=0.5,
                        fignum=False, aspect='auto')
            plt.colorbar()
            plt.gcf().set_size_inches((10., 4.))
            plt.savefig("output/resid_datamodel_000_%05d.png" % n)


    #---------------------------------------------------------------------------
    # (C) Point source amplitude sampler
    #---------------------------------------------------------------------------

    if SAMPLE_PTSRC_AMPS:

        # Get the projection operator with most recent gains applied
        t0 = time.time()
        proj = vis_proj_operator0.copy()
        for k, bl in enumerate(antpairs):
            ant1, ant2 = bl
            i1 = np.where(ants == ant1)[0][0]
            i2 = np.where(ants == ant2)[0][0]
            proj[k,:,:,:] *= (gains * (1. + current_delta_gain))[i1,:,:,np.newaxis] \
                           * (gains * (1. + current_delta_gain))[i2,:,:,np.newaxis].conj()
        timing_info(ftime, n, "(C) Applying gains to ptsrc proj. operator", time.time() - t0)

        # Precompute the point source matrix operator
        t0 = time.time()
        ptsrc_precomp_mat = hydra.ptsrc_sampler.precompute_op(proj, inv_noise_var)
        timing_info(ftime, n, "(C) Precomp. ptsrc matrix operator", time.time() - t0)

        # Construct current state of model (residual from amplitudes = 1)
        resid = data.copy() \
              - ( proj.reshape((-1, Nptsrc)) @ np.ones_like(amp_prior_std) ).reshape(current_data_model.shape)

        # Construct RHS of linear system
        bsrc = hydra.ptsrc_sampler.construct_rhs(resid.flatten(),
                                                 inv_noise_var.flatten(),
                                                 amp_prior_std,
                                                 proj,
                                                 realisation=True)

        ptsrc_lhs_shape = (Nptsrc, Nptsrc)
        def ptsrc_lhs_operator(x):
            return hydra.ptsrc_sampler.apply_operator(x,
                                                      amp_prior_std,
                                                      ptsrc_precomp_mat)

        # Build linear operator object
        ptsrc_linear_op = LinearOperator(matvec=ptsrc_lhs_operator,
                                         shape=ptsrc_lhs_shape)

        # Solve using Conjugate Gradients
        t0 = time.time()
        x_soln, convergence_info = cg(ptsrc_linear_op, bsrc)
        timing_info(ftime, n, "(C) Point source sampler", time.time() - t0)
        x_soln *= amp_prior_std # we solved for x = S^-1/2 s, so recover s
        print("    Example soln:", x_soln[:5]) # this is fractional deviation from assumed amplitude, so should be close to 0
        np.save("output/ptsrc_amp_%05d" % n, x_soln)

        # Plot point source amplitude perturbations
        if PLOTTING:
            plt.subplot(111)
            plt.plot(x_soln, 'r.')
            plt.axhline(0., ls='dashed', color='k')
            plt.axhline(-np.max(amp_prior_std), ls='dotted', color='gray')
            plt.axhline(np.max(amp_prior_std), ls='dotted', color='gray')
            plt.ylim((-1., 1.))
            plt.title("%05d" % n)
            plt.gcf().set_size_inches((5., 4.))
            plt.savefig("output/ptsrc_amp_%05d.png" % n)

        # Update visibility model with latest solution (does not include any gains)
        # Applies projection operator to ptsrc amplitude vector
        current_data_model = ( vis_proj_operator0.reshape((-1, Nptsrc)) @ (1. + x_soln) ).reshape(current_data_model.shape)

        # Plot visibility waterfalls for current model
        if PLOTTING:
            plt.matshow(current_data_model[0,:,:].real,
                        vmin=-vminmax_vis, vmax=vminmax_vis, aspect='auto')
            plt.colorbar()
            plt.title("%05d" % n)
            plt.gcf().set_size_inches((5., 4.))
            plt.savefig("output/data_model_000_%05d.png" % n)

    #---------------------------------------------------------------------------
    # (D) Beam sampler
    #---------------------------------------------------------------------------

    if SAMPLE_BEAM:
        # Have to have an initial guess and do some precompute
        if n == 0:
            # Make a copy of the data that is more convenient for the beam calcs.
            data_beam = hydra.beam_sampler.reshape_data_arr(data,
                                                            Nfreqs,
                                                            Ntimes,
                                                            Nants)
            # Doubles the autos, but we don't use them so it doesn't matter.
            # This makes it so we do not have to keep track of whether we are sampling
            # The beam coeffs or their conjugate!
            data_beam = data_beam + np.swapaxes(data_beam, -1, -2).conj()

            inv_noise_var_beam = hydra.beam_sampler.reshape_data_arr(inv_noise_var,
                                                                     Nfreqs,
                                                                     Ntimes,
                                                                     Nants)

            txs, tys, _ = convert_to_tops(ra, dec, times, hera_latitude)
            Zmatr = hydra.beam_sampler.construct_zernike_matrix(beam_nmax,
                                                                np.array(txs),
                                                                np.array(tys))

            # All the same, so just repeat (for now)
            beam_coeffs = np.array(Nants * \
                                   [hydra.beam_sampler.fit_zernike_to_beam(
                                                                     beams[0],
                                                                     freqs,
                                                                     Zmatr,
                                                                     txs,
                                                                     tys), ])
            beam_coeffs = np.swapaxes(beam_coeffs, 0, 3).astype(complex)
            ncoeffs = beam_coeffs.shape[0]


            amp_use = x_soln if SAMPLE_PTSRC_AMPS else ptsrc_amps
            flux_use = get_flux_from_ptsrc_amp(amp_use, freqs, beta_ptsrc)
            Mjk = hydra.beam_sampler.construct_Mjk(Zmatr, ant_pos, flux_use, ra, dec,
                                             freqs, times, polarized=False,
                                             latitude=hera_latitude)

            # Hardcoded parameters. Make variations smooth in time/freq.
            sig_freq = 0.5 * (freqs[-1] - freqs[0])
            sig_time = 0.5 * (times[-1] - times[0])
            cov_tuple = hydra.beam_sampler.make_prior_cov(freqs, times, ncoeffs, 1e-4, sig_freq,
                                                          sig_time, ridge=1e-6)
            cho_tuple = hydra.beam_sampler.do_cov_cho(cov_tuple, check_op=False)
            # Be lazy and just use the initial guess.
            coeff_mean = hydra.beam_sampler.split_real_imag(beam_coeffs[:, :, :, 0],
                                                            'vec')

        t0 = time.time()

        # Round robin loop through the antennas
        for ant_samp_ind in range(Nants):
            zern_trans = hydra.beam_sampler.get_zernike_to_vis(Mjk, beam_coeffs,
                                                     ant_samp_ind, Nants)
            cov_Tdag = hydra.beam_sampler.get_cov_Tdag(cho_tuple, zern_trans)

            inv_noise_var_use = hydra.beam_sampler.select_subarr(inv_noise_var_beam,
                                                           ant_samp_ind, Nants)
            data_use = hydra.beam_sampler.select_subarr(data_beam, ant_samp_ind, Nants)

            # Have to split real/imag - circ. Gauss so just factor of 2, no off-diags :)
            inv_noise_var_use = np.repeat(inv_noise_var_use[:, :, :, np.newaxis],
                                          2, axis=3) * 0.5
            inv_noise_var_sqrt_use = np.sqrt(inv_noise_var_use)


            # This one is actually complex so we use a special fn. in hydra.beam_sampler
            data_use = hydra.beam_sampler.split_real_imag(data_use, 'vec')

            # Construct RHS vector
            rhs_unflatten = hydra.beam_sampler.construct_rhs(data_use,
                                                             inv_noise_var_use,
                                                             inv_noise_var_sqrt_use,
                                                             coeff_mean,
                                                             cov_Tdag,
                                                             cho_tuple)
            bbeam = rhs_unflatten.flatten()
            shape = (Nfreqs, Ntimes, ncoeffs,  2)
            cov_Tdag_Ninv_T = hydra.beam_sampler.get_cov_Tdag_Ninv_T(inv_noise_var_use,
                                                                     cov_Tdag,
                                                                     zern_trans)

            def beam_lhs_operator(x):
                y = hydra.beam_sampler.apply_operator(np.reshape(x, shape),
                                                      cov_Tdag_Ninv_T)
                return(y.flatten())

            # What the shape would be if the matrix were represented densely
            beam_lhs_shape = (np.prod(shape), np.prod(shape))

            # Build linear operator object
            beam_linear_op = LinearOperator(matvec=beam_lhs_operator,
                                            shape=beam_lhs_shape)
            print("Beginning solve")
            # Solve using Conjugate Gradients
            x_soln, convergence_info = cg(beam_linear_op, bbeam, maxiter=100)
            print(f"Done solving, convergence_info: {convergence_info}")
            x_soln_res = np.reshape(x_soln, shape)
            x_soln_swap = np.transpose(x_soln_res, axes=(2, 0, 1, 3))

            # Update the coeffs between rounds
            beam_coeffs[:, :, :, ant_samp_ind] = 1.0 * x_soln_swap[:, :, :, 0] \
                                               + 1.j * x_soln_swap[:, :, :, 1]

        timing_info(ftime, n, "(D) Beam sampler", time.time() - t0)
        np.save("output/beam_%05d" % n, beam_coeffs)

    #---------------------------------------------------------------------------
    # (P) Probability values and importance weights
    #---------------------------------------------------------------------------
    if CALCULATE_STATS:
        # Calculate importance weights for this Gibbs sample
        # FIXME: Ignores priors for now! They will cancel anyway, unless the
        # prior terms also contain approximations

        # Calculate data minus model (chi^2) for the exact model
        ggv = hydra.apply_gains(current_data_model,
                                gains * (1. + current_delta_gain),
                                ants,
                                antpairs,
                                inline=False)
        chisq_exact = (data - ggv) * np.sqrt(inv_noise_var)

        # Calculate data minus model for the approximate model
        ggv0 = hydra.apply_gains(current_data_model,
                                 gains,
                                 ants,
                                 antpairs,
                                 inline=False)
        ggv_approx = ggv0 + hydra.gain_sampler.apply_proj(current_delta_gain,
                                                          A_real,
                                                          A_imag,
                                                          ggv0)
        chisq_approx = (data - ggv_approx) * np.sqrt(inv_noise_var)

        # Calculate chi^2 (log-likelihood) term for each model
        logl_exact = -0.5 * (  np.sum(chisq_exact.real**2.)
                             + np.sum(chisq_exact.imag**2.))
        logl_approx = -0.5 * (  np.sum(chisq_approx.real**2.)
                              + np.sum(chisq_approx.imag**2.))

        # Calculate importance weight (ratio of likelihoods here)
        importance_weight = np.exp(logl_exact - logl_approx)

        # Approximate number of degrees of freedom
        Ndof = 2*data.size # real + imaginary
        if SAMPLE_GAINS:
            Ndof -= gain_lhs_shape[-1]
        if SAMPLE_VIS:
            Ndof -= vis_lhs_shape[-1]
        if SAMPLE_PTSRC_AMPS:
            Ndof -= ptsrc_lhs_shape[-1]

        # Print log-likelihood and importance weight
        print("(P) Log-likelihood and importance weights:")
        print("    Exact logL   = %+8.5e" % logl_exact)
        print("    Approx. logL = %+8.5e" % logl_approx)
        print("    Delta logL   = %+8.5e" % (logl_exact - logl_approx))
        print("    Import. wgt  = %+8.5e" % importance_weight)
        print("    Deg. freedom = %+8.5e" % Ndof)
        print("    chi^2 / Ndof = %+8.5e" % (-2.*logl_approx / Ndof))
        with open("output/stats.dat", "ab") as f:
            np.savetxt(f, np.atleast_2d([n, logl_exact, logl_approx, importance_weight, Ndof]))

    #---------------------------------------------------------------------------
    # (O) Output diagnostics
    #---------------------------------------------------------------------------
    if OUTPUT_DIAGNOSTICS:

        # Output residual
        ggv = hydra.apply_gains(current_data_model,
                                gains*(1.+current_delta_gain),
                                ants,
                                antpairs,
                                inline=False)
        resid = data - ggv

        if PLOTTING:
            plt.matshow(np.abs(resid[0]), vmin=-5., vmax=5., aspect='auto')
            plt.colorbar()
            plt.title("%05d" % n)
            plt.gcf().set_size_inches((5., 4.))
            plt.savefig("output/resid_abs_000_%05d.png" % n)

            # Output chi^2
            chisq = (resid[0].real**2. + resid[0].imag**2.) / noise_var[0]
            chisq_tot = np.sum( (resid.real**2. + resid.imag**2.) / noise_var )
            plt.matshow(chisq, vmin=0., vmax=40., aspect='auto')
            plt.title(r"$\chi^2_{\rm tot} = %5.3e$" % chisq_tot)
            plt.colorbar()
            plt.title("%05d" % n)
            plt.gcf().set_size_inches((5., 4.))
            plt.savefig("output/chisq_000_%05d.png" % n)



    # Close all figures made in this iteration
    if PLOTTING:
        plt.close('all')
    timing_info(ftime, n, "(Z) Full iteration", time.time() - t0iter)
