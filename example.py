#!/usr/bin/env python

import numpy as np
import pylab as plt
import hydra

import numpy.fft as fft
import scipy.linalg
from scipy.sparse.linalg import cg, gmres, LinearOperator, bicgstab
from scipy.signal import blackmanharris
from scipy.sparse import coo_matrix
import pyuvsim
import time, os, resource
import multiprocessing
from hydra.utils import flatten_vector, reconstruct_vector, timing_info, \
                            build_hex_array, get_flux_from_ptsrc_amp, convert_to_tops

import argparse

description = "Example Gibbs sampling of the joint posterior of several analysis " \
              "parameters in 21-cm power spectrum estimation from a simulated " \
              "visibility data set"
parser = argparse.ArgumentParser(description=description)
parser.add_argument("--seed", type=int, action="store", default=0,
                    required=False, dest="seed",
                    help="Set the random seed.")
parser.add_argument("--gains", action="store_true",
                    required=False, dest="sample_gains",
                    help="Sample gains.")
parser.add_argument("--vis", action="store_true",
                    required=False, dest="sample_vis",
                    help="Sample visibilities in general.")
parser.add_argument("--ptsrc", action="store_true",
                    required=False, dest="sample_ptsrc",
                    help="Sample point source amplitudes.")
parser.add_argument("--beam", action="store_true",
                    required=False, dest="sample_beam",
                    help="Sample beams.")
parser.add_argument("--stats", action="store_true",
                    required=False, dest="calculate_stats",
                    help="Calculate statistics about the sampling results.")
parser.add_argument("--diagnostics", action="store_true",
                    required=False, dest="output_diagnostics",
                    help="Output diagnostics.")
parser.add_argument("--timing", action="store_true", required=False,
                    dest="save_timing_info", help="Save timing info.")
parser.add_argument("--plotting", action="store_true",
                    required=False, dest="plotting",
                    help="Output plots.")
parser.add_argument('--hex-array', type=int, action="store", default=(3,4),
                    required=False, nargs='+', dest="hex_array",
                    help="Hex array layout, specified as the no. of antennas "
                         "in the 1st and middle rows, e.g. '--hex-array 3 4'.")
parser.add_argument("--Nptsrc", type=int, action="store", default=100,
                    required=False, dest="Nptsrc",
                    help="Number of point sources to use in simulation (and model).")
parser.add_argument("--Ntimes", type=int, action="store", default=30,
                    required=False, dest="Ntimes",
                    help="Number of times to use in the simulation.")
parser.add_argument("--Nfreqs", type=int, action="store", default=60,
                    required=False, dest="Nfreqs",
                    help="Number of frequencies to use in the simulation.")
parser.add_argument("--Niters", type=int, action="store", default=100,
                    required=False, dest="Niters",
                    help="Number of joint samples to gather.")
parser.add_argument("--sigma-noise", type=float, action="store",
                    default=0.05, required=False, dest="sigma_noise",
                    help="Standard deviation of the noise, in the same units "
                         "as the visibility data.")
parser.add_argument("--beam-nmax", type=int, action="store",
                    default=10, required=False, dest="beam_nmax",
                    help="Maximum radial degree of the Zernike basis for the beams.")
parser.add_argument("--solver", type=str, action="store",
                    default='cg', required=False, dest="solver_name",
                    help="Which sparse matrix solver to use for linear systems ('cg' or 'gmres').")
parser.add_argument("--output-dir", type=str, action="store",
                    default="./output", required=False, dest="output_dir",
                    help="Output directory.")
parser.add_argument("--multiprocess", action="store_true", dest="multiprocess",
                    required=False,
                    help="Whether to use multiprocessing in vis sim calls.")
args = parser.parse_args()

# Set switches
SAMPLE_GAINS = args.sample_gains
SAMPLE_VIS = args.sample_vis
SAMPLE_PTSRC_AMPS = args.sample_ptsrc
SAMPLE_BEAM = args.sample_beam
CALCULATE_STATS = args.calculate_stats
OUTPUT_DIAGNOSTICS = args.output_diagnostics
SAVE_TIMING_INFO = args.save_timing_info
PLOTTING = args.plotting
MULTIPROCESS = args.multiprocess

# Print what's switched on
print("    Gain sampler:       ", SAMPLE_GAINS)
print("    Vis. sampler:       ", SAMPLE_VIS)
print("    Ptsrc. amp. sampler:", SAMPLE_PTSRC_AMPS)
print("    Beam sampler:       ", SAMPLE_BEAM)

# Check that at least one thing is being sampled
if not SAMPLE_GAINS and not SAMPLE_VIS and not SAMPLE_PTSRC_AMPS and not SAMPLE_BEAM:
    raise ValueError("No samplers were enabled. Must enable at least one "
                     "of 'gains', 'vis', 'ptsrc', 'beams'.")

# Simulation settings -- want some shorter variable names
Nptsrc = args.Nptsrc
Ntimes = args.Ntimes
Nfreqs = args.Nfreqs
Niters = args.Niters
hex_array = tuple(args.hex_array)
assert len(hex_array) == 2, "hex-array argument must have length 2."
beam_nmax = args.beam_nmax
sigma_noise = args.sigma_noise

hera_latitude = -30.7215 * np.pi / 180.0

# Check that output directory exists
output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("\nOutput directory:", output_dir)

# Linear solver to use
if args.solver_name == 'cg':
    solver = cg
elif args.solver_name == 'gmres':
    solver = gmres
elif args.solver_name == 'bicgstab':
    solver = bicgstab
else:
    raise ValueError("Solver '%s' not recognised." % args.solver_name)
print("    Solver:  %s" % args.solver_name)

# Random seed
np.random.seed(args.seed)
print("    Seed:    %d" % args.seed)

# Check number of threads available
Nthreads = os.environ.get('OMP_NUM_THREADS')
if Nthreads is None:
    Nthreads = multiprocessing.cpu_count()
else:
    Nthreads = int(Nthreads)
print("    Threads: %d available" % Nthreads)

# Timing file
ftime = os.path.join(output_dir, "timing.dat")

#-------------------------------------------------------------------------------
# (1) Simulate some data
#-------------------------------------------------------------------------------

# Simulate some data
times = np.linspace(0.2, 0.5, Ntimes)
freqs = np.linspace(100., 120., Nfreqs)

#ant_pos = build_hex_array(hex_spec=(3,4), d=14.6)
ant_pos = build_hex_array(hex_spec=hex_array, d=14.6)
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
        multiprocess=MULTIPROCESS
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
    plt.savefig(os.path.join(output_dir, "delta_g_true_000.png"))

    # Plot input (simulated) visibility model
    vminmax_vis = np.max(model0[0,:,:].real)
    plt.matshow(model0[0,:,:].real, vmin=-vminmax_vis, vmax=vminmax_vis)
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, "data_model_000_xxxxx.png"))

# Add noise
data += sigma_noise * np.sqrt(0.5) \
      * (  1.0 * np.random.randn(*data.shape) \
         + 1.j * np.random.randn(*data.shape))

# Plot data (including noise)
if PLOTTING:
    plt.matshow(data[0,:,:].real, vmin=-vminmax_vis, vmax=vminmax_vis)
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, "data_000.png"))

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
                                latitude=hera_latitude,
                                multiprocess=MULTIPROCESS
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
        x_soln, convergence_info = solver(gain_linear_op, b)
        timing_info(ftime, n, "(A) Gain sampler", time.time() - t0)


        # Reshape solution into complex array and multiply by S^1/2 to get set of
        # Fourier coeffs of the actual solution for the frac. gain perturbation
        x_soln = hydra.utils.reconstruct_vector(x_soln, gain_shape)
        x_soln = hydra.gain_sampler.apply_sqrt_pspec(gain_pspec_sqrt, x_soln)

        # x_soln is a set of Fourier coefficients, so transform to real space
        # (ifft gives Fourier -> data space)
        xgain = np.zeros_like(x_soln)
        for k in range(xgain.shape[0]):
            xgain[k, :, :] = fft.ifft2(x_soln[k, :, :])

        print("    Gain sample:", xgain[0,0,0], xgain.shape)
        np.save(os.path.join(output_dir, "delta_gain_%05d" % n), x_soln)

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
                plt.savefig(os.path.join(output_dir, "delta_g_%03d_%05d.png" % (i, n)))

            # Residual with true gains (abs)
            plt.matshow(np.abs(xgain[0]) - np.abs(delta_g[0]),
                        vmin=-np.max(np.abs(delta_g[0])),
                        vmax=np.max(np.abs(delta_g[0])),
                        aspect='auto')
            plt.colorbar()
            plt.title("%05d" % n)
            plt.gcf().set_size_inches((6., 4.))
            plt.savefig(os.path.join(output_dir, "gain_resid_amp_000_%05d.png" % n))

            # Residual with true gains (real)
            plt.matshow(np.real(xgain[0]) - np.real(delta_g[0]),
                        vmin=-np.max(np.real(delta_g[0])),
                        vmax=np.max(np.real(delta_g[0])),
                        aspect='auto')
            plt.colorbar()
            plt.title("%05d" % n)
            plt.gcf().set_size_inches((6., 4.))
            plt.savefig(os.path.join(output_dir, "gain_resid_real_000_%05d.png" % n))

            # Residual with true gains (imag)
            plt.matshow(np.imag(xgain[0]) - np.imag(delta_g[0]),
                        vmin=-np.max(np.imag(delta_g[0])),
                        vmax=np.max(np.imag(delta_g[0])),
                        aspect='auto')
            plt.colorbar()
            plt.title("%05d" % n)
            plt.gcf().set_size_inches((6., 4.))
            plt.savefig(os.path.join(output_dir, "gain_resid_imag_000_%05d.png" % n))

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
            plt.savefig(os.path.join(output_dir, "gain_resid_compare_imag_000_%05d.png" % n))

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
            plt.savefig(os.path.join(output_dir, "gain_resid_compare_imag_000_%05d.png" % n))

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
        x_soln, convergence_info = solver(vis_linear_op, bvis)
        timing_info(ftime, n, "(B) Visibility sampler", time.time() - t0)

        # Reshape solution into complex array and multiply by S^1/2 to get set of
        # Fourier coeffs of the actual solution for the frac. gain perturbation
        x_soln = hydra.utils.reconstruct_vector(x_soln, data.shape)
        x_soln = hydra.vis_sampler.apply_sqrt_pspec(vis_pspec_sqrt,
                                                    x_soln,
                                                    vis_group_id,
                                                    ifft=True)

        print("    Vis sample:", x_soln[0,0,0], x_soln.shape)
        np.save(os.path.join(output_dir, "vis_%05d" % n), x_soln)

        if PLOTTING:
            plt.matshow(current_data_model[0].real + x_soln[0].real,
                        vmin=-vminmax_vis,
                        vmax=vminmax_vis)
            plt.colorbar()
            plt.title("%05d" % n)
            plt.savefig(os.path.join(output_dir, "vis_000_%05d.png" % n))

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
            plt.savefig(os.path.join(output_dir, "resid_datamodel_000_%05d.png" % n))


    #---------------------------------------------------------------------------
    # (C) Point source amplitude sampler
    #---------------------------------------------------------------------------

    if SAMPLE_PTSRC_AMPS:

        # Get the projection operator with most recent gains applied
        t0 = time.time()
        proj = vis_proj_operator0.copy()
        gain_pert = gains * (1. + current_delta_gain)
        for k, bl in enumerate(antpairs):
            ant1, ant2 = bl
            i1 = np.where(ants == ant1)[0][0]
            i2 = np.where(ants == ant2)[0][0]
            proj[k,:,:,:] *= gain_pert[i1,:,:,np.newaxis] \
                           * gain_pert[i2,:,:,np.newaxis].conj()
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
        x_soln, convergence_info = solver(ptsrc_linear_op, bsrc)
        timing_info(ftime, n, "(C) Point source sampler", time.time() - t0)
        x_soln *= amp_prior_std # we solved for x = S^-1/2 s, so recover s
        print("    Example soln:", x_soln[:5]) # this is fractional deviation from assumed amplitude, so should be close to 0
        np.save(os.path.join(output_dir, "ptsrc_amp_%05d" % n), x_soln)

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
            plt.savefig(os.path.join(output_dir, "ptsrc_amp_%05d.png" % n))

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
            plt.savefig(os.path.join(output_dir, "data_model_000_%05d.png" % n))

    #---------------------------------------------------------------------------
    # (D) Beam sampler
    #---------------------------------------------------------------------------

    if SAMPLE_BEAM:
        def plot_beam_cross(beam_coeffs, ant_ind, iter, tag='', type='cross'):
            # Shape ncoeffs, Nfreqs, Nant -- just use a ref freq
            coeff_use = beam_coeffs[:, 0, :]
            Nants = coeff_use.shape[-1]
            # Zmatr has shape Ntimes, Nsource, ncoeff -- just grab first time
            ra_use = np.linspace(0, np.pi/2, num=100)
            dec_use = np.linspace(-0.6,-0.4, num=100)
            RA, DEC = np.meshgrid(ra_use, dec_use)
            txs, tys, tzs = convert_to_tops(RA.flatten(), DEC.flatten(), times,
                                            hera_latitude)
            Zmatr_use = hydra.beam_sampler.construct_zernike_matrix(beam_nmax,
                                                                np.array(txs),
                                                                np.array(tys))
            if type == 'cross':
                fig, ax = plt.subplots(figsize=(16, 9), nrows=Nants, ncols=Nants)
                for ant_ind1 in range(Nants):
                    beam_use1 = Zmatr_use @ (coeff_use[:, ant_ind1])
                    for ant_ind2 in range(Nants):
                        beam_use2 = Zmatr_use @ (coeff_use[:, ant_ind2])
                        beam_cross = (beam_use1 * beam_use2.conj())[-1]
                        if ant_ind1 >= ant_ind2:
                            ax[ant_ind1, ant_ind2].scatter(RA.flatten(), DEC.flatten(),
                                                           c=np.abs(beam_cross),
                                                           vmin=0, vmax=1)
                        else:
                            ax[ant_ind1, ant_ind2].scatter(RA.flatten(), DEC.flatten(),
                                                           c=np.angle(beam_cross),
                                                           vmin=-np.pi, vmax=np.pi,
                                                           cmap='twilight')
            else:
                fig, ax = plt.subplots(ncols=2)
                beam_use = (Zmatr_use@(coeff_use[:, ant_ind]))[-1]
                ax[0].scatter(RA.flatten(), DEC.flatten(),
                                               c=np.abs(beam_use),
                                               vmin=0, vmax=1)
                ax[1].scatter(RA.flatten(), DEC.flatten(),
                                               c=np.angle(beam_use),
                                               vmin=-np.pi, vmax=np.pi,
                                               cmap='twilight')

            fig.savefig(f"output/beam_plot_ant_{ant_ind}_iter_{iter}_{type}_{tag}.png")
            plt.close(fig)
            return
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
            inv_noise_var_beam = inv_noise_var_beam + np.swapaxes(inv_noise_var_beam, -1, -2)

            txs, tys, tzs = convert_to_tops(ra, dec, times, hera_latitude)

            Zmatr = hydra.beam_sampler.construct_zernike_matrix(beam_nmax,
                                                                np.array(txs),
                                                                np.array(tys))

            # All the same, so just repeat (for now)
            beam_coeffs = np.array(Nants * \
                                   [hydra.beam_sampler.fit_zernike_to_beam(
                                                                     beams[0],
                                                                     1e6 * freqs,
                                                                     Zmatr,
                                                                     txs,
                                                                     tys), ])
            # Want shape ncoeff, Nfreqs, Nants
            beam_coeffs = np.swapaxes(beam_coeffs, 0, 2).astype(complex)
            np.save(os.path.join(output_dir, "best_fit_beam"), beam_coeffs)
            ncoeffs = beam_coeffs.shape[0]

            if PLOTTING:
                plot_beam_cross(beam_coeffs, 0, 0, '_best_fit')


            amp_use = x_soln if SAMPLE_PTSRC_AMPS else ptsrc_amps
            flux_use = get_flux_from_ptsrc_amp(amp_use, freqs, beta_ptsrc)
            Mjk = hydra.beam_sampler.construct_Mjk(Zmatr, ant_pos, flux_use, ra, dec,
                                             freqs*1e6, times, polarized=False,
                                             latitude=hera_latitude,
                                             multiprocess=MULTIPROCESS)

            # Hardcoded parameters. Make variations smooth in time/freq.
            sig_freq = 0.5 * (freqs[-1] - freqs[0])
            prior_std=1e2
            cov_tuple = hydra.beam_sampler.make_prior_cov(freqs, times, ncoeffs,
                                                          prior_std, sig_freq,
                                                          ridge=1e-6)
            cho_tuple = hydra.beam_sampler.do_cov_cho(cov_tuple, check_op=False)
            cov_tuple_0 = hydra.beam_sampler.make_prior_cov(freqs, times, ncoeffs,
                                                          prior_std, sig_freq,
                                                          ridge=1e-6,
                                                          constrain_phase=True,
                                                          constraint=1)
            cho_tuple_0 = hydra.beam_sampler.do_cov_cho(cov_tuple, check_op=False)
            # Be lazy and just use the initial guess.
            coeff_mean = hydra.beam_sampler.split_real_imag(beam_coeffs[:, :, 0],
                                                            'vec')
            dmm, chi2 = hydra.beam_sampler.get_chi2(Mjk, beam_coeffs, data_beam,
                                               inv_noise_var_beam)
            print(f"Beam chi-square before sampling: {chi2}")

        t0 = time.time()

        # Round robin loop through the antennas
        for ant_samp_ind in range(Nants):
            if ant_samp_ind > 0:
                cov_tuple_use = cov_tuple
                cho_tuple_use = cho_tuple
            else:
                cov_tuple_use = cov_tuple_0
                cho_tuple_use = cho_tuple_0
            zern_trans = hydra.beam_sampler.get_zernike_to_vis(Zmatr, ant_pos,
                                                               flux_use, ra, dec,
                                                               freqs*1e6, times,
                                                               beam_coeffs,
                                                               ant_samp_ind,
                                                               polarized=False,
                                                               latitude=hera_latitude,
                                                               multiprocess=MULTIPROCESS)


            inv_noise_var_use = hydra.beam_sampler.select_subarr(inv_noise_var_beam,
                                                           ant_samp_ind, Nants)
            data_use = hydra.beam_sampler.select_subarr(data_beam, ant_samp_ind, Nants)

            # Construct RHS vector
            rhs_unflatten = hydra.beam_sampler.construct_rhs(data_use,
                                                             inv_noise_var_use,
                                                             coeff_mean,
                                                             zern_trans,
                                                             cov_tuple_use,
                                                             cho_tuple_use)
            bbeam = rhs_unflatten.flatten()
            shape = (Nfreqs, ncoeffs,  2)
            cov_Tdag_Ninv_T = hydra.beam_sampler.get_cov_Tdag_Ninv_T(inv_noise_var_use,
                                                                     zern_trans,
                                                                     cov_tuple_use)
            axlen = np.prod(shape)
            matr = cov_Tdag_Ninv_T.reshape([axlen, axlen]) + np.eye(axlen)
            if PLOTTING:

                print(f"Condition number for LHS {np.linalg.cond(matr)}")
                plt.figure()
                mx = np.amax(np.abs(matr))
                plt.matshow(np.log10(np.abs(matr) / mx), vmax=0, vmin=-8)
                plt.colorbar(label="$log_{10}$(|LHS|)")
                plt.savefig(f"output/beam_LHS_matrix_iter_{n}_ant_{ant_samp_ind}.pdf")
                plt.close()


            def beam_lhs_operator(x):
                y = hydra.beam_sampler.apply_operator(np.reshape(x, shape),
                                                      cov_Tdag_Ninv_T)
                return(y.flatten())

            # What the shape would be if the matrix were represented densely
            beam_lhs_shape = (axlen, axlen)

            # Build linear operator object
            beam_linear_op = LinearOperator(matvec=beam_lhs_operator,
                                            shape=beam_lhs_shape)

            print("Beginning solve")
            #x_soln, convergence_info = solver(beam_linear_op, bbeam,
                                              #maxiter=None)
            x_soln = np.linalg.solve(matr, bbeam)
            convergence_info=0
            print(f"Done solving, Niter: {convergence_info}")
            btest = beam_linear_op(x_soln)
            allclose = np.allclose(btest, bbeam)
            if not allclose:
                abs_diff = np.abs(btest-bbeam)
                wh_max_diff = np.argmax(abs_diff)
                max_diff = abs_diff[wh_max_diff]
                max_val = bbeam[wh_max_diff]
                print(f"btest not close to bbeam, max_diff: {max_diff}, max_val: {max_val}")
            x_soln_res = np.reshape(x_soln, shape)

            # Has shape Nfreqs, ncoeff, ncomp
            # Want shape ncoeff, Nfreqs, ncomp
            x_soln_swap = np.swapaxes(x_soln_res, 0, 1)

            # Update the coeffs between rounds
            beam_coeffs[:, :, ant_samp_ind] = 1.0 * x_soln_swap[:, :, 0] \
                                               + 1.j * x_soln_swap[:, :, 1]
            dmm, chi2 = hydra.beam_sampler.get_chi2(Mjk, beam_coeffs, data_beam,
                                               inv_noise_var_beam)
            print(f"Beam chi-square after sampling, iteration {n}: {chi2}")
            if PLOTTING:
                plot_beam_cross(beam_coeffs, ant_samp_ind, n, '',type='beam')


        if PLOTTING:
            plot_beam_cross(beam_coeffs, ant_samp_ind, n, '')
        timing_info(ftime, n, "(D) Beam sampler", time.time() - t0)
        np.save(os.path.join(output_dir, "beam_%05d" % n), beam_coeffs)

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
        with open(os.path.join(output_dir, "stats.dat"), "ab") as f:
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
            plt.savefig(os.path.join(output_dir, "resid_abs_000_%05d.png" % n))

            # Output chi^2
            chisq = (resid[0].real**2. + resid[0].imag**2.) / noise_var[0]
            chisq_tot = np.sum( (resid.real**2. + resid.imag**2.) / noise_var )
            plt.matshow(chisq, vmin=0., vmax=40., aspect='auto')
            plt.title(r"$\chi^2_{\rm tot} = %5.3e$" % chisq_tot)
            plt.colorbar()
            plt.title("%05d" % n)
            plt.gcf().set_size_inches((5., 4.))
            plt.savefig(os.path.join(output_dir, "chisq_000_%05d.png" % n))



    # Close all figures made in this iteration
    if PLOTTING:
        plt.close('all')
    timing_info(ftime, n, "(Z) Full iteration", time.time() - t0iter)


# Print final resource usage info
rusage = resource.getrusage(resource.RUSAGE_SELF)
print("\nResource usage (final):")
print("    Max. RSS (MB):   %8.2f" % (rusage.ru_maxrss/1024.))
print("    User time (sec): %8.2f" % (rusage.ru_utime))
