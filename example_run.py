#!/usr/bin/env python

import numpy as np
import pylab as plt
import hydra

import numpy.fft as fft
import scipy.linalg
from scipy.sparse.linalg import cg, gmres, LinearOperator
from scipy.signal import blackmanharris
from scipy.sparse import coo_matrix
import pyuvsim
from pyuvdata import UVData
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
parser.add_argument("--data-file", type=str, action="store",
                    required=True, dest="data_file",
                    help="Input data file.")
parser.add_argument("--gain-ref-file", type=str, action="store",
                    required=True, dest="gain_ref_file",
                    help="Reference gain solution.")
parser.add_argument("--source-catalogue", type=str, action="store",
                    default="./output", required=False, dest="source_catalogue",
                    help="Point source catalogue.")
parser.add_argument('--freq-pad', type=int, action="store", default=(0,0),
                    required=False, nargs='+', dest="freq_pad",
                    help="No. of channels of padding to add in frequency, "
                         "e.g. '--freq-pad 2 2'.")
parser.add_argument('--lst-pad', type=int, action="store", default=(0,0),
                    required=False, nargs='+', dest="lst_pad",
                    help="No. of samples of padding to add in LST, "
                         "e.g. '--lst-pad 2 2'.")
parser.add_argument("--output-dir", type=str, action="store",
                    default="./output", required=False, dest="output_dir",
                    help="Output directory.")
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

hera_latitude = -30.7215 * np.pi / 180.0

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
Niters = args.Niters
beam_nmax = args.beam_nmax
sigma_noise = args.sigma_noise
data_file = args.data_file
gain_ref_file = args.gain_ref_file
source_catalogue = args.source_catalogue
freq_pad = args.freq_pad
lst_pad = args.lst_pad

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
else:
    raise ValueError("Solver '%s' not recognised." % args.solver_name)
print("    Solver:  %s" % args.solver_name)

# Random seed
np.random.seed(args.seed)
print("    Seed:    %d" % args.seed)

# Check number of threads available
Nthreads = int(os.environ.get('OMP_NUM_THREADS'))
if Nthreads is None:
    Nthreads = multiprocessing.cpu_count()
print("    Threads: %d available" % Nthreads)

# Timing file
ftime = os.path.join(output_dir, "timing.dat")

#-------------------------------------------------------------------------------
# (1) Load data
#-------------------------------------------------------------------------------

# Load UVData data file
uvd = UVData()
uvd.read_uvh5(data_file)
print("Vis. data loaded from:", data_file)

# Telescope latitude
telescope_latitude = uvd.telescope_location_lat_lon_alt[0] # radians

# Get antenna info
ant_pos = hydra.utils.antenna_dict_from_uvd(uvd)
ants = np.array(list(ant_pos.keys()))
Nants = len(ants)

# Extract data and active baselines
data, antpairs, _ = hydra.extract_vis_from_uvdata(uvd, 
                                                  exclude_autos=True, 
                                                  lst_pad=lst_pad, 
                                                  freq_pad=freq_pad)
ants1, ants2 = list(zip(*antpairs))
Nfreqs_orig = data.shape[1] - sum(freq_pad)
Ntimes_orig = data.shape[2] - sum(lst_pad)

# Times and frequencies
# (FIXME: Time ordering not guaranteed)
times = hydra.utils.extend_coords_with_padding(np.unique(uvd.lst_array),
                                                   pad=lst_pad) # rad
freqs = hydra.utils.extend_coords_with_padding(np.unique(uvd.freq_array)/1e6,
                                                   pad=freq_pad) # MHz
Nfreqs = freqs.size
Ntimes = times.size
assert data.shape == (len(antpairs), Nfreqs, Ntimes)

# Load point source catalogue (ra/dec in degrees)
src_ra, src_dec, src_flux, src_beta = np.loadtxt(source_catalogue)
Nptsrc = src_ra.size
fluxes = get_flux_from_ptsrc_amp(src_flux, freqs, src_beta, ref_freq=100.)
src_ra = np.deg2rad(src_ra)
src_dec = np.deg2rad(src_dec)
print("Source catalogue loaded from:", source_catalogue)

# Print basic info about data
print("    Nants: ", Nants)
print("    Nfreqs: %d (padding %d | %d | %d)" % (Nfreqs, freq_pad[0], Nfreqs_orig, freq_pad[1]))
print("    Ntimes: %d (padding %d | %d | %d)" % (Ntimes, lst_pad[0], Ntimes_orig, lst_pad[1]))
print("    Nptsrc:", Nptsrc)

# Beams (FIXME)
#beams = [pyuvsim.analyticbeam.AnalyticBeam('gaussian', diameter=14.)
#         for ant in ants]
beams = [pyuvsim.AnalyticBeam('airy', diameter=14.6) for ant in ants]


# Define gains and gain perturbations
gains = hydra.utils.load_gain_model(gain_ref_file, 
                                    lst_pad=lst_pad, 
                                    freq_pad=freq_pad, 
                                    pad_value=1.)
assert gains.shape == (Nants, Nfreqs, Ntimes)
gains_actual = gains.copy()

# FIXME:
gains = 0.*gains + 1. # Use unity for initial guess of gain model

# Empty gain fluctuation model in FFT basis
frate = np.fft.fftfreq(times.size, d=times[1] - times[0])
tau = np.fft.fftfreq(freqs.size, d=freqs[1] - freqs[0])
delta_g = np.zeros_like(gains)

# Apply a Blackman-Harris window to apodise the edges
#window = blackmanharris(model0.shape[1], sym=True)[np.newaxis,:,np.newaxis] \
#       * blackmanharris(model0.shape[2], sym=True)[np.newaxis,np.newaxis,:]
window = 1. # no window for now

# Plot data
if PLOTTING:
    vminmax_vis = 1.1*np.max(data[0,:,:].real) # FIXME
    plt.matshow(data[0,:,:].real, vmin=-10., vmax=10.) #, vmin=-vminmax_vis, vmax=vminmax_vis)
    plt.title("data[0] real")
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, "data_000.png"))

#-------------------------------------------------------------------------------
# (2) Set up Gibbs sampler
#-------------------------------------------------------------------------------

# Run simulation to construct initial model 
t0 = time.time()
_sim_vis = hydra.vis_simulator.simulate_vis(
        ants=ant_pos,
        fluxes=fluxes,
        ra=src_ra,
        dec=src_dec,
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

# Get initial visibility model guesses (use the actual baseline model for now)
# This SHOULD NOT include gain factors of any kind
current_data_model = 1.*model0.copy() * window

# Initial gain perturbation guesses
current_delta_gain = np.zeros_like(gains)

# Initial point source amplitude factor
current_ptsrc_a = np.ones(src_ra.size)

# Precompute visibility projection operator (without gain factors) for ptsrc
# amplitude sampling step. NOTE: This has to be updated within the Gibbs loop
# if other components of the visibility model are being sampled
t0 = time.time()
vis_proj_operator0 = hydra.ptsrc_sampler.calc_proj_operator(
                                ra=src_ra,
                                dec=src_dec,
                                fluxes=fluxes,
                                ant_pos=ant_pos,
                                antpairs=antpairs,
                                freqs=freqs,
                                times=times,
                                beams=beams,
                                latitude=telescope_latitude
)
timing_info(ftime, 0, "(0) Precomp. ptsrc proj. operator", time.time() - t0)

# Set priors and auxiliary information
# FIXME: amp_prior_std is a prior around amp=0 I think, so can skew things low!
noise_var = (sigma_noise)**2. * np.ones(data.shape)
inv_noise_var = window / noise_var

# FIXME: Add padding to inverse noise variance
inv_noise_var[:,:freq_pad[0],:] = 0.
inv_noise_var[:,Nfreqs_orig+freq_pad[0]:,:] = 0.
inv_noise_var[:,:,:lst_pad[0]] = 0.
inv_noise_var[:,:,Ntimes_orig+lst_pad[0]:] = 0.

gain_pspec_sqrt = 0.1 * np.ones((gains.shape[1], gains.shape[2]))
gain_pspec_sqrt[0,0] = 1e-2 # FIXME: Try to fix the zero point?

# FIXME: Gain smoothing via priors
ii, jj = np.meshgrid(np.arange(gains.shape[2]), np.arange(gains.shape[1]))
gain_pspec_sqrt *= np.exp(-0.5 * np.sqrt(ii**2. + jj**2.)/1.**2.)

#plt.matshow(gain_pspec_sqrt, aspect='auto')
#plt.colorbar()
#plt.show()
#exit()

amp_prior_std = 0.1 * np.ones(Nptsrc) # 10% prior
#amp_prior_std[19] = 1e-3 # FIXME
vis_pspec_sqrt = 0.01 * np.ones((1, Nfreqs, Ntimes)) # currently same for all visibilities
vis_group_id = np.zeros(len(antpairs), dtype=int) # index 0 for all

A_real, A_imag = hydra.gain_sampler.proj_operator(ants, antpairs)

gain_shape = gains.shape
N_gain_params = 2 * gains.shape[0] * gains.shape[1] * gains.shape[2]
N_vis_params = 2 * data.shape[0] * data.shape[1] * data.shape[2]


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
            vminmax = 0.02 # FIXME

            # Residual with true gains (abs)
            plt.matshow(np.abs(xgain[0]) - np.abs(delta_g[0]),
                        vmin=-np.max(np.abs(delta_g[0])),
                        vmax=np.max(np.abs(delta_g[0])),
                        aspect='auto')
            plt.colorbar()
            plt.title("%05d" % n)
            plt.gcf().set_size_inches((6., 4.))
            plt.savefig(os.path.join(output_dir, "gain_resid_amp_000_%05d.png" % n))


            # FIXME
            # Plot real and imaginary part of current gain estimate and true gains
            for ant in ants:
                plt.subplot(121)
                g_act = np.abs(gains_actual[ant])
                plt.matshow(g_act, vmin=g_act.min(), vmax=g_act.max(), 
                            aspect='auto', fignum=False)
                plt.colorbar()
                plt.title("True gain (ant %03d)" % ant)
                
                plt.subplot(122)
                g_samp = np.abs((gains * (1. + xgain))[ant])
                #g_samp = np.abs(xgain[ant])
                plt.matshow(g_samp, vmin=g_act.min(), vmax=g_act.max(), 
                            aspect='auto', fignum=False)
                plt.colorbar()
                plt.title("Gain sample (ant %03d) %05d" % (ant, n))
                plt.gcf().set_size_inches((10., 4.))

                plt.savefig(os.path.join(output_dir, "gain_comp_%03d_%05d.png" % (ant, n)))

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

        # Update current state
        current_data_model = current_data_model + x_soln


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
            inv_noise_var_beam = inv_noise_var_beam + np.swapaxes(inv_noise_var_beam, -1, -2)

            txs, tys, tzs = convert_to_tops(ra, dec, times, telescope_latitude)

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
            beam_coeffs = np.swapaxes(beam_coeffs, 0, 3).astype(complex)
            ncoeffs = beam_coeffs.shape[0]


            amp_use = x_soln if SAMPLE_PTSRC_AMPS else ptsrc_amps
            flux_use = get_flux_from_ptsrc_amp(amp_use, freqs, beta_ptsrc)
            Mjk = hydra.beam_sampler.construct_Mjk(Zmatr, ant_pos, flux_use, ra, dec,
                                             freqs, times, polarized=False,
                                             latitude=telescope_latitude)

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
            cov_Tdag = hydra.beam_sampler.get_cov_Tdag(cov_tuple, zern_trans)

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
            axlen = np.prod(shape)
            if PLOTTING:
                matr = cov_Tdag_Ninv_T.reshape([axlen, axlen]) + np.eye(axlen)
                plt.figure()
                plt.matshow(np.log10(np.abs(matr)), vmax=0, vmin=-8)
                plt.colorbar()
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
            # Solve using Conjugate Gradients
            x_soln, convergence_info = solver(beam_linear_op, bbeam, maxiter=100)
            print(f"Done solving, convergence_info: {convergence_info}")
            x_soln_res = np.reshape(x_soln, shape)
            x_soln_swap = np.transpose(x_soln_res, axes=(2, 0, 1, 3))

            # Update the coeffs between rounds
            beam_coeffs[:, :, :, ant_samp_ind] = 1.0 * x_soln_swap[:, :, :, 0] \
                                               + 1.j * x_soln_swap[:, :, :, 1]

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
