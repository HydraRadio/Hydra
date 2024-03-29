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
from hera_sim.beams import PolyBeam
import time, os, resource
import multiprocessing
from hydra.utils import flatten_vector, reconstruct_vector, timing_info, \
                            build_hex_array, get_flux_from_ptsrc_amp, \
                            convert_to_tops, gain_prior_pspec_sqrt

import argparse

if __name__ == '__main__':

    # Don't use fork! It duplicates the memory used.
    #multiprocessing.set_start_method('forkserver', force=True)

    description = "Example Gibbs sampling of the joint posterior of several analysis " \
                  "parameters in 21-cm power spectrum estimation from a simulated " \
                  "visibility data set"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, action="store", default=0,
                        required=False, dest="seed",
                        help="Set the random seed.")
    
    # Samplers
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
    
    # Output options
    parser.add_argument("--stats", action="store_true",
                        required=False, dest="calculate_stats",
                        help="Calcultae statistics about the sampling results.")
    parser.add_argument("--diagnostics", action="store_true",
                        required=False, dest="output_diagnostics",
                        help="Output diagnostics.") # This will be ignored
    parser.add_argument("--timing", action="store_true", required=False,
                        dest="save_timing_info", help="Save timing info.")
    parser.add_argument("--plotting", action="store_true",
                        required=False, dest="plotting",
                        help="Output plots.")
    
    # Array and data shape options
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
    
    # Noise level
    parser.add_argument("--sigma-noise", type=float, action="store",
                        default=0.05, required=False, dest="sigma_noise",
                        help="Standard deviation of the noise, in the same units "
                             "as the visibility data.")
    
    parser.add_argument("--solver", type=str, action="store",
                        default='cg', required=False, dest="solver_name",
                        help="Which sparse matrix solver to use for linear systems ('cg' or 'gmres').")
    parser.add_argument("--output-dir", type=str, action="store",
                        default="./output", required=False, dest="output_dir",
                        help="Output directory.")
    parser.add_argument("--multiprocess", action="store_true", dest="multiprocess",
                        required=False,
                        help="Whether to use multiprocessing in vis sim calls.")
    
    # Point source sim params
    parser.add_argument("--ra-bounds", type=float, action="store", default=(0, 1),
                        nargs=2, required=False, dest="ra_bounds",
                        help="Bounds for the Right Ascension of the randomly simulated sources")
    parser.add_argument("--dec-bounds", type=float, action="store", default=(-0.6, 0.4),
                        nargs=2, required=False, dest="dec_bounds",
                        help="Bounds for the Declination of the randomly simulated sources")
    parser.add_argument("--lst-bounds", type=float, action="store", default=(0.2, 0.5),
                        nargs=2, required=False, dest="lst_bounds",
                        help="Bounds for the LST range of the simulation, in radians.")
    parser.add_argument("--freq-bounds", type=float, action="store", default=(100., 120.),
                        nargs=2, required=False, dest="freq_bounds",
                        help="Bounds for the frequency range of the simulation, in MHz.")
    parser.add_argument("--ptsrc-amp-prior-level", type=float, action="store", default=0.1,
                        required=False, dest="ptsrc_amp_prior_level",
                        help="Fractional prior on point source amplitudes")
    parser.add_argument("--vis-prior-level", type=float, action="store", default=0.1,
                        required=False, dest="vis_prior_level",
                        help="Prior on visibility values")

    parser.add_argument("--calsrc-std", type=float, action="store", default=-1.,
                        required=False, dest="calsrc_std",
                        help="Define a different std. dev. for the amplitude prior of a calibration source. If -1, do not use a calibration source.")
    parser.add_argument("--calsrc-radius", type=float, action="store", default=10.,
                        required=False, dest="calsrc_radius",
                        help="Radius around declination of the zenith in which to search for brightest source, which is then identified as the calibration source.")
                        
    # Gain prior
    parser.add_argument("--gain-prior-amp", type=float, action="store", default=0.1,
                        required=False, dest="gain_prior_amp",
                        help="Overall amplitude of gain prior.")
    parser.add_argument("--gain-prior-sigma-frate", type=float, action="store", default=None,
                        required=False, dest="gain_prior_sigma_frate",
                        help="Width of a Gaussian prior in fringe rate, in units of mHz.")
    parser.add_argument("--gain-prior-sigma-delay", type=float, action="store", default=None,
                        required=False, dest="gain_prior_sigma_delay",
                        help="Width of a Gaussian prior in delay, in units of ns.")
    parser.add_argument("--gain-prior-zeropoint-std", type=float, action="store", default=None,
                        required=False, dest="gain_prior_zeropoint_std",
                        help="If specified, fix the std. dev. of the (0,0) mode to some value.")
    parser.add_argument("--gain-prior-frate0", type=float, action="store", default=0.,
                        required=False, dest="gain_prior_frate0",
                        help="The central fringe rate of the Gaussian taper (mHz).")
    parser.add_argument("--gain-prior-delay0", type=float, action="store", default=0.,
                        required=False, dest="gain_prior_delay0",
                        help="The central delay of the Gaussian taper (ns).")
    parser.add_argument("--gain-mode-cut-level", type=float, action="store", default=None,
                        required=False, dest="gain_mode_cut_level",
                        help="If specified, gain modes with (prior power spectrum) < (gain-mode-cut-level) * max(prior power spectrum) will be excluded from the linear solve (i.e. set to zero).")
    parser.add_argument("--gain-always-linear", type=bool, action="store", default=False,
                        required=False, dest="gain_always_linear",
                        help="If True, the gain perturbations are always applied under the linear approximation (the x_i x_j^* term is neglected everywhere)")

    # Gain simulation
    parser.add_argument("--sim-gain-amp-std", type=float, action="store", default=0.05,
                        required=False, dest="sim_gain_amp_std",
                        help="Std. dev. of amplitude of simulated gain.")
    parser.add_argument("--sim-gain-sigma-frate", type=float, action="store", default=None,
                        required=False, dest="sim_gain_sigma_frate",
                        help="Width of a Gaussian in fringe rate, in units of mHz.")
    parser.add_argument("--sim-gain-sigma-delay", type=float, action="store", default=None,
                        required=False, dest="sim_gain_sigma_delay",
                        help="Width of a Gaussian in delay, in units of ns.")
    parser.add_argument("--sim-gain-frate0", type=float, action="store", default=0.,
                        required=False, dest="sim_gain_frate0",
                        help="The central fringe rate of the Gaussian taper (mHz).")
    parser.add_argument("--sim-gain-delay0", type=float, action="store", default=0.,
                        required=False, dest="sim_gain_delay0",
                        help="The central delay of the Gaussian taper (ns).")
    
    # Beam parameters
    parser.add_argument("--beam-sim-type", type=str, action="store", default="gaussian",
                        required=False, dest="beam_sim_type",
                        help="Which type of beam to use for the simulation. ['gaussian', 'polybeam']")
    parser.add_argument("--beam-prior-std", type=float, action="store", default=1,
                        required=False, dest="beam_prior_std",
                        help="Std. dev. of beam coefficient prior, in units of Zernike coefficient")
    parser.add_argument("--beam-nmax", type=int, action="store",
                        default=16, required=False, dest="beam_nmax",
                        help="Maximum radial degree of the Fourier-Bessel basis for the beams.")
    parser.add_argument("--beam-mmax", type=int, action="store",
                        default=0, required=False, dest="beam_mmax",
                        help="Maximum azimuthal degree of the Fourier-Bessel basis for the beams.")
    parser.add_argument("--rho-const", type=float, action="store", 
                        default=np.sqrt(1-np.cos(np.pi * 23 / 45)),
                        required=False, dest="rho_const",
                        help="A constant to define the radial projection for the beam spatial basis")
    
    args = parser.parse_args()

    # Set switches
    SAMPLE_GAINS = args.sample_gains
    SAMPLE_VIS = args.sample_vis
    SAMPLE_PTSRC_AMPS = args.sample_ptsrc
    SAMPLE_BEAM = args.sample_beam
    CALCULATE_STATS = args.calculate_stats
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
    
    # Beam simulation parameters
    beam_nmax = args.beam_nmax
    beam_mmax = args.beam_mmax
    
    # Noise specification
    sigma_noise = args.sigma_noise
    
    # Gain simulation parameters
    sim_gain_amp_std = args.sim_gain_amp_std
    
    # Source position and LST/frequency ranges
    ra_low, ra_high = (min(args.ra_bounds), max(args.ra_bounds))
    dec_low, dec_high = (min(args.dec_bounds), max(args.dec_bounds))
    lst_min, lst_max = (min(args.lst_bounds), max(args.lst_bounds))
    freq_min, freq_max = (min(args.freq_bounds), max(args.freq_bounds))
    
    # Array latitude
    hera_latitude = np.deg2rad(-30.7215)

    # Ptsrc and vis prior settings
    ptsrc_amp_prior_level = args.ptsrc_amp_prior_level
    vis_prior_level = args.vis_prior_level

    calsrc_radius = args.calsrc_radius
    if args.calsrc_std < 0.:
        calsrc = False
    else:
        calsrc = True
        calsrc_std = args.calsrc_std

    # Gain sim settings
    sim_gain_amp = args.sim_gain_amp_std
    sim_gain_sigma_frate = args.sim_gain_sigma_frate
    sim_gain_sigma_delay = args.sim_gain_sigma_delay
    sim_gain_frate0 = args.sim_gain_frate0
    sim_gain_delay0 = args.sim_gain_delay0
    print("    Gain sim:")
    print("        amp:          ", sim_gain_amp)
    print("        sigma_frate:  ", sim_gain_sigma_frate)
    print("        sigma_delay:  ", sim_gain_sigma_delay)
    print("        frate0:       ", sim_gain_frate0)
    print("        delay0:       ", sim_gain_delay0)
    
    # Gain prior settings
    gain_prior_amp = args.gain_prior_amp
    gain_prior_sigma_frate = args.gain_prior_sigma_frate
    gain_prior_sigma_delay = args.gain_prior_sigma_delay
    gain_prior_zeropoint_std = args.gain_prior_zeropoint_std
    gain_prior_frate0 = args.gain_prior_frate0
    gain_prior_delay0 = args.gain_prior_delay0
    gain_mode_cut_level = args.gain_mode_cut_level
    gain_always_linear = args.gain_always_linear
    print("    Gain prior:")
    print("        amp:          ", gain_prior_amp)
    print("        sigma_frate:  ", gain_prior_sigma_frate)
    print("        sigma_delay:  ", gain_prior_sigma_delay)
    print("        zeropoint_std:", gain_prior_zeropoint_std)
    print("        frate0:       ", gain_prior_frate0)
    print("        delay0:       ", gain_prior_delay0)
    print("    Gains always linear:", gain_always_linear)
    print("")

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
    times = np.linspace(lst_min, lst_max, Ntimes)
    freqs = np.linspace(freq_min, freq_max, Nfreqs)

    ant_pos = build_hex_array(hex_spec=hex_array, d=14.6)
    ants = np.array(list(ant_pos.keys()))
    Nants = len(ants)
    print("Nants =", Nants)

    antpairs = []
    for i in range(len(ants)):
        for j in range(i, len(ants)):
            if i != j:
                # Exclude autos
                antpairs.append((i,j))


    ants1, ants2 = list(zip(*antpairs))

    # Generate random point source locations
    # RA goes from [0, 2 pi] and Dec from [-pi / 2, +pi / 2].
    ra = np.random.uniform(low=ra_low, high=ra_high, size=Nptsrc)
    
    # inversion sample to get them uniform on the sphere, in case wide bounds are used
    U = np.random.uniform(low=0, high=1, size=Nptsrc)
    dsin = np.sin(dec_high) - np.sin(dec_low)
    dec = np.arcsin(U * dsin + np.sin(dec_low)) # np.arcsin returns on [-pi / 2, +pi / 2]

    # Generate fluxes
    beta_ptsrc = -2.7
    ptsrc_amps = 10.**np.random.uniform(low=-1., high=2., size=Nptsrc)
    fluxes = get_flux_from_ptsrc_amp(ptsrc_amps, freqs, beta_ptsrc)
    print("pstrc amps (input):", ptsrc_amps[:5])
    np.save(os.path.join(output_dir, "ptsrc_amps0"), ptsrc_amps)
    np.save(os.path.join(output_dir, "ptsrc_coords0"), np.column_stack((ra, dec)).T)

    # Select what would be the calibration source (brightest, close to beam)
    calsrc_idxs = np.where(np.abs(dec - hera_latitude)*180./np.pi < calsrc_radius)[0]
    assert len(calsrc_idxs) > 0, "No sources found within %d deg of the zenith" % calsrc_radius
    calsrc_idx = calsrc_idxs[np.argmax(ptsrc_amps[calsrc_idxs])]
    calsrc_amp = ptsrc_amps[calsrc_idx]
    print("Calibration source:")
    print("  Enabled:            %s" % calsrc)
    print("  Index:              %d" % calsrc_idx)
    print("  Amplitude:          %6.3e" % calsrc_amp)
    print("  Dist. from zenith:  %6.2f deg" \
          % np.rad2deg(np.abs(dec[calsrc_idx] - hera_latitude)))
    print("  Flux @ lowest freq: %6.3e Jy" % fluxes[calsrc_idx,0])
    print("")

    # Beams
    if "polybeam" in args.beam_sim_type.lower():
        # PolyBeam fitted to HERA Fagnoni beam
        beam_coeffs=[  0.29778665, -0.44821433, 0.27338272, 
                      -0.10030698, -0.01195859, 0.06063853, 
                      -0.04593295,  0.0107879,  0.01390283, 
                      -0.01881641, -0.00177106, 0.01265177, 
                      -0.00568299, -0.00333975, 0.00452368,
                       0.00151808, -0.00593812, 0.00351559
                     ]
        beams = [PolyBeam(beam_coeffs, spectral_index=-0.6975, ref_freq=1.e8)
                 for ant in ants]
    else:
        beams = [pyuvsim.analyticbeam.AnalyticBeam('gaussian', diameter=14.)
                 for ant in ants]
    print("beam type:", args.beam_sim_type)



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
    np.save(os.path.join(output_dir, "model0"), model0)

    # Define gains and gain perturbations
    gains = (1. + 0.j) * np.ones((Nants, Nfreqs, Ntimes), dtype=model0.dtype)

    # Generate gain fluctuations from FFT basis
    frate = np.fft.fftfreq(times.size, d=times[1] - times[0])
    tau = np.fft.fftfreq(freqs.size, d=freqs[1] - freqs[0])
    delta_g_sqrt_pspec = gain_prior_pspec_sqrt(
                                lsts=times, 
                                freqs=freqs, 
                                gain_prior_amp=sim_gain_amp_std, 
                                gain_prior_sigma_frate=sim_gain_sigma_frate, 
                                gain_prior_sigma_delay=sim_gain_sigma_delay, 
                                gain_prior_zeropoint_std=None,
                                frate0=sim_gain_frate0, 
                                delay0=sim_gain_delay0 )
    
    # Make Gaussian realisation of gain fluctuation power spectrum, different for each antenna
    delta_g = np.array([np.fft.ifft2(delta_g_sqrt_pspec 
                                     * np.random.randn(*delta_g_sqrt_pspec.shape)) 
                        for i in range(Nants)],
                        dtype=model0.dtype)

    np.save(os.path.join(output_dir, "gains0"), gains)
    np.save(os.path.join(output_dir, "delta_g0"), delta_g)

    # Apply a Blackman-Harris window to apodise the edges
    #window = blackmanharris(model0.shape[1], sym=True)[np.newaxis,:,np.newaxis] \
    #       * blackmanharris(model0.shape[2], sym=True)[np.newaxis,np.newaxis,:]
    window = 1. # no window for now

    # Apply gains to model
    data = model0.copy() * window
    hydra.apply_gains(data, gains * (1. + delta_g), ants, antpairs, inline=True)

    # Add noise
    data += sigma_noise * np.sqrt(0.5) \
          * (  1.0 * np.random.randn(*data.shape) \
             + 1.j * np.random.randn(*data.shape))

    np.save(os.path.join(output_dir, "data0"), data)

    #-------------------------------------------------------------------------------
    # (2) Set up Gibbs sampler
    #-------------------------------------------------------------------------------

    # Get initial visibility model guesses (use the actual baseline model for now)
    # This SHOULD NOT include gain factors of any kind
    current_data_model = 1.*model0.copy() * window

    # Initial gain perturbation guesses
    current_delta_gain = np.zeros_like(delta_g)

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

    # Gain prior
    gain_pspec_sqrt = gain_prior_pspec_sqrt(
                                lsts=times, 
                                freqs=freqs, 
                                gain_prior_amp=gain_prior_amp, 
                                gain_prior_sigma_frate=gain_prior_sigma_frate, 
                                gain_prior_sigma_delay=gain_prior_sigma_delay, 
                                gain_prior_zeropoint_std=gain_prior_zeropoint_std,
                                frate0=gain_prior_frate0, 
                                delay0=gain_prior_delay0 )
            
    # Exclude gain modes that are strongly downweighted by the prior from the 
    # linear system
    reduced_idxs = None
    if gain_mode_cut_level is not None:
        # Find modes that are strongly suppressed by prior (1 = keep, 0 = discard)
        _cut_level = gain_mode_cut_level * gain_pspec_sqrt.max()
        gmodes = (1. + 1.j) * np.ones(gains.shape, dtype=np.complex128)
        gmodes[:, gain_pspec_sqrt < _cut_level] *= 0. # modes to discard
        gmodes = hydra.gain_sampler.flatten_vector(gmodes) # flatten
        reduced_idxs = np.where(gmodes == 1.)[0] # indices of kept modes
        
        print("(0) Excluding some gain modes: %d / %d excluded" \
              % (reduced_idxs.size, gmodes.size))
        print("    Gain mode cut threshold: %6.2e x max" % gain_mode_cut_level)
    
    # Ptsrc priors
    amp_prior_std = ptsrc_amp_prior_level * np.ones(Nptsrc)
    print("(0) Ptsrc amp. prior level:", ptsrc_amp_prior_level)
    if calsrc:
        amp_prior_std[calsrc_idx] = calsrc_std
    
    # Visibility priors
    # (currently same for all visibilities)
    vis_pspec_sqrt = vis_prior_level * np.ones((1, Nfreqs, Ntimes))
    vis_group_id = np.zeros(len(antpairs), dtype=int) # index 0 for all

    # Construct projection operators and store shapes
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
            
            # LHS operator and RHS vector of linear system
            # Uses reduced_idxs to exclude modes that are strongly down-weighted
            b = hydra.gain_sampler.flatten_vector(
                    hydra.gain_sampler.construct_rhs(resid,
                                                     inv_noise_var,
                                                     gain_pspec_sqrt,
                                                     A_real,
                                                     A_imag,
                                                     ggv,
                                                     realisation=True),
                    reduced_idxs=reduced_idxs
                )

            def gain_lhs_operator(x):
                return hydra.gain_sampler.flatten_vector(
                            hydra.gain_sampler.apply_operator(
                                hydra.gain_sampler.reconstruct_vector(
                                                        x, 
                                                        gain_shape, 
                                                        reduced_idxs=reduced_idxs),
                                                 inv_noise_var,
                                                 gain_pspec_sqrt,
                                                 A_real,
                                                 A_imag,
                                                 ggv),
                            reduced_idxs=reduced_idxs
                                     )

            # Build linear operator object
            if reduced_idxs is not None:
                gain_lhs_shape = (reduced_idxs.size, reduced_idxs.size)
            else:
                gain_lhs_shape = (N_gain_params, N_gain_params)
            gain_linear_op = LinearOperator(matvec=gain_lhs_operator,
                                            shape=gain_lhs_shape)

            # Solve using Conjugate Gradients or similar
            t0 = time.time()
            x_soln, convergence_info = solver(gain_linear_op, b)
            timing_info(ftime, n, "(A) Gain sampler", time.time() - t0)
            print("    Gain sampler convergence info:", convergence_info)

            # Reshape solution into complex array and multiply by S^1/2 to get set of
            # Fourier coeffs of the actual solution for the frac. gain perturbation
            x_soln = hydra.utils.reconstruct_vector(x_soln, 
                                                    gain_shape, 
                                                    reduced_idxs=reduced_idxs)
            x_soln = hydra.gain_sampler.apply_sqrt_pspec(gain_pspec_sqrt, x_soln)

            # x_soln is a set of Fourier coefficients, so transform to real space
            # (ifft gives Fourier -> data space)
            xgain = np.zeros_like(x_soln)
            for k in range(xgain.shape[0]):
                xgain[k, :, :] = fft.ifft2(x_soln[k, :, :])

            print("    Gain sample:", xgain[0,0,0], xgain.shape)
            np.save(os.path.join(output_dir, "delta_gain_%05d" % n), x_soln)

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
                  - (  proj.reshape((-1, Nptsrc)) 
                     @ np.ones_like(amp_prior_std) ).reshape(current_data_model.shape)

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
            # this is fractional deviation from assumed amplitude; should be close to 0
            print("    Example soln:", x_soln[:5])
            np.save(os.path.join(output_dir, "ptsrc_amp_%05d" % n), x_soln)

            # Update visibility model with latest solution (does not include any gains)
            # Applies projection operator to ptsrc amplitude vector
            current_data_model = (  vis_proj_operator0.reshape((-1, Nptsrc)) 
                                  @ (1. + x_soln) ).reshape(current_data_model.shape)

        #---------------------------------------------------------------------------
        # (D) Beam sampler
        #---------------------------------------------------------------------------

        if SAMPLE_BEAM:
            def plot_beam_cross(beam_coeffs, ant_ind, iter, tag='', type='cross'):
                # Shape ncoeffs, Nfreqs, Nant -- just use a ref freq
                coeff_use = beam_coeffs[:, 0, :]
                Nants = coeff_use.shape[1]
                if type == 'cross':
                    fig, ax = plt.subplots(figsize=(16, 9), nrows=Nants, ncols=Nants,
                                           subplot_kw={'projection': 'polar'})
                    for ant_ind1 in range(Nants):

                        beam_use1 = bess_matr_fit @ coeff_use[:, ant_ind1, 0, 0]
                        for ant_ind2 in range(Nants):
                            beam_use2 = bess_matr_fit @ (coeff_use[:, ant_ind2, 0, 0])
                            beam_cross = (beam_use1 * beam_use2.conj())
                            if ant_ind1 >= ant_ind2:
                                ax[ant_ind1, ant_ind2].pcolormesh(PHI,
                                                                  RHO,
                                                                  np.abs(beam_cross),
                                                                  vmin=0, vmax=1)
                            else:
                                ax[ant_ind1, ant_ind2].pcolormesh(PHI,
                                                                  RHO,
                                                                  np.angle(beam_cross),
                                                                  vmin=-np.pi,
                                                                  vmax=np.pi,
                                                                  cmap='twilight')
                else:
                    fig, ax = plt.subplots(ncols=2, subplot_kw={'projection': 'polar'})
                    beam_use = (bess_matr_fit@(coeff_use[:, ant_ind, 0, 0]))
                    ax[0].pcolormesh(PHI, RHO, np.abs(beam_use),
                                     vmin=0, vmax=1)
                    ax[1].pcolormesh(PHI, RHO, np.angle(beam_use),
                                     vmin=-np.pi, vmax=np.pi, cmap='twilight')

                fig.savefig(f"{output_dir}/beam_plot_ant_{ant_ind}_iter_{iter}_{type}_{tag}.png")
                plt.close(fig)
                return
            
            # Have to have an initial guess and do some precompute
            if n == 0:
                beam_nmodes, beam_mmodes = np.meshgrid(np.arange(1, beam_nmax + 1), np.arange(-beam_mmax, beam_mmax + 1))
                beam_nmodes = beam_nmodes.flatten()
                beam_mmodes = beam_mmodes.flatten()
                # Make a copy of the data that is more convenient for the beam calcs.
                data_beam = hydra.beam_sampler.reshape_data_arr(data[np.newaxis, np.newaxis],
                                                                Nfreqs,
                                                                Ntimes,
                                                                Nants, 1)
                # Doubles the autos, but we don't use them so it doesn't matter.
                # This makes it so we do not have to keep track of whether we are sampling
                # The beam coeffs or their conjugate!
                data_beam = data_beam + np.swapaxes(data_beam, -1, -2).conj()

                inv_noise_var_beam = hydra.beam_sampler.reshape_data_arr(inv_noise_var[np.newaxis, np.newaxis],
                                                                         Nfreqs,
                                                                         Ntimes,
                                                                         Nants, 1)
                inv_noise_var_beam = inv_noise_var_beam + np.swapaxes(inv_noise_var_beam, -1, -2)

                za_fit = np.arange(91) * np.pi / 180
                rho_fit = np.sqrt(1 - np.cos(za_fit)) / args.rho_const
                phi_fit = np.linspace(0, 2 * np.pi, num=360)
                PHI, RHO = np.meshgrid(phi_fit, rho_fit)

                bess_matr_fit = hydra.beam_sampler.get_bess_matr(beam_nmodes,
                                                                 beam_mmodes,
                                                                 RHO, PHI)

                beam_coeffs_fit = hydra.beam_sampler.fit_bess_to_beam(
                                                  beams[0],
                                                  1e6 * freqs,
                                                  beam_nmodes,
                                                  beam_mmodes,
                                                  RHO, PHI)

                beam_coeffs_fit = beam_coeffs_fit
                print("Printing beam best fit dynamic range")
                print(np.amax(np.abs(beam_coeffs_fit)), np.amin(np.abs(beam_coeffs_fit)))
                print("Printing data dynamic range")
                print(np.amax(np.abs(data_beam)), np.amin(np.abs(data_beam)))
                txs, tys, tzs = convert_to_tops(ra, dec, times, hera_latitude)

                # area-preserving
                rho = np.sqrt(1 - tzs) / args.rho_const
                phi = np.arctan2(tys, txs)
                bess_matr = hydra.beam_sampler.get_bess_matr(beam_nmodes,
                                                             beam_mmodes,
                                                             rho, phi)

                # All the same, so just repeat (for now)
                beam_coeffs = np.array(Nants * [beam_coeffs_fit])
                # Want shape ncoeff, Nfreqs, Nants, Npol, Npol
                beam_coeffs = np.swapaxes(beam_coeffs, 0, 2).astype(complex)
                np.save(os.path.join(output_dir, "best_fit_beam"), beam_coeffs)
                ncoeffs = beam_coeffs.shape[0]

                if PLOTTING:
                    plot_beam_cross(beam_coeffs, 0, 0, '_best_fit')


                amp_use = x_soln if SAMPLE_PTSRC_AMPS else ptsrc_amps
                flux_use = get_flux_from_ptsrc_amp(amp_use, freqs, beta_ptsrc)

                # Hardcoded parameters. Make variations smooth in time/freq.
                sig_freq = 0.5 * (freqs[-1] - freqs[0])
                cov_tuple = hydra.beam_sampler.make_prior_cov(freqs, times, ncoeffs,
                                                              args.beam_prior_std, sig_freq,
                                                              ridge=1e-6)
                cho_tuple = hydra.beam_sampler.do_cov_cho(cov_tuple, check_op=False)
                cov_tuple_0 = hydra.beam_sampler.make_prior_cov(freqs, times, ncoeffs,
                                                              args.beam_prior_std, sig_freq,
                                                              ridge=1e-6,
                                                              constrain_phase=True,
                                                              constraint=1)
                cho_tuple_0 = hydra.beam_sampler.do_cov_cho(cov_tuple, check_op=False)
                # Be lazy and just use the initial guess.
                coeff_mean = beam_coeffs[:, :, 0]

            t0 = time.time()

            # Round robin loop through the antennas
            for ant_samp_ind in range(Nants):
                if ant_samp_ind > 0:
                    cov_tuple_use = cov_tuple
                    cho_tuple_use = cho_tuple
                else:
                    cov_tuple_use = cov_tuple_0
                    cho_tuple_use = cho_tuple_0
                bess_trans = hydra.beam_sampler.get_bess_to_vis(bess_matr, ant_pos,
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
                                                                 bess_trans,
                                                                 cov_tuple_use,
                                                                 cho_tuple_use)
                bbeam = rhs_unflatten.flatten()
                #print(rhs_unflatten.shape)

                shape = (Nfreqs, ncoeffs,  1, 1, 2)
                cov_Qdag_Ninv_Q = hydra.beam_sampler.get_cov_Qdag_Ninv_Q(inv_noise_var_use,
                                                                         bess_trans,
                                                                         cov_tuple_use)
                #print(cov_Qdag_Ninv_Q.shape)

                axlen = np.prod(shape)

                # fPbpQBcCF->fbQcFBpPC
                matr = cov_Qdag_Ninv_Q.transpose((0,2,4,6,8,5,3,1,7)).reshape([axlen, axlen]) + np.eye(axlen)
                if PLOTTING:

                    print(f"Condition number for LHS {np.linalg.cond(matr)}")
                    plt.figure()
                    mx = np.amax(np.abs(matr))
                    plt.matshow(np.log10(np.abs(matr) / mx), vmax=0, vmin=-8)
                    plt.colorbar(label="$log_{10}$(|LHS|)")
                    plt.savefig(f"{output_dir}/beam_LHS_matrix_iter_{n}_ant_{ant_samp_ind}.pdf")
                    plt.close()


                def beam_lhs_operator(x):
                    y = hydra.beam_sampler.apply_operator(np.reshape(x, shape),
                                                          cov_Qdag_Ninv_Q)
                    return(y.flatten())

                # What the shape would be if the matrix were represented densely
                beam_lhs_shape = (axlen, axlen)

                x_soln = np.linalg.solve(matr, bbeam)

                test_close = False
                if test_close:
                    btest = beam_linear_op(x_soln)
                    allclose = np.allclose(btest, bbeam)
                    if not allclose:
                        abs_diff = np.abs(btest-bbeam)
                        wh_max_diff = np.argmax(abs_diff)
                        max_diff = abs_diff[wh_max_diff]
                        max_val = bbeam[wh_max_diff]
                        raise AssertionError(f"btest not close to bbeam, max_diff: {max_diff}, max_val: {max_val}")
                x_soln_res = np.reshape(x_soln, shape)

                # Has shape Nfreqs, ncoeff, Npol, Npol, ncomp
                # Want shape ncoeff, Nfreqs, Npol, Npol, ncomp
                x_soln_swap = np.swapaxes(x_soln_res, 0, 1)

                # Update the coeffs between rounds
                beam_coeffs[:, :, ant_samp_ind] = 1.0 * x_soln_swap[:, :, :, :, 0] \
                                                   + 1.j * x_soln_swap[:, :, :, :, 1]
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
            
        timing_info(ftime, n, "(Z) Full iteration", time.time() - t0iter)


    # Print final resource usage info
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    print("\nResource usage (final):")
    print("    Max. RSS (MB):   %8.2f" % (rusage.ru_maxrss/1024.))
    print("    User time (sec): %8.2f" % (rusage.ru_utime))
