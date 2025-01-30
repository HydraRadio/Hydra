#!/usr/bin/env python

from mpi4py import MPI
import numpy as np
import pylab as plt
import hydra

import numpy.fft as fft
import scipy.linalg
from scipy.sparse.linalg import cg, gmres, LinearOperator, bicgstab
from scipy.signal.windows import blackmanharris
from scipy.sparse import coo_matrix
import pyuvsim
from hera_sim.beams import PolyBeam
import time, os, sys, resource
from hydra.utils import flatten_vector, reconstruct_vector, timing_info, \
                            build_hex_array, get_flux_from_ptsrc_amp, \
                            convert_to_tops, gain_prior_pspec_sqrt, \
                            freqs_times_for_worker, partial_fourier_basis_2d_from_nmax, \
                            status
from hydra.example import generate_random_ptsrc_catalogue, run_example_simulation
import hydra.linear_solver as linsolver


if __name__ == '__main__':

    # MPI setup
    comm = MPI.COMM_WORLD
    nworkers = comm.Get_size()
    myid = comm.Get_rank()

    # Parse commandline arguments
    args = hydra.config.get_config()

    # Check for debug mode
    debug = args.debug

    # Set switches
    SAMPLE_GAINS = args.sample_gains
    #SAMPLE_VIS = args.sample_vis
    SAMPLE_COSMO_FIELD = args.sample_cosmo_field
    SAMPLE_PTSRC_AMPS = args.sample_ptsrc
    SAMPLE_REGION_AMPS = args.sample_regions
    SAMPLE_BEAM = args.sample_beam
    SAMPLE_SH = args.sample_sh
    SAMPLE_SH_PSPEC = args.sample_sh_pspec
    SAMPLE_PSPEC = args.sample_pspec
    CALCULATE_STATS = args.calculate_stats
    SAVE_TIMING_INFO = args.save_timing_info
    PLOTTING = args.plotting

    # Print what's switched on
    if myid == 0:
        print("    Debug mode:                   ", debug)
        print("    Gain perturbation sampler:    ", SAMPLE_GAINS)
        print("    Cosmo field sampler:          ", SAMPLE_COSMO_FIELD)
        #print("    Vis. sampler:       ", SAMPLE_VIS)
        print("    Ptsrc. amplitude sampler:     ", SAMPLE_PTSRC_AMPS)
        print("    Diffuse region amp. sampler:  ", SAMPLE_REGION_AMPS)
        print("    Primary beam sampler:         ", SAMPLE_BEAM)
        print("    Spherical harmonic sampler:   ", SAMPLE_SH)
        print("    SH power spectrum sampler:    ", SAMPLE_SH_PSPEC)

    # Check that at least one thing is being sampled
    if not SAMPLE_GAINS and not SAMPLE_PTSRC_AMPS and not SAMPLE_BEAM \
       and not SAMPLE_SH and not SAMPLE_REGION_AMPS and not SAMPLE_SH_PSPEC \
       and not SAMPLE_COSMO_FIELD:
        raise ValueError("No samplers were enabled. Must enable at least one "
                         "of 'gains', 'ptsrc', 'regions', 'beams', 'sh', 'cl', 'pspec', 'cosmo'.")


    ############
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
    #ra_low, ra_high = (min(args.ra_bounds), max(args.ra_bounds))
    #dec_low, dec_high = (min(args.dec_bounds), max(args.dec_bounds))
    lst_min, lst_max = (min(args.lst_bounds), max(args.lst_bounds))
    freq_min, freq_max = (min(args.freq_bounds), max(args.freq_bounds))
    
    # Array latitude
    array_latitude = np.deg2rad(args.latitude)

    #--------------------------------------------------------------------------
    # Prior settings
    #--------------------------------------------------------------------------

    # Ptsrc, region, and vis prior settings
    ptsrc_amp_prior_level = args.ptsrc_amp_prior_level
    region_amp_prior_level = args.region_amp_prior_level

    # Gain prior settings
    gain_prior_amp = args.gain_prior_amp


    #--------------------------------------------------------------------------
    # Run and solver settings
    #--------------------------------------------------------------------------
    # Check that output directory exists
    output_dir = args.output_dir
    if myid == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("\nOutput directory:", output_dir)
    comm.barrier()

    # Linear solver to use
    if args.solver_name == 'cg':
        solver = cg
    elif args.solver_name == 'gmres':
        solver = gmres
    elif args.solver_name == 'bicgstab':
        solver = bicgstab
    elif args.solver_name == 'mpicg':
        solver = 'mpicg'
    else:
        raise ValueError("Solver '%s' not recognised." % args.solver_name)
    if myid == 0:
        print("    Solver:  %s" % args.solver_name)

    # Random seed
    np.random.seed(args.seed + myid) # need unique seed for each worker
    if myid == 0:
        print("    Seed:    %d" % args.seed)

    # Check number of threads available
    Nthreads = os.environ.get('OMP_NUM_THREADS')
    if myid == 0:
        print("    Parallelisation: %03d MPI workers, %s threads on root worker" \
               % (nworkers, Nthreads))

    # Timing file
    ftime = os.path.join(output_dir, "timing.dat")

    #-------------------------------------------------------------------------------
    # (1) Simulate some data
    #-------------------------------------------------------------------------------

    # Simulate some data
    times = np.linspace(lst_min, lst_max, Ntimes)
    freqs = np.linspace(freq_min, freq_max, Nfreqs)

    # FIXME: Could be more flexible
    ngrid = int(np.sqrt(nworkers))
    assert nworkers == ngrid * ngrid, "Currently restricted to having a square number of workers"
    fchunks = ngrid
    tchunks = ngrid

    # Get frequency/time indices for this worker
    freq_idxs, time_idxs, worker_map = freqs_times_for_worker(
                                                  comm=comm, 
                                                  freqs=freqs, 
                                                  times=times, 
                                                  fchunks=fchunks, 
                                                  tchunks=tchunks)
    freq_chunk = freqs[freq_idxs]
    time_chunk = times[time_idxs]


    #--------------------------------------------------------------------------
    # Generate random point source catalogue and distribute between workers
    #--------------------------------------------------------------------------
    ra = np.zeros(Nptsrc, dtype=np.float64)
    dec = np.zeros_like(ra)
    ptsrc_amps = np.zeros_like(ra)

    if myid == 0:
        # Generate random catalogue
        ra, dec, ptsrc_amps = generate_random_ptsrc_catalogue(Nptsrc, 
                                                              ra_bounds=args.ra_bounds, 
                                                              dec_bounds=args.dec_bounds, 
                                                              logflux_bounds=(-1., 2.))
        # Save generated catalogue info
        np.save(os.path.join(output_dir, "ptsrc_amps0"), ptsrc_amps)
        np.save(os.path.join(output_dir, "ptsrc_coords0"), np.column_stack((ra, dec)).T)

    # Broadcast full catalogue from root to all other workers
    comm.Bcast(ra, root=0)
    comm.Bcast(dec, root=0)
    comm.Bcast(ptsrc_amps, root=0)
    if debug:
        status(myid, "Received %d point sources (sum of amps: %f)" \
              % (ra.size, np.sum(ptsrc_amps).real), colour='b')
    comm.barrier()

    #--------------------------------------------------------------------------
    # Run point source visibility sim for this worker's chunk of freq./time space
    #--------------------------------------------------------------------------
    t0 = time.time()

    if myid == 0:
            status(None, "Simulating point source sky model with %d sources" 
                          % ra.size, 'c')
            status(None, "Simulation beam type: %s" % args.beam_sim_type, 'y')

    model0_chunk, fluxes_chunk, beams, ant_info = run_example_simulation( 
                                                       output_dir=output_dir, 
                                                       times=time_chunk,
                                                       freqs=freq_chunk,
                                                       ra=ra, 
                                                       dec=dec, 
                                                       ptsrc_amps=ptsrc_amps,
                                                       array_latitude=array_latitude,
                                                       hex_array=args.hex_array,
                                                       beam_type=args.beam_sim_type)
    status(myid, "Finished ptsrc simulation in %6.3f sec" % (time.time() - t0), colour='b')
    
    # Unpack antenna info
    ants, ant_pos, antpairs, ants1, ants2 = ant_info
    comm.barrier()

    #--------------------------------------------------------------------------
    # Run diffuse model simulation for this worker's chunk of frequency/time space
    #--------------------------------------------------------------------------

    if args.sim_diffuse_sky_model != 'none':
        t0 = time.time()

        if myid == 0:
            status(None, "Simulating diffuse sky model %s" % args.sim_diffuse_sky_model, 'c')

        # Get pixel values
        diffuse_pixel_ra, diffuse_pixel_dec, diffuse_fluxes_chunk \
            = hydra.region_sampler.get_diffuse_sky_model_pixels(
                                            freq_chunk, 
                                            nside=args.sim_diffuse_nside,
                                            sky_model=args.sim_diffuse_sky_model)

        # Simulation beams
        if "polybeam" in args.beam_sim_type.lower():
            # PolyBeam fitted to HERA Fagnoni beam
            beam_coeffs=[  0.29778665, -0.44821433, 0.27338272, 
                          -0.10030698, -0.01195859, 0.06063853, 
                          -0.04593295,  0.0107879,  0.01390283, 
                          -0.01881641, -0.00177106, 0.01265177, 
                          -0.00568299, -0.00333975, 0.00452368,
                           0.00151808, -0.00593812, 0.00351559
                         ]
            sim_beams = [PolyBeam(beam_coeffs, spectral_index=-0.6975, ref_freq=1.e8)
                         for ant in ants]
        else:
            sim_beams = [pyuvsim.analyticbeam.AnalyticBeam('gaussian', diameter=14.)
                         for ant in ants]
        if myid == 0:
            status(None, "Simulation beam type: %s" % args.beam_sim_type, 'y')

        # Calculate projection operator for each region
        diffuse_proj = hydra.region_sampler.calc_proj_operator(
                                          region_pixel_ra=diffuse_pixel_ra, 
                                          region_pixel_dec=diffuse_pixel_dec, 
                                          region_fluxes=diffuse_fluxes_chunk,
                                          region_idxs=[np.arange(diffuse_pixel_ra.size),], 
                                          ant_pos=ant_pos, 
                                          antpairs=antpairs, 
                                          freqs=freq_chunk, 
                                          times=time_chunk, 
                                          beams=sim_beams
                                        )
        model0_diffuse_chunk = diffuse_proj[:,:,:,0] # Take the 0th (and only) region

        # Clean up
        del diffuse_proj, diffuse_pixel_ra, diffuse_pixel_dec, diffuse_fluxes_chunk
        status(myid, "Finished diffuse simulation in %6.3f sec" \
                                     % (time.time() - t0), 'b')

        # Add diffuse model to ptsrc model
        model0_chunk += model0_diffuse_chunk

    comm.barrier()
    #--------------------------------------------------------------------------
    # Identify calibration source (brightest near beam)
    #--------------------------------------------------------------------------
    # Calibration source
    calsrc_radius = args.calsrc_radius
    if args.calsrc_std < 0.:
        calsrc = False
    else:
        calsrc = True
        calsrc_std = args.calsrc_std

    # Select what would be the calibration source (brightest, close to beam)
    calsrc_idxs = np.where(np.abs(dec - array_latitude)*180./np.pi < calsrc_radius)[0]
    assert len(calsrc_idxs) > 0, "No sources found within %d deg of the zenith" % calsrc_radius
    calsrc_idx = calsrc_idxs[np.argmax(ptsrc_amps[calsrc_idxs])]
    calsrc_amp = ptsrc_amps[calsrc_idx]
    if myid == 0:
        print("Calibration source:")
        print("  Enabled:            %s" % calsrc)
        print("  Index:              %d" % calsrc_idx)
        print("  Amplitude:          %6.3e" % calsrc_amp)
        print("  Dist. from zenith:  %6.2f deg" \
              % np.rad2deg(np.abs(dec[calsrc_idx] - array_latitude)))
        print("  Flux @ lowest freq: %6.3e Jy" % fluxes_chunk[calsrc_idx,0])
        print("")


    #--------------------------------------------------------------------------
    # Simulate antenna gains
    #--------------------------------------------------------------------------
    # Gain sim settings
    sim_gain_amp = args.sim_gain_amp_std

    comm.barrier()

    # Construct partial Fourier basis with only low-order modes, evaluated on 
    # the time/freq. ranges belonging to this worker
    # NOTE: If you do freq.max() - freq.min(), this will not be periodic!
    Lfreq = (freqs[1] - freqs[0]) * freqs.size
    Ltime = (times[1] - times[0]) * times.size
    Fbasis, k_freq, k_time = partial_fourier_basis_2d_from_nmax(
                                                freqs=freq_chunk, 
                                                times=time_chunk, 
                                                nmaxfreq=args.gain_nmaxfreq, 
                                                nmaxtime=args.gain_nmaxtime, 
                                                Lfreq=Lfreq, 
                                                Ltime=Ltime,
                                                freq0=freqs[0],
                                                time0=times[0],
                                                shape0=(freqs.size, times.size),
                                                positive_only=args.gain_only_positive_modes)
    Ngain_modes = k_freq.size
    Nants = len(ants)

    # Save Fbasis operator and modes
    np.save(os.path.join(output_dir, "Fbasis_w%04d" % myid), Fbasis)
    if myid == 0:
        np.save(os.path.join(output_dir, "k_freq"), k_freq)
        np.save(os.path.join(output_dir, "k_time"), k_time)

    # Define gains and gain perturbations
    gains_chunk = (1. + 1.j) * np.ones((Nants, freq_chunk.size, time_chunk.size), 
                                       dtype=model0_chunk.dtype)
    
    # Make it so that gain amps aren't all the same
    gain0_level = np.zeros(Nants, dtype=np.complex128)
    if myid == 0:
        gain0_level[:] = 0.05 * (np.random.randn(Nants) + 1.j*np.random.randn(Nants))
    comm.Bcast(gain0_level, root=0)
    for i in range(Nants):
        gains_chunk[i] += gain0_level[i]
    if myid == 0:
        np.save(os.path.join(output_dir, "sim_gain0_level"), gain0_level)
    #status(myid, gain0_level, 'r')

    # Simple prior on gain perturbation modes
    prior_std_delta_g = sim_gain_amp_std * np.ones(Ngain_modes)
    prior_std_delta_g[0] *= 0. # FIXME: This sets the gain zero mode to zero

    # Random gain perturbation amplitudes (Nants, Ngain_modes)
    # Do realisation on root node and then broadcast to other workers
    delta_g_amps0 = np.zeros(Nants*Ngain_modes, dtype=gains_chunk.dtype)
    if myid == 0:
        np.random.seed(1)
        delta_g_amps0[:] = (prior_std_delta_g 
                              * (  1.0 * np.random.randn(Nants, Ngain_modes) \
                                 + 1.j * np.random.randn(Nants, Ngain_modes) )).flatten()
    delta_g_amps0 = delta_g_amps0.flatten().copy()
    comm.Bcast(delta_g_amps0, root=0)
    delta_g_amps0 = delta_g_amps0.reshape((Nants, Ngain_modes))
    status(myid, "Received %d delta_g amps (sum of amps: %f)" \
          % (delta_g_amps0.size, np.sum(delta_g_amps0).real), colour='b')

    # Dot product with partial Fourier operator to get simulated gain perturbations. 
    # These are for this worker's time/freq. chunk, but should be continuous if you 
    # stitch the chunks together.
    # (Nants, Nfreqs, Ntimes) = (Ngain_modes, Nfreqs, Ntimes) . (Nants, Ngain_modes)
    delta_g_chunk0 = np.tensordot(delta_g_amps0, Fbasis, axes=((1,), (0,)))
    comm.barrier()

    # Apply gains to model
    data_chunk = model0_chunk.copy()
    hydra.apply_gains(data_chunk, 
                      gains_chunk * (1. + delta_g_chunk0), 
                      ants, 
                      antpairs, 
                      inline=True)

    # Add noise
    noise_chunk = sigma_noise * np.sqrt(0.5) \
                              * (  1.0 * np.random.randn(*data_chunk.shape) \
                                 + 1.j * np.random.randn(*data_chunk.shape))
    data_chunk += noise_chunk
    comm.barrier()
    if myid == 0:
        status(None, "Simulation step finished", colour='b')

    # Save simulated model info
    np.save(os.path.join(output_dir, "sim_model0_chunk_w%04d" % myid), model0_chunk)
    np.save(os.path.join(output_dir, "sim_data_chunk_w%04d" % myid), data_chunk)
    np.save(os.path.join(output_dir, "sim_delta_g0_chunk_w%04d" % myid), delta_g_chunk0)
    np.save(os.path.join(output_dir, "sim_delta_g_amps0_w%04d" % myid), delta_g_amps0)


    #-------------------------------------------------------------------------------
    # (2) Set up Gibbs sampler
    #-------------------------------------------------------------------------------

    # Get initial visibility model guesses (use the actual baseline model for now)
    # This SHOULD NOT include gain factors of any kind
    current_data_model_chunk = model0_chunk.copy()
    current_data_model_chunk_ptsrc = 0
    current_data_model_chunk_region = 0
    current_data_model_chunk_sh = 0
    current_data_model_chunk_cosmo = 0

    # Initial gain perturbation guesses
    current_delta_gain = np.zeros_like(delta_g_chunk0)

    # Initial point source amplitude factor
    #current_ptsrc_a = np.ones(ra.size)

    # Set priors and auxiliary information
    noise_var_chunk = (sigma_noise)**2. * np.ones(data_chunk.shape)
    inv_noise_var_chunk = 1. / noise_var_chunk

    # Gain prior
    gain_pspec_sqrt = gain_prior_amp * np.ones(Fbasis.shape[0])

    # Fix gain prior zero mode if requested
    if args.gain_prior_zero_mode_std is not None:
        zero_mode_idx = np.where(np.logical_and(k_freq == 0., k_time == 0.))[0]
        gain_pspec_sqrt[zero_mode_idx] = float(args.gain_prior_zero_mode_std)

        if myid == 0:
            status(None, "Gain zero-mode prior level: %6.4e" 
                         % args.gain_prior_zero_mode_std, colour='b')

    # Ptsrc priors
    ptsrc_amp_prior_std = ptsrc_amp_prior_level * np.ones(Nptsrc)
    if myid == 0:
        status(None, "Ptsrc amp. prior level: %s" % ptsrc_amp_prior_level, colour='b')
    if calsrc:
        amp_prior_std[calsrc_idx] = calsrc_std

    # Precompute gain perturbation projection operators
    A_real, A_imag = None, None
    if SAMPLE_GAINS:

        t0 = time.time()
        A_real, A_imag = hydra.gain_sampler.proj_operator(ants, antpairs)
        if myid == 0:
            status(myid, "Precomp. gain proj. operator took %6.3f sec" \
                         % (time.time() - t0), 'b')

    # Precompute sky region projection operator
    region_proj = None
    if SAMPLE_REGION_AMPS:
        t0 = time.time()

        # Build segmented sky model (per worker)
        Nregions = args.region_nregions
        region_ra, region_dec, region_fluxes_chunk \
                = hydra.region_sampler.get_diffuse_sky_model_pixels(
                                                freq_chunk, 
                                                nside=args.region_nside,
                                                sky_model=args.region_sky_model)
        
        region_idxs = hydra.region_sampler.segmented_diffuse_sky_model_pixels(
                                                region_ra, 
                                                region_dec, 
                                                region_fluxes_chunk, 
                                                freq_chunk, 
                                                Nregions, 
                                                smoothing_fwhm=args.region_smoothing_fwhm)

        # Update Nregions in case it changed
        Nregions = len(region_idxs)

        # Calculate projection operator for each region
        region_proj = hydra.region_sampler.calc_proj_operator(
                                          region_pixel_ra=region_ra, 
                                          region_pixel_dec=region_dec, 
                                          region_fluxes=region_fluxes_chunk,
                                          region_idxs=region_idxs, 
                                          ant_pos=ant_pos, 
                                          antpairs=antpairs, 
                                          freqs=freq_chunk, 
                                          times=time_chunk, 
                                          beams=beams
                                        )

        # Region priors
        region_amp_prior_std = region_amp_prior_level * np.ones(Nregions)

        if myid == 0:
            status(myid, "Precomputed region proj. operator in %6.3f sec" \
                         % (time.time() - t0), 'b')


    # Precompute point source projection operator
    ptsrc_proj = None
    if SAMPLE_PTSRC_AMPS:
        t0 = time.time()
        ptsrc_proj = hydra.ptsrc_sampler.calc_proj_operator(
                                          ra=ra, 
                                          dec=dec, 
                                          fluxes=fluxes_chunk, 
                                          ant_pos=ant_pos, 
                                          antpairs=antpairs, 
                                          freqs=freq_chunk, 
                                          times=time_chunk, 
                                          beams=beams
                                        )
        if myid == 0:
            status(myid, "Precomputed ptsrc proj. operator in %6.3f sec" \
                         % (time.time() - t0), 'b')


    # Combine ptsrc and region amp projection operators and priors
    if SAMPLE_PTSRC_AMPS or SAMPLE_REGION_AMPS:
        if SAMPLE_PTSRC_AMPS and SAMPLE_REGION_AMPS:
            source_proj = np.concatenate((ptsrc_proj, region_proj), axis=-1) # join along last dimension
            amp_prior_std = np.concatenate((ptsrc_amp_prior_std, region_amp_prior_std))
        elif SAMPLE_REGION_AMPS:
            source_proj = region_proj
            amp_prior_std = region_amp_prior_std
        else:
            source_proj = ptsrc_proj
            amp_prior_std = ptsrc_amp_prior_std

    #-------------------
    # Precompute cosmo field projection operator
    cosmo_proj = None
    if SAMPLE_COSMO_FIELD:
        t0 = time.time()

        # Get sample points and unit flux per freq. channel
        cosmo_grid_ra, cosmo_grid_dec = hydra.cosmo_sampler.make_cosmo_field_grid(args)
        cosmo_fluxes_chunk = np.ones((cosmo_grid_ra.size, freq_chunk.size))

        print(cosmo_grid_ra.shape, cosmo_grid_dec.shape, "<<<<<<<<")

        # Calculate projection operator (this re-uses the point source 
        # projection operator code, )
        cosmo_proj = hydra.ptsrc_sampler.calc_proj_operator(
                                          ra=cosmo_grid_ra, 
                                          dec=cosmo_grid_dec, 
                                          fluxes=cosmo_fluxes_chunk, 
                                          ant_pos=ant_pos, 
                                          antpairs=antpairs, 
                                          freqs=freq_chunk, 
                                          times=time_chunk, 
                                          beams=beams
                                        )
        if myid == 0:
            status(myid, "Precomputed cosmo proj. operator in %6.3f sec" \
                         % (time.time() - t0), 'b')

        # FIXME
        cosmo_pspec_kbins = np.linspace(0., 10, 25)
        cosmo_pspec_current = 0.1*cosmo_pspec_kbins + np.ones(cosmo_pspec_kbins.size)

        cosmo_background_params = {
            'h':        0.69,
            'omega_m':  0.31
        }

        # Report on Fourier modes in the 3D cosmo field
        if myid == 0:

            # Calculate comoving 3D Fourier modes
            kx, ky, knu = hydra.cosmo_sampler.comoving_fourier_modes(
                                                x=np.unique(cosmo_grid_ra), 
                                                y=np.unique(cosmo_grid_dec), 
                                                freqs=freqs,
                                                **cosmo_background_params)

            knu3d, kx3d, ky3d = np.meshgrid(knu, kx, ky)
            k = np.sqrt(kx3d**2. + ky3d**2. + knu3d**2.)

            # Print report on available modes
            print("Cosmo field Fourier mode ranges:")
            print("    kx: (%7.4f, %7.4f) Mpc^-1" % (kx.min(), kx.max()))
            print("    ky: (%7.4f, %7.4f) Mpc^-1" % (ky.min(), ky.max()))
            print("   knu: (%7.4f, %7.4f) Mpc^-1" % (knu.min(), knu.max()))
            print("   |k|: (%7.4f, %7.4f) Mpc^-1" % (k.min(), k.max()))
            print("")
            
    
    # Precompute spherical harmonic projection operator
    sh_response_chunk = None
    if SAMPLE_SH:
        if myid == 0:
            status(None, "Precomputing SH proj. operator for lmax = %d, nside = %d" \
                         % (args.sh_lmax, args.sh_nside), 'c')
        t0 = time.time()
        sh_response_chunk, sh_autos, sh_ell, sh_m \
                            = hydra.sh_sampler.vis_proj_operator_no_rot(
                                            freqs=freq_chunk, 
                                            lsts=time_chunk, 
                                            beams=beams, 
                                            ant_pos=ant_pos, 
                                            lmax=args.sh_lmax, 
                                            nside=args.sh_nside,
                                            latitude=array_latitude,
                                            ref_freq=args.sh_ref_freq,
                                            spectral_idx=args.sh_spectral_idx)

        # Spherical harmonic prior mean
        Nshmodes = sh_response_chunk.shape[1] 
        sh_prior_mean = np.zeros(Nshmodes)
        sh_prior_var = (args.sh_prior_std)**2. * np.ones(Nshmodes)
        sh_current = sh_prior_mean.copy()

        if myid == 0:
            status(myid, "Precomp. sph. harmonic proj. operator took %6.3f sec" \
                         % (time.time() - t0), 'b')


    if SAMPLE_BEAM:

        # FIXME: Need to add MPI compatibility

        # Make a copy of the data that is more convenient for the beam calcs.
        data_beam = reshape_data_arr(data[np.newaxis, np.newaxis],
                                     Nfreqs,
                                     Ntimes,
                                     Nants, 1)
        
        # Doubles the autos, but we don't use them so it doesn't matter.
        # This makes it so we do not have to keep track of whether we are sampling
        # The beam coeffs or their conjugate!
        data_beam = data_beam + np.swapaxes(data_beam, -1, -2).conj()

        # Reshape inverse noise variance array
        inv_noise_var_beam = reshape_data_arr(inv_noise_var[np.newaxis, np.newaxis],
                                                             Nfreqs,
                                                             Ntimes,
                                                             Nants, 1)
        inv_noise_var_beam = inv_noise_var_beam + np.swapaxes(inv_noise_var_beam, -1, -2)

        # Output info about dynamic range
        _ddmax, _ddmin = np.amax(np.abs(data_beam)), np.amin(np.abs(data_beam))
        status(None, "Data dynamic range: %8.6e -- %8.6e" % (_ddmin, _ddmax), 'c' )


    #--------------------------------------------------------------------------
    # Gibbs sampler
    #--------------------------------------------------------------------------
    # Iterate the Gibbs sampler
    if myid == 0:
        print("="*60)
        print("Starting Gibbs sampler (%d iterations)" % Niters)
        print("="*60)
    
    for n in range(Niters):
        if myid == 0:
            print("-"*60)
            print(">>> Iteration %4d / %4d" % (n+1, Niters))
            print("-"*60)
        t0iter = time.time()

        #---------------------------------------------------------------------------
        # (A) Gain sampler
        #---------------------------------------------------------------------------
        if SAMPLE_GAINS:
            if myid == 0:
                status(None, "Gain sampler iteration %d" % n, 'b')

            # Current_data_model DOES NOT include gbar_i gbar_j^* factor, so we need
            # to apply it here to calculate the residual
            ggv_chunk = hydra.apply_gains(current_data_model_chunk,
                                          gains_chunk,
                                          ants,
                                          antpairs,
                                          inline=False)
            resid_chunk = data_chunk - ggv_chunk

            # Shape of the gain solution vector
            #x_shape = 2*len(ants)*Fbasis.shape[0]

            # Calculte RHS vector
            t0 = time.time()
            bgain = hydra.gain_sampler.construct_rhs_mpi(
                                                  comm=comm, 
                                                  resid=resid_chunk, 
                                                  inv_noise_var=inv_noise_var_chunk, 
                                                  pspec_sqrt=gain_pspec_sqrt, 
                                                  A_real=A_real, 
                                                  A_imag=A_imag, 
                                                  model_vis=ggv_chunk, 
                                                  Fbasis=Fbasis, 
                                                  realisation=True,
                                                  seed=100000*myid+n)
            if myid == 0:
                status(None, "Gain sampler construct RHS took %6.3f sec" 
                             % (time.time() - t0), 'b')

            # Bundle LHS operator into lambda function
            gain_lhs_fn = lambda v: hydra.gain_sampler.apply_operator_mpi(
                                                  comm=comm, 
                                                  x=v, 
                                                  inv_noise_var=inv_noise_var_chunk, 
                                                  pspec_sqrt=gain_pspec_sqrt, 
                                                  A_real=A_real, 
                                                  A_imag=A_imag, 
                                                  model_vis=ggv_chunk, 
                                                  Fbasis=Fbasis).flatten()

            # Run CG linear solver
            t0 = time. time()
            xgain = hydra.linear_solver.cg(Amat=None, 
                                           bvec=bgain, 
                                           linear_op=gain_lhs_fn, 
                                           use_norm_tol=True)
            if myid == 0:
                status(None, "Gain sampler solver took %6.3f sec" 
                             % (time.time() - t0), 'b')
            
            # We solved for x = S^-1/2 s, so recover s
            xgain = (  1.0*xgain[:xgain.size//2]
                     + 1.j*xgain[xgain.size//2:] ).reshape(delta_g_amps0.shape)
            xgain *= gain_pspec_sqrt[np.newaxis,:]

            # Print solution as sanity check
            if myid == 0:
                status(None, "Gain soln:" + str(xgain[1,:3]), 'y')
                status(None, "True soln:" + str(delta_g_amps0[1,:3]), 'y')

            # Save solution as new sample
            if myid == 0:
                # this is fractional deviation from assumed amplitude; should be close to 0
                np.save(os.path.join(output_dir, "delta_g_amps_%05d" % n), xgain)


            # Update current state of gain model
            current_delta_gain = np.tensordot(xgain, Fbasis, axes=((1,), (0,)))
            comm.barrier()


        #---------------------------------------------------------------------------
        # (BBBB) Cosmo field sampler
        #---------------------------------------------------------------------------
        if SAMPLE_COSMO_FIELD:


            # FIXME: Testing
            current_data_model_chunk_ptsrc = model0_chunk

            # Current_data_model DOES NOT include gbar_i gbar_j^* factor, so we need
            # to apply it here to calculate the residual
            model_resid_chunk = current_data_model_chunk_ptsrc \
                              + current_data_model_chunk_region \
                              + current_data_model_chunk_sh

            # Guard against all components being zero
            if model_resid_chunk == 0:
                model_resid_chunk = np.zeros_like(data_chunk)

            # Current_data_model DOES NOT include gbar_i gbar_j^* factor, so we need
            # to apply it here to calculate the residual
            ggv_chunk = hydra.apply_gains(model_resid_chunk,
                                          gains_chunk,
                                          ants,
                                          antpairs,
                                          inline=False)
            resid_chunk = data_chunk - ggv_chunk

            # Precompute: get LHS and RHS operators for linear system
            t0 = time.time()

            # Calculate prior term
            pspec3d = hydra.cosmo_sampler.calculate_pspec_on_grid(
                                                kbins=cosmo_pspec_kbins, 
                                                pspec=cosmo_pspec_current, 
                                                x=np.unique(cosmo_grid_ra), 
                                                y=np.unique(cosmo_grid_dec), 
                                                freqs=freqs, 
                                                **cosmo_background_params)

            # Precompute N^-1 part of LHS operator, and calculate RHS
            cosmo_lhs_Ninv_op, cosmo_rhs \
                = hydra.cosmo_sampler.precompute_mpi(
                                       comm,
                                       freqs=freqs,
                                       ants=ants, 
                                       antpairs=antpairs, 
                                       freq_chunk=freq_chunk, 
                                       time_chunk=time_chunk,
                                       proj_chunk=cosmo_proj,
                                       data_chunk=resid_chunk,
                                       inv_noise_var_chunk=inv_noise_var_chunk,
                                       gain_chunk=gains_chunk * (1. + current_delta_gain),
                                       pspec3d=pspec3d, 
                                       realisation=True)

            if myid == 0:
                status(None, "Cosmo field sampler linear system precompute took %6.3f sec" 
                             % (time.time() - t0), 'c')

            # Solve linear system (only root worker)
            if myid == 0:
                t0 = time.time()

                # Set solution size
                cosmo_soln_shape = (freqs.size, 
                                    np.unique(cosmo_grid_ra).size, 
                                    np.unique(cosmo_grid_dec).size)

                # LHS matrix-vector product function
                cosmo_lhs = lambda x: hydra.cosmo_sampler.apply_lhs_operator(
                                                            x.reshape(cosmo_soln_shape), 
                                                            cosmo_lhs_Ninv_op, 
                                                            pspec3d).flatten()

                # Run CG solver
                cosmo_soln = hydra.linear_solver.cg(Amat=None, 
                                                    bvec=cosmo_rhs.flatten(), 
                                                    linear_op=cosmo_lhs, 
                                                    comm=None)
                cosmo_soln_3d = cosmo_soln.reshape(cosmo_soln_shape)
                print("COSMO SOLUTION:", cosmo_soln)

                
                import pylab as plt
                plt.subplot(121)
                plt.matshow(cosmo_soln_3d[0], fignum=False, aspect='auto')
                plt.colorbar()

                plt.subplot(122)
                plt.matshow(cosmo_soln_3d[1], fignum=False, aspect='auto')
                plt.colorbar()
                plt.show()
                exit()
                

                status(None, "Cosmo field linear solver took %6.3f sec" 
                             % (time.time() - t0), 'c')
                status(None, "    Example cosmo soln:" + str(cosmo_soln[:4]))

            comm.barrier()
            
            # FIXME: Need to update current_data_model_chunk_cosmo
            current_data_model_chunk_cosmo = 0 # FIXME

        #---------------------------------------------------------------------------
        # (B) Source sampler (ptsrc, regions, or both)
        #---------------------------------------------------------------------------
        if SAMPLE_PTSRC_AMPS or SAMPLE_REGION_AMPS:


            # Current_data_model DOES NOT include gbar_i gbar_j^* factor, so we need
            # to apply it here to calculate the residual
            model_resid_chunk = current_data_model_chunk_cosmo \
                              + current_data_model_chunk_sh
            
            # Guard against all components being zero
            if model_resid_chunk == 0:
                model_resid_chunk = np.zeros_like(data_chunk)

            ggv_chunk = hydra.apply_gains(model_resid_chunk,
                                          gains_chunk,
                                          ants,
                                          antpairs,
                                          inline=False)
            resid_chunk = data_chunk - ggv_chunk


            # Get LHS and RHS operators for linear system
            t0 = time.time()
            source_op, source_rhs = hydra.ptsrc_sampler.precompute_mpi(
                                       comm,
                                       ants=ants, 
                                       antpairs=antpairs, 
                                       freq_chunk=freq_chunk, 
                                       time_chunk=time_chunk,
                                       proj_chunk=source_proj,
                                       data_chunk=resid_chunk,
                                       inv_noise_var_chunk=inv_noise_var_chunk,
                                       gain_chunk=gains_chunk * (1. + current_delta_gain),
                                       amp_prior_std=amp_prior_std, 
                                       realisation=True)

            comm.barrier()
            if myid == 0:
                status(None, "Source sampler linear system precompute took %6.3f sec" 
                             % (time.time() - t0), 'c')

            # Solve linear system
            x_soln = np.zeros(amp_prior_std.shape, dtype=amp_prior_std.dtype)
            
            if solver == 'mpicg':
                # Use MPI solver, which will distribute the linear system across workers

                # Get shape of ptsrc linear operator from root node
                source_op_shape = None
                source_op_shape_new = comm.bcast(source_op.shape, root=0)

                # Determine which workers get which blocks
                if source_op_shape != source_op_shape_new:
                    # This is the first iteration; assign workers to groups
                    source_op_shape = source_op_shape_new
                    comm_groups, block_map, block_shape \
                        = linsolver.setup_mpi_blocks(comm, 
                                                     matrix_shape=source_op_shape, 
                                                     split=ngrid)

                # Collect matrix/vector blocks on each worker
                my_Amat, my_bvec = None, None
                if comm_groups is not None:
                    comm_active = comm_groups[0]
                    my_Amat, my_bvec = linsolver.collect_linear_sys_blocks(comm_active, 
                                                                           block_map, 
                                                                           block_shape, 
                                                                           Amat=source_op, 
                                                                           bvec=source_rhs)
                comm.barrier()

                # Run MPI CG solver
                t0 = time.time()
                _xsoln = linsolver.cg_mpi(comm_groups, 
                                          my_Amat, 
                                          my_bvec, 
                                          source_op_shape[0], 
                                          block_map)
                if myid == 0:
                    x_soln = _xsoln # only root worker has complete x_soln
                    status(None, "Source sampler MPI CG solve took %6.3f sec" \
                                 % (time.time() - t0), 'c')

                comm.barrier()

            else:
                # Use serial CG solver on root worker
                if myid == 0:
                    t0 = time.time()
                    x_soln = scipy.linalg.solve(source_op, source_rhs, assume_a='her')
                    status(None, "Source sampler serial CG solve took %6.3f sec" \
                                 % (time.time() - t0), 'c')
                comm.barrier()

            # Save solution as new sample
            if myid == 0:
                x_soln *= amp_prior_std # we solved for x = S^-1/2 s, so recover s
                # this is fractional deviation from assumed amplitude; should be close to 0
                np.save(os.path.join(output_dir, "src_amp_%05d" % n), x_soln)

            # Broadcast x_soln to all workers and update model
            comm.Bcast(x_soln, root=0)
            comm.barrier()
            if myid == 0:
                status(myid, "    Example ptsrc soln:" + str(x_soln[:3]))
                status(myid, "    Example region soln:" + str(x_soln[Nptsrc:Nptsrc+3]))

            # Update visibility model with latest solution (does not include any gains)
            # Applies projection operator to ptsrc amplitude vector
            # Gains should not be applied here (see)
            if SAMPLE_PTSRC_AMPS and not SAMPLE_REGION_AMPS:
                x_soln_ptsrc = x_soln[:]
            if SAMPLE_REGION_AMPS and not SAMPLE_PTSRC_AMPS:
                x_soln_regions = x_soln[:]
            if SAMPLE_PTSRC_AMPS and SAMPLE_REGION_AMPS:
                x_soln_ptsrc = x_soln[:Nptsrc]
                x_soln_regions = x_soln[Nptsrc:]
            
            if SAMPLE_PTSRC_AMPS:
                current_data_model_chunk_ptsrc = (  ptsrc_proj.reshape((-1, Nptsrc)) 
                                               @ (1. + x_soln_ptsrc) ).reshape(
                                                            current_data_model_chunk.shape)
            if SAMPLE_REGION_AMPS:
                current_data_model_chunk_region = (  region_proj.reshape((-1, Nregions)) 
                                               @ (1. + x_soln_regions) ).reshape(
                                                            current_data_model_chunk.shape)
            current_data_model_chunk = current_data_model_chunk_ptsrc \
                                     + current_data_model_chunk_region \
                                     + current_data_model_chunk_sh \
                                     + current_data_model_chunk_cosmo

        #---------------------------------------------------------------------------
        # (C) Spherical harmonic a_lm sampler
        #---------------------------------------------------------------------------

        if SAMPLE_SH:

            if myid == 0:
                status(None, "Spherical harmonic mode sampler iteration %d" % n, 'b')

            # Current_data_model DOES NOT include gbar_i gbar_j^* factor, so we need
            # to apply it here to calculate the residual
            model_resid_chunk = current_data_model_chunk_ptsrc \
                              + current_data_model_chunk_region \
                              + current_data_model_chunk_cosmo
            
            # Guard against all components being zero
            if model_resid_chunk == 0:
                model_resid_chunk = np.zeros_like(data_chunk)

            ggv_chunk = hydra.apply_gains(model_resid_chunk,
                                          gains_chunk,
                                          ants,
                                          antpairs,
                                          inline=False)
            resid_chunk = data_chunk - ggv_chunk

            # Define left hand side operator
            # FIXME: Need to make MPI-enabled version of this function
            sh_lhs_operator = lambda x: hydra.sh_sampler.apply_lhs_no_rot_mpi( 
                                                        comm=comm,
                                                        a_cr=x, 
                                                        inv_noise_var=inv_noise_var_chunk, 
                                                        inv_prior_var=1./sh_prior_var, 
                                                        vis_response=sh_response_chunk )

            # Generate random maps for the realisations
            omega_a = np.random.randn(sh_prior_mean.size)
            omega_n = (  1.0*np.random.randn(*current_data_model_chunk.shape) 
                       + 1.j*np.random.randn(*current_data_model_chunk.shape) ) \
                      / np.sqrt(2.) # this will be a different realisation on each worker

            # Construct the right hand side
            t0 = time.time()
            sh_rhs = hydra.sh_sampler.construct_rhs_no_rot_mpi(
                                           comm=comm,
                                           data=resid_chunk,
                                           inv_noise_var=inv_noise_var_chunk, 
                                           inv_prior_var=1./sh_prior_var,
                                           omega_a=omega_a,
                                           omega_n=omega_n,
                                           a_0=sh_prior_mean,
                                           vis_response=sh_response_chunk )
            if myid == 0:
                status(None, "SH sampler construct RHS took %6.3f sec" \
                             % (time.time() - t0), 'c')
            
            # Run and time solver
            # The cg() function ensures that each worker has the same solution vector
            t0 = time.time()
            sh_soln = hydra.linear_solver.cg(Amat=None,
                                             bvec=sh_rhs,
                                             linear_op=sh_lhs_operator,
                                             maxiters=1000, 
                                             abs_tol=1e-8, 
                                             use_norm_tol=False,
                                             x0=sh_current, # initial guess
                                             comm=comm)

            if myid == 0:
                status(None, "SH sampler solve took %6.3f sec" \
                             % (time.time() - t0), 'c')
                print("sh_soln:", sh_soln)

            # Save solution as new sample
            if myid == 0:
                np.save(os.path.join(output_dir, "sh_amp_%05d" % n), sh_soln)

            # Update visibility model
            sh_current = sh_soln
            current_data_model_chunk_sh = sh_response_chunk @ sh_soln
            current_data_model_chunk = current_data_model_chunk_ptsrc \
                                     + current_data_model_chunk_region \
                                     + current_data_model_chunk_sh \
                                     + current_data_model_chunk_cosmo

        #---------------------------------------------------------------------------
        # (C) Spherical harmonic C_ell sampler
        #---------------------------------------------------------------------------

        if SAMPLE_SH_PSPEC:

            if myid == 0:
                status(None, "SH power spectrum sampler iteration %d" % n, 'b')

            # FIXME: Loop over ells
            sh_current
            scipy.stats.invgamma.rvs(loc=1, scale=1)

        #---------------------------------------------------------------------------
        # (D) E-field beam sampler
        #---------------------------------------------------------------------------

        if SAMPLE_BEAM:
            if myid == 0:
                status(None, "Beam sampler iteration %d" % n, 'b')

            t0b = time.time()
            t0 = time.time()
            bess_sky_contraction = hydra.beam_sampler.get_bess_sky_contraction(bess_outer, 
                                                                               ant_pos, 
                                                                               flux_use, 
                                                                               ra,
                                                                               dec, 
                                                                               freqs*1e6, 
                                                                               times,
                                                                               polarized=False, 
                                                                               latitude=hera_latitude, 
                                                                               multiprocess=False)
            timing_info(ftime, n, "(D) Computed beam bess_sky_contraction", time.time() - t0)

            # Round robin loop through the antennas
            for ant_samp_ind in range(Nants):
                if ant_samp_ind > 0:
                    cov_tuple_use = cov_tuple
                    cho_tuple_use = cho_tuple
                else:
                    cov_tuple_use = cov_tuple_0
                    cho_tuple_use = cho_tuple_0

                # Construct beam projection operator
                t0 = time.time()
                bess_trans = hydra.beam_sampler.get_bess_to_vis(bess_matr, ant_pos,
                                                                   flux_use, ra, dec,
                                                                   freqs*1e6, times,
                                                                   beam_coeffs,
                                                                   ant_samp_ind,
                                                                   polarized=False,
                                                                   latitude=array_latitude,
                                                                   multiprocess=False)

                timing_info(ftime, n, "(D) Computed beam get_bess_to_vis", time.time() - t0)
                
                t0 = time.time()
                bess_trans = hydra.beam_sampler.get_bess_to_vis_from_contraction(bess_sky_contraction,
                                                                                 beam_coeffs, 
                                                                                 ants, 
                                                                                 ant_samp_ind)
                timing_info(ftime, n, "(D) Computed beam get_bess_to_vis_from_contraction", time.time() - t0)

                #bess_trans = hydra.beam_sampler.get_bess_to_vis(bess_matr, ant_pos,
                 #                                                  flux_use, ra, dec,
                  #                                                 freqs*1e6, times,
                   #                                                beam_coeffs,
                    #                                               ant_samp_ind,
                     #                                              polarized=False,
                      #                                             latitude=hera_latitude,
                       #                                            multiprocess=MULTIPROCESS)

                status(None, "\tDoing other per-iteration pre-compute", 'c')
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
                

                shape = (Nfreqs, ncoeffs,  1, 1, 2)
                cov_Qdag_Ninv_Q = hydra.beam_sampler.get_cov_Qdag_Ninv_Q(inv_noise_var_use,
                                                                         bess_trans,
                                                                         cov_tuple_use)

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

                # What the shape would be if the matrix were represented densely
                beam_lhs_shape = (axlen, axlen)
                print("\tSolving")
                t0 = time.time()
                x_soln = np.linalg.solve(matr, bbeam)
                timing_info(ftime, n, "(D) Solved for beam", time.time() - t0)

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
        
                # FIXME: Must update sky models!


        #---------------------------------------------------------------------------
        # (E) 21cm field sampler
        #---------------------------------------------------------------------------

        """
        # Set up arrays for sampling
        signal_cr = np.zeros((Niter, Ntimes, Nfreqs), dtype=complex)
        signal_S = np.zeros((Niter, Nfreqs, Nfreqs))
        signal_ps = np.zeros((Niter, Nfreqs))
        fg_amps = np.zeros((Niter, Ntimes, Nmodes), dtype=complex)
        # Useful debugging statistics
        chisq = np.zeros((Niter, Ntimes, Nfreqs))
        ln_post = np.zeros(Niter)

        # Set initial value for signal_S
        signal_S = S_initial.copy()
        """

        if SAMPLE_PSPEC:

            # FIXME: Need to do this for each baseline

            # Do Gibbs iteration
            signal_cr, signal_S, signal_ps, fg_amps, chisq, ln_post\
                = gibbs_step_fgmodes(
                    vis=vis * flags,
                    flags=flags,
                    signal_S=signal_S,
                    fgmodes=fgmodes,
                    Ninv=Ninv,
                    ps_prior=ps_prior,
                    f0=None,
                    nproc=nproc,
                    map_estimate=map_estimate,
                    verbose=verbose
                )

        #---------------------------------------------------------------------------
        # (P) Probability values and importance weights
        #---------------------------------------------------------------------------
        if CALCULATE_STATS:
            raise NotImplementedError()
            # Calculate importance weights for this Gibbs sample
            # FIXME: Ignores priors for now! They will cancel anyway, unless the
            # prior terms also contain approximations

            # Calculate data minus model (chi^2) for the exact model
            ggv = hydra.apply_gains(current_data_model_chunk,
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
        # (Q) Resource report
        #---------------------------------------------------------------------------
    
        # Print resource usage info for this iteration
        if myid == 0:
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            print("\nResource usage (iter %05d):" % (n+1))
            print("    Max. RSS (MB):   %8.2f" % (rusage.ru_maxrss/1024.))
            print("    User time (sec): %8.2f" % (rusage.ru_utime), flush=True)
        
        comm.barrier()



status(myid, "Finished run", 'g')
comm.barrier()
