#!/usr/bin/env python

import numpy as np
import hydra

from scipy.sparse.linalg import cg, gmres, bicgstab
import time, os
import multiprocessing
from hydra.utils import timing_info, build_hex_array, get_flux_from_ptsrc_amp, \
                         convert_to_tops

import argparse

if __name__ == '__main__':

    description = "Example Gibbs sampling of the joint posterior of beam "  \
                  "parameters from a simulated visibility data set " 
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, action="store", default=1001,
                        required=False, dest="seed",
                        help="Set the random seed.")
    
    # Misc
    parser.add_argument("--recalc-sc-op", action="store_true", required=False,
                        dest="recalc_sc_op", 
                        help="Recalculate a large operator in between iterations" \
                            "despite that it is constant; useful for profiling")
    parser.add_argument("--test-close", action="store_true", required=False,
                        dest="test_close",
                        help="Test whether linear solver solutions are close or not.")
    
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
    parser.add_argument("--Ntimes", type=int, action="store", default=10,
                        required=False, dest="Ntimes",
                        help="Number of times to use in the simulation.")
    parser.add_argument("--Nfreqs", type=int, action="store", default=10,
                        required=False, dest="Nfreqs",
                        help="Number of frequencies to use in the simulation.")
    parser.add_argument("--Niters", type=int, action="store", default=100,
                        required=False, dest="Niters",
                        help="Number of joint samples to gather.")
    parser.add_argument("--array-lat", type=float, required=False,
                        default=-30.7215, action="store",
                        dest="array_lat", help="Array latitude, in degrees.")
    
    # Instrumental parameters for noise estimation
    parser.add_argument("--integration-depth", type=float, action="store",
                        default=10., required=False, dest="integration_depth",
                        help="Integration time, in seconds")
    parser.add_argument("--ch-wid", type=float, action="store",
                        default=200e6 / 2048, required=False, dest="ch_wid",
                        help="Fine channel width for visibilities, in Hz.")
    
    # Computational nuances
    parser.add_argument("--solver", type=str, action="store",
                        default='cg', required=False, dest="solver_name",
                        help="Which sparse matrix solver to use for linear systems ('cg' or 'gmres' or 'bicgstab').")
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
    parser.add_argument("--freq-low", type=float, action="store", required=False,
                        default=145e6, dest="freq_low", 
                        help="Lowest frequency in the simulation, in Hz.")
    
    # Beam parameters
    parser.add_argument("--beam-file", type=str, action="store",
                        required=True, dest="beam_file",
                        help="Path to file containing a fiducial beam.")
    parser.add_argument("--beam-prior-std", type=float, action="store", default=1.,
                        required=False, dest="beam_prior_std",
                        help="Std. dev. of beam coefficient prior, in units of FB coefficient")
    parser.add_argument("--Nbasis", type=int, action="store", required=False, default=32,
                        help="Number of basis functions to use for beam estimation.")
    parser.add_argument("--nmax", type=int, action="store", default=80,
                        required=False, help="Maximum radial mode for beam modeling.")
    parser.add_argument("--mmax", type=int, action="store", default=45, 
                        required=False, help="Maxmimum azimuthal mode for beam modeling.")
    parser.add_argument("--rho-const", type=float, action="store", 
                        default=np.sqrt(1-np.cos(np.pi * 23 / 45)),
                        required=False, dest="rho_const",
                        help="A constant to define the radial projection for the"
                             " beam spatial basis")
    parser.add_argument("--trans-std", required=False, type=float,
                    default=1e-2, dest="trans_std",
                    help="Standard deviation for random tilt of beam")
    parser.add_argument("--rot-std-deg", required=False, type=float, 
                        dest="rot_std_deg", default=1., 
                        help="Standard deviation for random beam rotation, in degrees.")
    parser.add_argument("--stretch-std", required=False, type=float, 
                        dest="stretch_std", default=1e-2, 
                        help="Standard deviation for random beam stretching.")
    parser.add_argument("--pca-modes", required=True, type=str,
                        dest="pca_modes", help="Path to saved PCA eigenvectors.")
    
    args = parser.parse_args()

    hex_array = tuple(args.hex_array)
    assert len(args.hex_array) == 2, "hex-array argument must have length 2."

    # In case these are passed out of order, also shorter names
    ra_low, ra_high = (min(args.ra_bounds), max(args.ra_bounds))
    dec_low, dec_high = (min(args.dec_bounds), max(args.dec_bounds))
    lst_min, lst_max = (min(args.lst_bounds), max(args.lst_bounds))

    # Convert from degrees to radian
    array_lat = np.deg2rad(args.array_lat)

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
    times = np.linspace(lst_min, lst_max, args.Ntimes)
    freqs = np.arange(args.freq_low, args.freq_low + args.Nfreqs * args.ch_wid, args.ch_wid)

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
    ra = np.random.uniform(low=ra_low, high=ra_high, size=args.Nptsrc)
    
    # inversion sample to get them uniform on the sphere, in case wide bounds are used
    U = np.random.uniform(low=0, high=1, size=args.Nptsrc)
    dsin = np.sin(dec_high) - np.sin(dec_low)
    dec = np.arcsin(U * dsin + np.sin(dec_low)) # np.arcsin returns on [-pi / 2, +pi / 2]

    # Generate fluxes
    beta_ptsrc = -2.7
    ptsrc_amps = 10.**np.random.uniform(low=-1., high=2., size=args.Nptsrc)
    fluxes = get_flux_from_ptsrc_amp(ptsrc_amps, freqs, beta_ptsrc)
    print("pstrc amps (input):", ptsrc_amps[:5])
    np.save(os.path.join(output_dir, "ptsrc_amps0"), ptsrc_amps)
    np.save(os.path.join(output_dir, "ptsrc_coords0"), np.column_stack((ra, dec)).T)


    beams = []
    for ant_ind in range(Nants):
        load = os.path.exists(f"{output_dir}/perturbed_beam_beamvals_seed_{args.seed + ant_ind}.npy")
        save = not load
        pow_sb = hydra.beam_sampler.get_pert_beam(args.seed + ant_ind,
                                                  args.beam_file, 
                                                  trans_std=args.trans_std,
                                                  rot_std_deg=args.rot_std_deg,
                                                  stretch_std=args.stretch_std,
                                                  mmax=args.mmax, 
                                                  nmax=args.nmax,
                                                  sqrt=True, Nfeeds=2, 
                                                  num_modes_comp=32, save=save,
                                                  outdir=args.output_dir, load=load)
        beams.append(pow_sb)
    mmodes = np.arange(-args.mmax, args.mmax + 1)
    unpert_sb = hydra.sparse_beam.sparse_beam(args.beam_file, nmax=args.nmax, 
                                              mmodes=mmodes, Nfeeds=2, 
                                              alpha=args.rho_const,
                                              num_modes_comp=32,
                                              sqrt=True)

    sim_outpath = os.path.join(output_dir, "model0.npy")
    if not os.path.exists(sim_outpath):
        # Run a simulation
        t0 = time.time()
        _sim_vis = hydra.vis_simulator.simulate_vis(
                ants=ant_pos,
                fluxes=fluxes,
                ra=ra,
                dec=dec,
                freqs=freqs, # Make sure this is in Hz!
                lsts=times,
                beams=beams,
                polarized=False,
                precision=2,
                latitude=args.array_lat,
                use_feed="x",
                multiprocess=args.multiprocess,
                force_no_beam_sqrt=True,
            )
        timing_info(ftime, 0, "(0) Simulation", time.time() - t0)
        np.save(sim_outpath, _sim_vis)
    else:
        _sim_vis = np.load(sim_outpath)

    autos = np.abs(_sim_vis[:, :, np.arange(Nants), np.arange(Nants)])
    noise_var = autos[:, :, None] * autos[:, :, :, None] / (args.integration_depth * args.ch_wid)

    noise = (np.random.normal(scale=np.sqrt(noise_var)) + 1.j * np.random.normal(scale=np.sqrt(noise_var))) / np.sqrt(2)
    data = _sim_vis + noise
    del _sim_vis # Save some memory
    del noise

    np.save(os.path.join(output_dir, "data0"), data)

    inv_noise_var = 1/noise_var

    txs, tys, tzs = convert_to_tops(ra, dec, times, args.array_lat)

    bess_matr, trig_matr = unpert_sb.get_dmatr_interp(np.arctan2(tys, txs).flatten(), 
                                                      np.arccos(tzs).flatten())
    bess_matr = bess_matr.reshape(args.Ntimes, args.Nptsrc, args.nmax)
    trig_matr = trig_matr.reshape(args.Ntimes, args.Nptsrc, 2 * args.mmax + 1)

    mid_freq = freqs[args.Nfreqs // 2]
    closest_chan = np.argmin(np.abs(mid_freq - pow_sb.freq_array))
    mean_mode = pow_sb.bess_fits[:, :, 0, 0, 0, closest_chan]
    mean_mode = mean_mode / np.sum(np.abs(mean_mode)**2) 
    pca_modes = np.load(args.pca_modes)[:, :args.Nbasis - 1].reshape(args.nmax, 2 * args.mmax + 1, args.Nbasis - 1) 

    
    Pmatr = np.concatenate([mean_mode[:, :, None], pca_modes], axis=2)

    BPmatr = np.tensordot(bess_matr, Pmatr, axes=1).transpose(3, 0, 1, 2) # Nbasis, Ntimes, Nsrc, Naz
    Dmatr = np.sum(BPmatr * trig_matr, axis=3).transpose(1, 2, 0) # Ntimes, Nsrc, Nbasis
    Dmatr_outer = hydra.beam_sampler.get_bess_outer(Dmatr)
    
    beam_coeffs = np.zeros([Nants, args.Nfreqs, args.Nbasis, 1, 1])
    beam_coeffs[:, :, 0] = 1
    # Want shape Nbasis, Nfreqs, Nants, Npol, Npol
    beam_coeffs = np.swapaxes(beam_coeffs, 0, 2).astype(complex)

    sig_freq = 0.5 * (freqs[-1] - freqs[0])
    cov_tuple = hydra.beam_sampler.make_prior_cov(freqs, times, args.Nbasis,
                                                  args.beam_prior_std, sig_freq,
                                                  ridge=1e-6)
    cho_tuple = hydra.beam_sampler.do_cov_cho(cov_tuple, check_op=False)
    # Be lazy and just use the initial guess.
    coeff_mean = beam_coeffs[:, :, 0]
    
    t0 = time.time()
    bess_sky_contraction = hydra.beam_sampler.get_bess_sky_contraction(Dmatr_outer, 
                                                                       ant_pos, 
                                                                       fluxes, 
                                                                       ra,
                                                                       dec, 
                                                                       freqs, 
                                                                       times,
                                                                       polarized=False, 
                                                                       latitude=args.array_lat, 
                                                                       multiprocess=args.multiprocess)
    tsc = time.time() - t0
    timing_info(ftime, 0, "(0) bess_sky_contraction", tsc)
    print(f"bess_sky_contraction took {tsc} seconds")
    

    # Iterate the Gibbs sampler
    print("="*60)
    print("Starting Gibbs sampler (%d iterations)" % args.Niters)
    print("="*60)
    for n in range(args.Niters):
        print("-"*60)
        print(">>> Iteration %4d / %4d" % (n+1, args.Niters))
        print("-"*60)
        t0iter = time.time()

        for ant_samp_ind in range(Nants):
            bess_trans = hydra.beam_sampler.get_bess_to_vis_from_contraction(bess_sky_contraction,
                                                                             beam_coeffs, 
                                                                             ants, 
                                                                             ant_samp_ind)
            
            inv_noise_var_use = hydra.beam_sampler.select_subarr(inv_noise_var[None, None], # add pol axes of length 1
                                                                 ant_samp_ind, 
                                                                 Nants)
            data_use = hydra.beam_sampler.select_subarr(data[None, None], ant_samp_ind, Nants)

            # Construct RHS vector
            rhs_unflatten = hydra.beam_sampler.construct_rhs(data_use,
                                                             inv_noise_var_use,
                                                             coeff_mean,
                                                             bess_trans,
                                                             cov_tuple,
                                                             cho_tuple)
            bbeam = rhs_unflatten.flatten()
                

            shape = (args.Nfreqs, args.Nbasis,  1, 1, 2)
            cov_Qdag_Ninv_Q = hydra.beam_sampler.get_cov_Qdag_Ninv_Q(inv_noise_var_use,
                                                                     bess_trans,
                                                                     cov_tuple)
            
            axlen = np.prod(shape)

            # fPbpQBcCF->fbQcFBpPC
            matr = cov_Qdag_Ninv_Q.transpose((0,2,4,6,8,5,3,1,7)).reshape([axlen, axlen]) + np.eye(axlen)

            def beam_lhs_operator(x):
                y = hydra.beam_sampler.apply_operator(np.reshape(x, shape),
                                                      cov_Qdag_Ninv_Q)
                return(y.flatten())

            # What the shape would be if the matrix were represented densely
            beam_lhs_shape = (axlen, axlen)
            print("Solving")
            x_soln = np.linalg.solve(matr, bbeam)
            print("Done solving")


            if args.test_close:
                btest = matr @ x_soln
                allclose = np.allclose(btest, bbeam)
                if not allclose:
                    abs_diff = np.abs(btest-bbeam)
                    wh_max_diff = np.argmax(abs_diff)
                    max_diff = abs_diff[wh_max_diff]
                    max_val = bbeam[wh_max_diff]
                    raise AssertionError(f"btest not close to bbeam, max_diff: {max_diff}, max_val: {max_val}")
            x_soln_res = np.reshape(x_soln, shape)

            # Has shape Nfreqs, Nbasis, Npol, Npol, ncomp
            # Want shape Nbasis, Nfreqs, Npol, Npol, ncomp
            x_soln_swap = np.swapaxes(x_soln_res, 0, 1)

            # Update the coeffs between rounds
            beam_coeffs[:, :, ant_samp_ind] = 1.0 * x_soln_swap[:, :, :, :, 0] \
                                                + 1.j * x_soln_swap[:, :, :, :, 1]
            
        timing_info(ftime, n, "(D) Beam sampler", time.time() - t0)
        np.save(os.path.join(output_dir, "beam_%05d" % n), beam_coeffs)


     

    
