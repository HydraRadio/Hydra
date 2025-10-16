#!/usr/bin/env python

import time, os

import numpy as np
import hydra

from scipy.stats import norm, rayleigh
from scipy.linalg import cholesky
from hydra.utils import timing_info, build_hex_array, get_flux_from_ptsrc_amp, \
                         convert_to_tops
from pyuvdata.analytic_beam import GaussianBeam, AiryBeam
from pyuvdata import UVBeam, BeamInterface

import argparse
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines

from .beam_example_utils import run_vis_sim, perturbed_beam, get_analytic_beam

if __name__ == '__main__':

    description = "Example Gibbs sampling of the joint posterior of beam "  \
                  "parameters from a simulated visibility data set " 
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--beam-seed", type=int, action="store", default=1001,
                        required=False, dest="beam_seed",
                        help="Set the random seed.")
    parser.add_argument("--chain-seed", type=str, action="store", default="None",
                        required=False, dest="chain_seed", 
                        help="Set a separate seed for initializing the Gibbs chain")
    parser.add_argument("--sky-seed", type=int, action="store", default=654,
                        dest="sky_seed")
    parser.add_argument("--noise-seed", type=int, required=False, default=7254,
                        dest="noise_seed")
    
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
    parser.add_argument("--roundup", action="store", type=int, required=False,
                        default=1000, 
                        help="How often to round files up and save them to one big file")
    
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
    parser.add_argument("--output-dir", type=str, action="store",
                        default="./output", required=False, dest="output_dir",
                        help="Output directory.")
    parser.add_argument("--anneal", action="store_true", required=False,
                        help="Slowly shift the weight between sampling form the prior and posterior over the course of many iterations.")
    parser.add_argument("--infnoise", action="store_true", required=False,
                        help="Sets the inverse noise variance to 0 in order to draw samples from the prior.")
    parser.add_argument("--perts-only", action="store_true", required=False,
                        dest="perts_only",
                        help="Only constrain perturbations rather than the full beam shape")
    parser.add_argument("--missing-sources", required=False, 
                        action="store_true", dest="missing_sources",
                        help="Whether to drop the bottom 10 percent of sources when inferring the beam")
    
    # Point source sim params
    parser.add_argument("--ra-bounds", type=float, action="store", default=(0, 2*np.pi),
                        nargs=2, required=False, dest="ra_bounds",
                        help="Bounds for the Right Ascension of the randomly simulated sources")
    parser.add_argument("--dec-bounds", type=float, action="store", 
                        default=(-np.pi/2, np.pi/2.),
                        nargs=2, required=False, dest="dec_bounds",
                        help="Bounds for the Declination of the randomly simulated sources")
    parser.add_argument("--lst-bounds", type=float, action="store", 
                        default=(np.pi/2, np.pi),
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
    parser.add_argument("--decent-prior", action="store_true", required=False, dest="decent_prior")
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
                    default=0, dest="trans_std",
                    help="Standard deviation for random tilt of beam")
    parser.add_argument("--rot-std-deg", required=False, type=float, 
                        dest="rot_std_deg", default=0., 
                        help="Standard deviation for random beam rotation, in degrees.")
    parser.add_argument("--stretch-std", required=False, type=float, 
                        dest="stretch_std", default=1e-2, 
                        help="Standard deviation for random beam stretching.")
    parser.add_argument("--beam-type", required=False, type=str,
                        dest="beam_type", default="gaussian")
    parser.add_argument("--pca-modes", required=False, type=str,
                        dest="pca_modes", help="Path to saved PCA eigenvectors.")
    parser.add_argument("--per-ant", required=False, action="store_true",
                        dest="per_ant",
                        help="Whether to use a different beam per antenna")
    parser.add_argument("--csl", required=False, type=float, default=0.2,
                        help="Sidelobe modulation amplitude")
    
    args = parser.parse_args()
    
    if args.chain_seed == "None":
        chain_seed = None
    else:
        chain_seed = int(args.chain_seed)

    hex_array = tuple(args.hex_array)
    assert len(args.hex_array) == 2, "hex-array argument must have length 2."

    # In case these are passed out of order, also shorter names
    ra_low, ra_high = (min(args.ra_bounds), max(args.ra_bounds))
    dec_low, dec_high = (min(args.dec_bounds), max(args.dec_bounds))
    lst_min, lst_max = (min(args.lst_bounds), max(args.lst_bounds))

    # Convert to radians
    array_lat = np.deg2rad(args.array_lat)

    if "vivaldi" in args.beam_file.lower():
        unpert_beam = "vivaldi"
    else:
        unpert_beam = "dipole"

    # Check that output directory exists
    output_dir = f"{args.output_dir}/per_ant/{args.per_ant}/beam_type/{args.beam_type}"
    output_dir = f"{output_dir}/unpert_beam/{unpert_beam}/Nptsrc/{args.Nptsrc}/Ntimes/{args.Ntimes}"
    output_dir = f"{output_dir}/Nfreqs/{args.Nfreqs}/Nbasis/{args.Nbasis}"
    output_dir = f"{output_dir}/prior_std/{args.beam_prior_std}"
    output_dir = f"{output_dir}/decent_prior/{args.decent_prior}/csl/{args.csl}"
    output_dir = f"{output_dir}/missing_sources/{args.missing_sources}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("\nOutput directory:", output_dir)

    # Random seed
    beam_rng = np.random.default_rng(args.beam_seed)
    print("    Seed:    %d" % args.beam_seed)

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
    for i in range(Nants):
        for j in range(i, Nants):
            if i != j:
                # Exclude autos
                antpairs.append((i,j))


    ants1, ants2 = list(zip(*antpairs))

    skyrng = np.random.default_rng(args.sky_seed)
    # Generate random point source locations
    # RA goes from [0, 2 pi] and Dec from [-pi / 2, +pi / 2].
    ra = skyrng.uniform(low=ra_low, high=ra_high, size=args.Nptsrc)
    
    # inversion sample to get them uniform on the sphere, in case wide bounds are used
    U = skyrng.uniform(low=0, high=1, size=args.Nptsrc)
    dsin = np.sin(dec_high) - np.sin(dec_low)
    dec = np.arcsin(U * dsin + np.sin(dec_low)) # np.arcsin returns on [-pi / 2, +pi / 2]

    # Generate fluxes
    beta_ptsrc = -2.7
    ptsrc_amps = 10.**skyrng.uniform(low=-1., high=2., size=args.Nptsrc)
    fluxes = get_flux_from_ptsrc_amp(ptsrc_amps, freqs * 1e-6, beta_ptsrc) # Have to put this in MHz...
    print("pstrc amps (input):", ptsrc_amps[:5])
    np.save(os.path.join(output_dir, "ptsrc_amps0"), ptsrc_amps)
    np.save(os.path.join(output_dir, "ptsrc_coords0"), np.column_stack((ra, dec)).T)

    mmodes = np.arange(-args.mmax, args.mmax + 1)
    unpert_sb = hydra.sparse_beam.sparse_beam(args.beam_file, nmax=args.nmax, 
                                              mmodes=mmodes, Nfeeds=2, 
                                              alpha=args.rho_const,
                                              num_modes_comp=args.Nbasis,
                                              sqrt=args.per_ant,
                                              freq_range=(np.amin(freqs), np.amax(freqs)),
                                              save_fn=f"{output_dir}/unpert_sb")

    if args.per_ant:
        beams = []
        for ant_ind in range(Nants):
            ref_cond = args.perts_only and ant_ind == 0
            if args.beam_type == "pert_sim":
                pow_sb = perturbed_beam(
                    args, 
                    output_dir, 
                    seed=args.beam_seed + ant_ind
                )        
                beams.append(pow_sb)
                if ref_cond:
                    ref_beam = UVBeam.from_file(args.beam_file)
                    ref_beam.peak_normalize()
            elif args.beam_type in ["gaussian", "airy"]:
                # Underilluminated HERA dishes
                beam_rng = np.random.default_rng(seed=args.beam_seed + ant_ind)
                beam, beam_class = get_analytic_beam(args, beam_rng)
                beams.append(beam)
                if ref_cond:
                    ref_beam = beam_class(diameter=12.)
            else:
                raise ValueError("beam-type arg must be one of ('gaussian', 'airy', 'pert_sim')")
    else:
        if args.beam_type == "unpert":
            bm = UVBeam.from_file(args.beam_file)
            bm.peak_normalize()
            beams = Nants * [bm]
        elif args.beam_type == "pert_sim":
            beam_rng = np.random.default_rng(seed=args.beam_seed)
            pow_sb = perturbed_beam(
                args, 
                output_dir, 
                seed=None,
                sin_pert_coeffs=beam_rng.normal(size=8),
                stretch_x = 1.01,
                stretch_y = 1.02
            )
            beams = Nants * [pow_sb]
        elif args.beam_type in ["gaussian", "airy"]:
            beam_rng = np.random.default_rng(seed=args.beam_seed)
            beam, beam_class = get_analytic_beam(args, beam_rng)
            beams = Nants * [beam]

    sim_outpath = os.path.join(output_dir, "model0.npy")
    _sim_vis = run_vis_sim(args, ftime, times, freqs, ant_pos, Nants,
                           ra, dec, fluxes, beams, array_lat, sim_outpath)
    if args.beam_type == "pert_sim":
        unpert_sim_outpath = os.path.join(output_dir, "model_unpert.npy")
        unpert_beam_UVB = UVBeam.from_file(args.beam_file)
        unpert_beam_UVB.peak_normalize()
        unpert_beam_list = Nants * [unpert_beam_UVB]
        if args.missing_sources:
            amps_inference = np.copy(ptsrc_amps)
            amps_inference[amps_inference < 1e0] = 0
            flux_inference = get_flux_from_ptsrc_amp(
                amps_inference, 
                freqs * 1e-6, 
                beta_ptsrc
            )
        else:
            flux_inference = fluxes
        unpert_vis = run_vis_sim(args, ftime, times, freqs, ant_pos, Nants,
                                ra, dec, flux_inference, unpert_beam_list, array_lat, unpert_sim_outpath, ref=True)

    autos = np.abs(_sim_vis[:, :, np.arange(Nants), np.arange(Nants)])
    noise_var = autos[:, :, None] * autos[:, :, :, None] / (args.integration_depth * args.ch_wid)

    #FIXME: technically we need the conjugate noise rzn on conjugate baselines...
    noise_rng = np.random.default_rng(args.noise_seed)
    noise = (noise_rng.normal(scale=np.sqrt(noise_var)) + 1.j * noise_rng.normal(scale=np.sqrt(noise_var))) / np.sqrt(2)
    data = _sim_vis + _sim_vis.swapaxes(-1,-2).conj() + noise # fix some zeros
    del _sim_vis # Save some memory
    del noise

    np.save(os.path.join(output_dir, "data0"), data)
    if args.perts_only: # Subtract off the vis. made with ref beam
        ref_sim_outpath = os.path.join(output_dir, "model0_ref.npy")
        ref_beams = Nants * [ref_beam]
        ref_beam_vis = run_vis_sim(args, ftime, times, freqs, ant_pos, 
                                  Nants, ra, dec, fluxes, ref_beams, sim_outpath)
        data -= (ref_beam_vis + ref_beam_vis.swapaxes(-1, -2).conj())
        del ref_beam_vis
        np.save(os.path.join(output_dir, "data_res"), data)

    inv_noise_var = 1/noise_var
    np.save(os.path.join(output_dir, "inv_noise_var.npy"), inv_noise_var)

    txs, tys, tzs = convert_to_tops(ra, dec, times, array_lat)

    za = np.arccos(tzs).flatten()
    az = np.arctan2(tys, txs).flatten()
    np.save(os.path.join(output_dir, "za.npy"), za)
    np.save(os.path.join(output_dir, "az.npy"), az)
    
    bess_matr, trig_matr = unpert_sb.get_dmatr_interp(az, 
                                                      za)
    bess_matr = bess_matr.reshape(args.Ntimes, args.Nptsrc, args.nmax)
    trig_matr = trig_matr.reshape(args.Ntimes, args.Nptsrc, 2 * args.mmax + 1)
    comp_inds = unpert_sb.get_comp_inds()
    nmodes = comp_inds[0][:, 0, 0, 0]
    mmodes = comp_inds[1][:, 0, 0, 0]
    np.save(
        os.path.join(output_dir, "nmodes.npy"),
        nmodes
    )
    np.save(
        os.path.join(output_dir, "mmodes.npy"),
        mmodes
    )
    per_source_Dmatr_out = os.path.join(output_dir, "Dmatr.npy")
    if not os.path.exists(per_source_Dmatr_out):
        if args.beam_type == "pert_sim" and args.per_ant: # Use PCA-based basis for sqrt(power beam) in a per antenna way
            # Need a mode for the mean of the prior. 
            # Use the FB coeffs for the frequency closest to the center frequency of the simulation.
            mid_freq = freqs[args.Nfreqs // 2]
            closest_chan = np.argmin(np.abs(mid_freq - pow_sb.freq_array))
            mean_mode = unpert_sb.bess_fits[:, :, 0, 0, 0, closest_chan]
            # Normalize it to have L^2 norm of 1
            mean_mode = mean_mode / np.sqrt(np.sum(np.abs(mean_mode)**2))
            # Load the PCA modes saved on disk ahead of time -- they are expressed in FB space
            pca_modes = np.load(args.pca_modes)[:, :args.Nbasis - 1].reshape(args.nmax, 2 * args.mmax + 1, args.Nbasis - 1) 
            # Make the full basis including the mean mode
            Pmatr = np.concatenate([mean_mode[:, :, None], pca_modes], axis=2)
            # Contract with FB design matrices so that the map goes from PCA -> source coordinates
            BPmatr = np.tensordot(bess_matr, Pmatr, axes=1).transpose(3, 0, 1, 2) # Nbasis, Ntimes, Nsrc, Naz
            # Matrix evaluating subset of FB modes at source coordinates
            Dmatr = np.sum(BPmatr * trig_matr, axis=3).transpose(1, 2, 0) # Ntimes, Nsrc, Nbasis
        elif args.beam_type in ["unpert", "pert_sim"]: # Use FB modes as in paper I/II
            # Subset of Bessel and Fourier design matrices corresponding to compression recipe from paper I
            bsparse = bess_matr[:, :, nmodes[:args.Nbasis]]
            tsparse = trig_matr[:, :, mmodes[:args.Nbasis]]
            # Matrix evaluating subset of FB modes at source coordinates
            Dmatr = bsparse * tsparse
        else: # Using analytic beams, only use radial modes
            Dmatr = bess_matr[:, :, :args.Nbasis]
            Q, R = np.linalg.qr(unpert_sb.bess_matr[:, :args.Nbasis]) # orthoganalize radial modes...
            reshape = (args.Ntimes * args.Nptsrc, args.Nbasis)
            shape = (args.Ntimes, args.Nptsrc, args.Nbasis)
            # Get the orthogonalized basis evaluated at the source coordinates
            Dmatr = np.linalg.solve(R.T, Dmatr.reshape(reshape).T).T.reshape(shape)
        np.save(per_source_Dmatr_out, Dmatr)
    else:
        Dmatr = np.load(per_source_Dmatr_out)

    
    # Have everything we need to analytically evaluate single-array beam
    if args.per_ant: 
        Dmatr_outer = hydra.beam_sampler.get_bess_outer(Dmatr)
        np.save(os.path.join(output_dir, "dmo.npy"), Dmatr_outer)
    
        beam_coeffs = np.zeros([Nants, args.Nfreqs, args.Nbasis, 1, 1])
        if not args.perts_only:
            if args.decent_prior:
                # FIXME: Hardcode!, start at answer!
                beam_coeffs += np.load("data/14m_airy_bessel_soln.npy")[None, None, :, None, None] 
            else:
                beam_coeffs[:, :, 0] = 1
            # Want shape Nbasis, Nfreqs, Nants, Npol, Npol
        beam_coeffs = np.swapaxes(beam_coeffs, 0, 2).astype(complex)

        sig_freq = 0.1 * (freqs[-1] - freqs[0])
        # FIXME: Hardcode!
        cov_file = "data/ramped_variance_bessel_cov.npy" if args.decent_prior else None
        cov_tuple = hydra.beam_sampler.make_prior_cov(freqs, args.beam_prior_std,
                                                    sig_freq, args.Nbasis,
                                                    ridge=1e-6,
                                                    cov_file=cov_file)
        cho_tuple = hydra.beam_sampler.do_cov_cho(cov_tuple, check_op=False)

        bsc_outpath = os.path.join(output_dir, "bsc.npy")
        if os.path.exists(bsc_outpath):
            bess_sky_contraction = np.load(bsc_outpath)
        else:
            t0 = time.time()
            bess_sky_contraction = hydra.beam_sampler.get_bess_sky_contraction(Dmatr_outer, 
                                                                            ant_pos, 
                                                                            fluxes, 
                                                                            ra,
                                                                            dec, 
                                                                            freqs, 
                                                                            times,
                                                                            polarized=False, 
                                                                            latitude=array_lat,)
            np.save(bsc_outpath, bess_sky_contraction)
            tsc = time.time() - t0
            timing_info(ftime, 0, "(0) bess_sky_contraction", tsc)
            print(f"bess_sky_contraction took {tsc} seconds")

        if args.perts_only:
            ref_contraction_outpath = os.path.join(output_dir, "ref_constraction.npy")
            if os.path.exists(ref_contraction_outpath):
                ref_contraction = np.load(ref_contraction_outpath)
            else:
                ref_beam_response = BeamInterface(ref_beam, beam_type="power")
                # FIXME: Hardcode feed x
                ref_beam_response = ref_beam_response.with_feeds(["x"])
                ref_beam_response = ref_beam_response.compute_response(az_array=az,
                                                                    za_array=za,
                                                                    freq_array=freqs)
                ref_beam_response = ref_beam_response.reshape([args.Nfreqs,
                                                            args.Ntimes,
                                                            args.Nptsrc,])
                # Square root of power beam
                ref_beam_response = np.sqrt(ref_beam_response)
                ref_contraction = hydra.beam_sampler.get_bess_sky_contraction(Dmatr, 
                                                                            ant_pos, 
                                                                            fluxes, 
                                                                            ra,
                                                                            dec, 
                                                                            freqs, 
                                                                            times,
                                                                            polarized=False, 
                                                                            latitude=array_lat,
                                                                            outer=False,
                                                                            ref_beam_response=ref_beam_response)

                np.save(ref_contraction_outpath, ref_contraction)
            coeff_mean = np.zeros_like(beam_coeffs[:, :, 0])
        else:
            # Be lazy and just use the initial guess -- either the right answer or 1 in the first basis function.
            coeff_mean = np.copy(beam_coeffs[:, :, 0])
        # shuffle the initial position by pulling from the prior
        if chain_seed is not None: 
            chain_rng = np.random.default_rng(seed=chain_seed)
            beam_coeffs = chain_rng.normal(loc=coeff_mean[:, :, None].real,
                                        scale=args.beam_prior_std,
                                        size=beam_coeffs.shape).astype(complex)


        # Iterate the Gibbs sampler
        print("="*60)
        print("Starting Gibbs sampler (%d iterations)" % args.Niters)
        print("="*60)
        for n in range(args.Niters):
            print("-"*60)
            print(">>> Iteration %4d / %4d" % (n+1, args.Niters))
            print("-"*60)
            t0iter = time.time()

            if args.anneal:
                temp = max(2000. - 2 * n, 1.)
            else:
                temp = 1.

            for ant_samp_ind in range(Nants):
                bess_trans = hydra.beam_sampler.get_bess_to_vis_from_contraction(bess_sky_contraction,
                                                                                 beam_coeffs, 
                                                                                 ants, 
                                                                                 ant_samp_ind)
                
                inv_noise_var_use = hydra.beam_sampler.select_subarr(inv_noise_var[None, None], # add pol axes of length 1
                                                                     ant_samp_ind, 
                                                                     Nants)
                if args.infnoise:
                    inv_noise_var_use[:] = 0
                else:
                    inv_noise_var_use /= temp
                data_use = hydra.beam_sampler.select_subarr(data[None, None], ant_samp_ind, Nants)
                if args.perts_only:
                    # Contract other antenna coefficients with object that has been pre-multiplied by reference beam
                    other_ants_with_ref = hydra.beam_sampler.get_bess_to_vis_from_contraction(ref_contraction,
                                                                                              beam_coeffs, 
                                                                                              ants, 
                                                                                              ant_samp_ind,
                                                                                              ref_contraction=True)
                    data_use -= other_ants_with_ref
                    ant_inds = hydra.beam_sampler.get_ant_inds(ant_samp_ind, Nants)
                    # Add the term that contracts the reference beam with current antenna's perturbations
                    bess_trans += ref_contraction[:, :, :, :, ant_inds, ant_samp_ind]

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
                
                # FIXME: This is skipped!
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
                
            timing_info(ftime, n, "(D) Beam sampler", time.time() - t0iter)
            np.save(os.path.join(output_dir, "beam_%05d" % n), beam_coeffs)

            np1 = n + 1
            if not np1 % args.roundup:
                print(f"Rounding up files for iteration: {np1}")
                files_to_round_up = glob.glob(f"{output_dir}/beam*.npy")
                sorted_files = sorted(files_to_round_up)
                sample_arr = np.zeros((args.roundup,) + beam_coeffs.shape)
                for file_ind, file in enumerate(sorted_files):
                    if "roundup" in file:
                        continue
                    sample_arr[file_ind] = np.load(file)
                    os.remove(file)
                np.save(os.path.join(output_dir, f"beam_roundup_{np1}"), sample_arr)
    else:
        pow_beam_Dmatr_outfile = os.path.join(output_dir, "pow_beam_Dmatr.npy")
        if not os.path.exists(pow_beam_Dmatr_outfile):
            Dmatr_start = time.time()
            pow_beam_Dmatr_dense = hydra.beam_sampler.get_bess_sky_contraction(
                Dmatr, 
                ant_pos, 
                flux_inference, 
                ra,
                dec, 
                freqs, 
                times,
                polarized=False, 
                latitude=array_lat,
                outer=False
            )
            Dmatr_end = time.time()
            print(f"Dmatr calculation took {Dmatr_end - Dmatr_start} seconds")


            pow_beam_Dmatr_dense = pow_beam_Dmatr_dense[0, 0]
            np.save(pow_beam_Dmatr_outfile, pow_beam_Dmatr_dense)
        else:
            pow_beam_Dmatr_dense = np.load(pow_beam_Dmatr_outfile)


        triu_inds = np.triu_indices(Nants, k=1)
        pow_beam_Dmatr = pow_beam_Dmatr_dense[
            :, ::2, triu_inds[0], triu_inds[1]
        ] # ftub
        Ninv = inv_noise_var[:, ::2, triu_inds[0], triu_inds[1]] # ftu
        pci_file = os.path.join(output_dir, "post_cov_inv.npy")
        pc_file = os.path.join(output_dir, "post_cov.npy")
        MAP_file = os.path.join(output_dir, "MAP_soln.npy")
        inference_files = [pci_file, pc_file, MAP_file]
        if all([os.path.exists(file) for file in inference_files]):
            LHS = np.load(pci_file)
            post_cov = np.load(pc_file)
            MAP_soln = np.load(MAP_file)
        else:
            if args.decent_prior:
                prior_mean = unpert_sb.comp_fits[0, 0]
                inv_prior_var = 1/(args.beam_prior_std * np.abs(prior_mean))**2 # Fractional uncertainty
                prior_Cinv = np.zeros([args.Nfreqs, args.Nbasis, args.Nbasis],
                                        dtype=complex)
                prior_Cinv = [np.diag(inv_prior_var[chan]) for chan in range(args.Nfreqs)]
                prior_Cinv = np.array(prior_Cinv)
            else:
                prior_Cinv = np.repeat(np.eye(args.Nbasis)[None], args.Nfreqs, axis=0)
                prior_Cinv /= args.beam_prior_std**2
                prior_mean = np.zeros([args.Nfreqs, args.Nbasis], dtype=complex)
            LHS = hydra.power_beam_sampler.construct_LHS(
                pow_beam_Dmatr,
                Ninv,
                prior_Cinv
            )
            # Use every other time step. Reserve other half for PPD check.
            inference_vis = data[:, ::2, triu_inds[0], triu_inds[1]]
            RHS = hydra.power_beam_sampler.construct_RHS(
                pow_beam_Dmatr,
                Ninv,
                prior_Cinv,
                inference_vis,
                prior_mean,
                flx=False
            )

            post_cov = np.linalg.inv(LHS)
            MAP_soln = np.linalg.solve(LHS, RHS[:, :, None])[:, :, 0]
            

            np.save(pci_file, LHS)
            np.save(MAP_file, MAP_soln)
            np.save(pc_file, post_cov)

        # Make matrix for transforming to image space.
        sparse_bmatr = unpert_sb.bess_matr[:, nmodes[:args.Nbasis]]
        sparse_tmatr = unpert_sb.trig_matr[:, mmodes[:args.Nbasis]]
        sparse_dmatr_recon = sparse_bmatr[:, None] * sparse_tmatr[None, :]

        ##########################################
        # Below here is just a bunch of plotting #
        ##########################################

        
        # Show image space projection of beam.
        # fb,zab->fza but without einsum as the middleman
        MAP_beam = np.tensordot(MAP_soln,
                                sparse_dmatr_recon,
                                axes=((-1,), (-1,)))

        midchan = args.Nfreqs // 2
        plotbeam = MAP_beam[midchan]
        np.save(os.path.join(output_dir, "MAP_beam.npy"), plotbeam)
        Az, Za = np.meshgrid(unpert_sb.axis1_array, unpert_sb.axis2_array)
        if args.missing_sources:
            np_attr = "abs"
        else:
            np_attr = "real"
        beam_color_scale = {"vmin": 1e-4, "vmax": 1}
        residual_color_scale = {"vmin": -1e-2, "vmax": 1e-2, "linthresh": 1e-4}

        fig = plt.figure(figsize=[6.5, 7])
        gs = GridSpec(3, 2)
        ax = np.empty([2, 2], dtype=object)
        for row_ind in range(2):
            for col_ind in range(2):
                ax[row_ind, col_ind] = fig.add_subplot(
                    gs[row_ind, col_ind],
                    projection="polar"
                )
        im = ax[0, 0].pcolormesh(
            Az,
            Za * 180/np.pi,
            plotbeam.real,
            norm=LogNorm(**beam_color_scale),
            cmap="inferno",
        )
        ax[0, 0].set_title("MAP Beam")
        fig.colorbar(im, ax=ax[0,0])

        image_var = np.einsum("bB,azb,azB->az",
                              post_cov[midchan],
                              sparse_dmatr_recon,
                              sparse_dmatr_recon.conj(),
                              optimize=True)
        image_std = np.sqrt(np.abs(image_var))
        im = ax[0, 1].pcolormesh(
            Az,
            Za * 180/np.pi,
            image_std,
            norm=LogNorm(),
            cmap="inferno"
        )
        ax[0, 1].set_title("Posterior uncertainty")
        fig.colorbar(im, ax=ax[0,1])

        if args.beam_type == "pert_sim":
            input_beam, _ = pow_sb.interp(
                az_array=Az.flatten(),
                za_array=Za.flatten(),
                freq_array=freqs,
            )
            input_beam = input_beam[0, 0, midchan].reshape(Az.shape)
        else:
            input_beam = unpert_sb.data_array[0, 0, midchan]
        errors = (input_beam - plotbeam)
        im = ax[1, 0].pcolormesh(
            Az,
            Za * 180/np.pi,
            errors.real,
            norm=SymLogNorm(**residual_color_scale),
            cmap="Spectral",
        )
        ax[1, 0].set_title("MAP Errors")
        fig.colorbar(im, ax=ax[1, 0])
        image_z = np.abs(errors)/image_std
        im = ax[1, 1].pcolormesh(
            Az,
            Za * 180/np.pi,
            image_z,
            norm=LogNorm(),
            cmap="inferno",
        )
        ax[1, 1].set_title("$z$ score")
        fig.colorbar(im, ax=ax[1,1])

        for row_ind in range(2):
            for col_ind in range(2):
                ax_ob = ax[row_ind, col_ind]
                if (row_ind == 1) and (col_ind == 0):
                    gridcolor="black"
                else:
                    gridcolor="white"
                adjust_beamplot(ax_ob, gridcolor=gridcolor)
        line_ax = fig.add_subplot(gs[2, :])
        beam_obs = [input_beam, getattr(np, np_attr)(plotbeam)]
        beam_labels = ["Perturbed Beam", "MAP Beam"]
        plot_beam_slice(line_ax, beam_obs, beam_labels)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "reconstruction_residual_plot.pdf"),
                    bbox_inches="tight")

        fig, ax = plt.subplots(figsize=[3.25, 3.25])
        _, bins, _ = ax.hist(
            image_z.flatten(), 
            bins="auto", 
            histtype="step",
            density=True
        )
        rayl_x = np.linspace(0, 10, num=100)
        rayl = rayleigh.pdf(rayl_x, scale=1/np.sqrt(2))
        ax.plot(rayl_x, rayl, linestyle="--", color="black")
        ax.set_xlabel("|z|")
        ax.set_ylabel("Probability Density")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "image_z_score.pdf"),
                    bbox_inches="tight")


        if args.beam_type == "pert_sim":
            fig = plt.figure(figsize=[6.5, 6.5])
            gs = GridSpec(2, 2)
            unpert_ax = fig.add_subplot(gs[0, 0],
                                        projection="polar")
            unpert_beam = unpert_sb.data_array[0, 0, midchan]
            im = unpert_ax.pcolormesh(
                Az,
                Za * 180/np.pi,
                unpert_beam,
                norm=LogNorm(**beam_color_scale),
                cmap="inferno",
            )
            unpert_ax.set_title("Unperturbed Beam")
            adjust_beamplot(unpert_ax)
            fig.colorbar(im, ax=unpert_ax)

            pert_ax = fig.add_subplot(gs[0, 1],
                                      projection="polar")
            im = pert_ax.pcolormesh(
                Az,
                Za * 180/np.pi,
                (input_beam - unpert_beam).real,
                norm=SymLogNorm(**residual_color_scale),
                cmap="Spectral",
            )
            pert_ax.set_title("Perturbations")
            adjust_beamplot(pert_ax, gridcolor="black")
            fig.colorbar(im, ax=pert_ax)

            line_ax = fig.add_subplot(gs[1, :])
            beam_obs = [unpert_beam, input_beam]
            beam_labels = ["Unperturbed", "Perturbed"]
            plot_beam_slice(line_ax, beam_obs, beam_labels)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "input_residual_plot.pdf"),
                        bbox_inches="tight")

            

        PPD_Dmatr = pow_beam_Dmatr_dense[:, 1::2, triu_inds[0], triu_inds[1]]

        postdicted_mean = np.einsum(
            "ftub,fb->ftu",
            PPD_Dmatr,
            MAP_soln,
            optimize=True
        )
        def get_z_scores(model, post_pred=False):
            if post_pred:
                var_post = np.einsum(
                    "ftub,fbB,ftuB->ftu",
                    PPD_Dmatr,
                    post_cov,
                    PPD_Dmatr.conj(),
                    optimize=True
                )
                var_ppd = 1/Ninv + np.abs(var_post)
                isig = np.sqrt(2 / var_ppd)
            else:
                isig = np.sqrt(2 * Ninv)

            PPD_data = data[:, 1::2, triu_inds[0], triu_inds[1]]
            zscore = (PPD_data - model) * isig
            zreal = zscore.real.flatten()
            zimag = zscore.imag.flatten()
            to_hist = np.array([zreal, zimag]).T

            return to_hist
        to_hist = get_z_scores(unpert_vis[:, 1::2, triu_inds[0], triu_inds[1]])
        to_hist_ppd = get_z_scores(postdicted_mean, post_pred=True)

        fig, ax = plt.subplots(figsize=(6.5, 3), ncols=2)
        bins = np.linspace(-10, 10, num=100)
        counts, _, _ = ax[0].hist(
            to_hist.flatten(), 
            bins="auto",
            histtype="step",
            density=True,
            label="Unperturbed Beam",
        )
        ax[0].hist(
            to_hist_ppd.flatten(),
            bins="auto", 
            histtype="step", 
            density=True, 
            label="Inferred Beam"
        )

        for ax_ob in ax:
            ax_ob.set_xlabel(r"$z$-score")
            ax_ob.set_ylabel("Probability Density")
        lbins = bins[:-1]
        rbins = bins[1:]
        bin_cent = (lbins + rbins) * 0.5
        pbin = norm.cdf(rbins) - norm.cdf(lbins)
        std_norm_counts = pbin * np.sum(counts)
        line3 = ax[0].plot(
            bin_cent, 
            norm.pdf(bin_cent), 
            linestyle="--", 
            color="black"
        )
        ax[0].set_xlim([-10, 10])

        counts, _, patch1 = ax[1].hist(
            to_hist.flatten(), 
            bins="auto",
            histtype="step",
            density=True,
            label="Unperturbed Beam",
        )
        _, _, patch2 = ax[1].hist(
            to_hist_ppd.flatten(), 
            bins="auto", 
            histtype="step", 
            density=True, 
            label="Inferred Beam"
        )
        ax[1].legend(
            handles=[patch1[0], patch2[0], line3[0]],
            labels=["Unperturbed Beam", "Inferred Beam", r"$\mathcal{N}(0, 1)$"],
            loc="upper left",
            frameon=False
        )
        if not args.missing_sources:
            ax[1].set_ylim([0, 0.5])
        else:
            ax[1].set_ylim([0, 0.03])
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, "residual_hist.pdf"), 
            bbox_inches="tight"
        )

        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        im = ax.matshow(
            np.abs(post_cov[midchan]), 
            cmap="inferno",
            norm=LogNorm()
        )
        ax.set_title("Mode Number")
        ax.set_ylabel("Mode Number")
        fig.colorbar(im, ax=ax, label=r"$|\Sigma_\mathrm{post}|$")
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, "post_cov.pdf"),
            bbox_inches="tight"
        )

        fig, ax = plt.subplots(figsize=(3.25, 6.25), nrows=2)
        mode_numbers = np.arange(1, args.Nbasis + 1)
        FB_stds = np.sqrt(np.abs(np.diag(post_cov[midchan])))
        these_comp_fits = unpert_sb.comp_fits[0, 0, midchan]
        z_update = np.abs((MAP_soln[midchan] - these_comp_fits))/FB_stds
        ax[0].plot(
            mode_numbers,
            np.abs(MAP_soln[midchan]), 
            color="lightcoral",
            label="MAP Beam"
        )
        ax[0].plot(
            mode_numbers,
            FB_stds,
            color="goldenrod",
            label="Posterior Std."
        )
        ax[0].plot(
            mode_numbers,
            np.abs(unpert_sb.comp_fits[0,0,0]),
            linestyle=":",
            color="black",
            label="Prior Std."
        )
        ax[1].plot(
            mode_numbers,
            z_update,
            color="lightcoral",
        )
        ax[1].set_xlabel("Mode Number")
        ax[0].set_ylabel(r"$|\mu_\mathrm{post}|$")
        ax[1].set_ylabel(r"$|z_\mathrm{update}|$")
        for ax_ob in ax:
            ax_ob.set_yscale("log")
            # ax_ob.set_xscale("log")
        ax[0].legend(frameon=False)
        ax[0].tick_params(
            which="both", 
            axis="x", 
            direction="in", 
            labelbottom=False
        )
        ax[1].tick_params(which="both", top=True, direction="in")
        fig.tight_layout(h_pad=0)
        fig.savefig(os.path.join(output_dir, "FB_coeff_lines.pdf"),
                    bbox_inches="tight")

        eval_file = os.path.join(output_dir, "evals.npy")
        evec_file = os.path.join(output_dir, "evecs.npy")
        eig_files = [eval_file, evec_file]
        if all([os.path.exists(file) for file in eig_files]):
            evals = np.load(eval_file)
            evecs = np.load(evec_file)
        else:
            evals, evecs = np.linalg.eig(post_cov[midchan])
            np.save(
                eval_file, evals
            )

            np.save(
                evec_file, evecs
            )
        fig, ax = plt.subplots(figsize=[3.25, 3.25])
        ax.plot(mode_numbers, evals.real, color="goldenrod")
        #ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("Eigenvalues")
        ax.set_xlabel("Mode Number")
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, "cov_evals.pdf"),
            bbox_inches="tight"
        )
        fig, ax = plt.subplots(figsize=[3.25, 3.25])
        ax.hist(z_update[200:], bins="auto", histtype="step", density=True)
        ax.plot(rayl_x, rayl, linestyle="--", color="black")
        ax.set_xlabel(r"$|z_\mathrm{update}|$")
        ax.set_ylabel("Probability Density")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "z_update_hist.pdf"), bbox_inches="tight")

        eval_sorter = np.argsort(evals)
        special_ind1 = 2
        special_ind2 = 3
        evec_1 = evecs[:, eval_sorter][:, special_ind1]
        evec_2 = evecs[:, eval_sorter][:, special_ind2]

        fig, ax = plt.subplots(
            figsize=[6.5, 6.5],
            nrows=2,
            ncols=2,
            subplot_kw={"projection": "polar"}
        )

        min_evec_sky = np.tensordot(evec_1,
                                    sparse_dmatr_recon,
                                    axes=((-1,), (-1,)))
        max_evec_sky = np.tensordot(evec_2,
                                    sparse_dmatr_recon,
                                    axes=((-1,), (-1,)))
        evec_labels = [f"Eigenvector {special_ind1 + 1}", 
                       f"Eigenvector {special_ind2 + 1}"]
        for evec_ind, evec in enumerate([min_evec_sky, max_evec_sky]):
            for comp_ind, comp in enumerate(["real", "imag"]):
                ax_ob = ax[evec_ind, comp_ind]
                im = ax_ob.pcolormesh(
                    Az,
                    Za * 180./np.pi,
                    getattr(np, comp)(evec),
                    norm=SymLogNorm(vmin=-1, vmax=1, linthresh=1e-3),
                    cmap="Spectral"
                )
                ax_ob.set_title(f"{evec_labels[evec_ind]} ({comp})")
                adjust_beamplot(ax_ob, gridcolor="black")
        fig.tight_layout()
        fig.colorbar(im, ax=ax.ravel().tolist(), label="Beam Value")
        fig.savefig(
            os.path.join(output_dir, "evec_sky.pdf"),
            bbox_inches="tight"
        )


        fig, ax = plt.subplots(
            nrows=10, 
            ncols=2, 
            figsize=[6.5, 32.5], 
            subplot_kw={"projection": "polar"}
        )
        for evec_ind in range(10):
            for comp_ind, comp in enumerate(["real", "imag"]):
                evec_FB = evecs[:, eval_sorter][:, evec_ind]
                evec = np.tensordot(evec_FB,
                                    sparse_dmatr_recon,
                                    axes=((-1,), (-1,)))
                ax_ob = ax[evec_ind, comp_ind]
                im = ax_ob.pcolormesh(
                    Az,
                    Za * 180./np.pi,
                    getattr(np, comp)(evec),
                    norm=SymLogNorm(vmin=-1, vmax=1, linthresh=1e-3),
                    cmap="Spectral"
                )
                ax_ob.set_title(f"eval {evec_ind} ({comp})")
                adjust_beamplot(ax_ob, gridcolor="black")      
        fig.tight_layout()
        fig.colorbar(im, ax=ax.ravel().tolist(), label="Beam Value")
        fig.savefig(
            os.path.join(output_dir, "smallest_evecs.pdf"),
            bbox_inches="tight"
        )

        hydra.beam_sampler.plot_FB_beam(
            plotbeam,
            unpert_sb.axis2_array,
            unpert_sb.axis1_array, 
            save=True,
            fn=os.path.join(output_dir, "beam_real_imag.pdf"),
            linthresh=1e-4
        )


        

        
