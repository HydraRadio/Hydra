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

def run_vis_sim(args, ftime, times, freqs, ant_pos, Nants, ra, dec,
                fluxes, beams, array_lat, sim_outpath, ref=False):
    """
    Do the visibility simulations

    Parameters:
        args (Namespace):
            Arguments that were parsed with argparse. Will specifically make use
            of Nfreqs, Ntimes, and beam_type.
        ftime (str):
            Path to timing file.
        times (array):
            LSTs to simulate (radians)
        freqs (array):
            Frequencies to simulate, in Hz
        ant_pos (dict):
            Dictionary of antenna positions.
        Nants (int):
            Number of antennas.
        ra (array):
            Right ascension of sources.
        dec (array):
            Declination of sources.
        fluxes (array):
            Flux density of sources.
        beams (List of UVBeam, AnalyticBeam, BeamInterface):
            Per-antenna beam objects.
        array_lat (float):
            The array latitude _in radians_
        sim_outpath (str):
            Path to output file.
        ref (bool):
            Whether this is a 'reference sim' based on a reference beam.
    Returns:
        _sim_vis (array, complex):
            Simulated visibilities with shape (Nfreqs, Ntimes, Nants, Nants).
            (Upper triangle?)
    """
    if not os.path.exists(sim_outpath):
        # Run a simulation
        t0 = time.time()
        _sim_vis = np.zeros([args.Nfreqs, args.Ntimes, Nants, Nants],
                            dtype=complex)
        for tind in range(args.Ntimes):
            _sim_vis[:, tind:tind + 1] =  hydra.vis_simulator.simulate_vis(
                    ants=ant_pos,
                    fluxes=fluxes,
                    ra=ra,
                    dec=dec,
                    freqs=freqs, # Make sure this is in Hz!
                    lsts=times[tind:tind + 1],
                    beams=beams,
                    polarized=False,
                    precision=2,
                    latitude=array_lat,
                    use_feed="x",
                    force_no_beam_sqrt=False,
                )
            if (args.beam_type == "pert_sim") and (not ref):
                for beam in beams:
                    beam.clear_cache() # Otherwise memory gets gigantic
        timing_info(ftime, 0, "(0) Simulation", time.time() - t0)
        np.save(sim_outpath, _sim_vis)
    else:
        _sim_vis = np.load(sim_outpath)
    return _sim_vis

def perturbed_beam(
        args, 
        output_dir, 
        seed=None,
        trans_x=0.,
        trans_y=0.,
        rot=0.,
        stretch_x=0.,
        stretch_y=0.,
        sin_pert_coeffs=np.zeros(8, dtype=float)
):
    """
    Make a perturbed beam. A file named perturbed_beam_beamvals_seed_{seed}.npy
    will be saved in the output_dir.
    
    Parameters:
        args (Namespace):
            Arguments that were parsed with argparse. Specifically makes use of
            beam_file, mmax, nmax, per_ant, Nbasis, csl. If seed is not None,
            will make use of trans_std, rot_std_deg, stretch_std, for random
            beam generation.
        output_dir (str):
            Path to output directory.
        seed (int):
            Random seed for perturbation generation. Will ignore trans_x, 
            trans_y, rot, stretch_x, stretch_y, and sin_pert_coeffs keywords
            and instead use args.trans_std, args.rot_std_deg, and args.stretch_std
            to generate random perturbations.
        trans_x (float):
            Radians by which to translate along the az=0 direction 
            (tilts the beam).
        trans_y (float):
            Radians by which to translate along the az=pi/2 direction
            (tilts the beam).
        rot (float):
            How many radians by which to rotate the coordinate system for
            perturbed beams.
        stretch_x (float):
            Factor by which to stretch the az=0 direction.
        stretch_y (float):
            Factor by which to stretch the az=pi/2 direction.
        sin_pert_coeffs (array):
            The low order Fourier coefficients for the sidelobe perturbations.
    Returns:
        pow_sb (sparse_beam):
            A sparse_beam object whose interp method evaluates the perturbed beam
            at the chosen coordinates.
    """
    outfile = f"{output_dir}/perturbed_beam_beamvals_seed_{seed}.npy"
    load = os.path.exists(outfile)
    save = not load
    if seed is not None: # Ignore inputs and generate them randomly here
        rng = np.random.default_rng(seed=seed)
        trans_x, trans_y = rng.normal(args.trans_std)
        rot = np.deg2rad(rng.normal(args.rot_std_deg))
        stretch_x, stretch_y = rng.normal(loc=1, scale=args.stretch_std)
        sin_pert_coeffs = rng.normal(size=8)

    pow_sb = hydra.beam_sampler.get_pert_beam(
        args.beam_file, 
        outfile,
        mmax=args.mmax, 
        nmax=args.nmax,
        sqrt=args.per_ant, 
        Nfeeds=2, 
        num_modes_comp=args.Nbasis, 
        save=save, 
        cSL=args.csl,
        load=load,
        trans_x=trans_x,
        trans_y=trans_y,
        rot=rot,
        stretch_x=stretch_x,
        stretch_y=stretch_y,
        sin_pert_coeffs=sin_pert_coeffs
    )
                                              
    return pow_sb


def get_analytic_beam(args, beam_rng):
    """
    Make an analytic Airy or Gaussian beam based on args.
    
    Parameters:
        args (Namespace):
            Arguments that were parsed with argparse. Specifically makes use of
            beam_type.
        beam_rng (np.random.Generator)
            The random generator for beam perturbations.
    Returns:
        beam (AnalyticBeam):
            The desired AnalyticBeam instance
        beam_class (AnalyticBeam subclass):
            The particular class of beam.
    """
    diameter = 12. + beam_rng.normal(loc=0, scale=0.2)
    if args.beam_type == "gaussian":
        beam_class = GaussianBeam
    else:
        beam_class = AiryBeam
    beam = beam_class(diameter=diameter)

    return beam, beam_class


def get_parser(description):
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
                        
    return parser

def init_prebeam_simulation_items(args, output_dir, freqs):
    
    if args.chain_seed == "None":
        chain_seed = None
    else:
        chain_seed = int(args.chain_seed)

    # Timing file
    ftime = os.path.join(output_dir, "timing.dat")


    mmodes = np.arange(-args.mmax, args.mmax + 1)
    unpert_sb = hydra.sparse_beam.sparse_beam(args.beam_file, nmax=args.nmax, 
                                              mmodes=mmodes, Nfeeds=2, 
                                              alpha=args.rho_const,
                                              num_modes_comp=args.Nbasis,
                                              sqrt=args.per_ant,
                                              freq_range=(np.amin(freqs), np.amax(freqs)),
                                              save_fn=f"{output_dir}/unpert_sb")



                                              
    return chain_seed, ftime, unpert_sb

def get_src_params(args, output_dir):
    ra_low, ra_high = (min(args.ra_bounds), max(args.ra_bounds))
    dec_low, dec_high = (min(args.dec_bounds), max(args.dec_bounds))
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
    return ra, dec, beta_ptsrc, ptsrc_amps, fluxes

def get_obs_params(args):
    lst_min, lst_max = (min(args.lst_bounds), max(args.lst_bounds))
    times = np.linspace(lst_min, lst_max, args.Ntimes)
    freqs = np.arange(
        args.freq_low, 
        args.freq_low + args.Nfreqs * args.ch_wid, 
        args.ch_wid
    )
    return times, freqs

def get_array_params(args):
    hex_array = tuple(args.hex_array)
    assert len(args.hex_array) == 2, "hex-array argument must have length 2."

    # Convert to radians
    array_lat = np.deg2rad(args.array_lat)

    ant_pos = build_hex_array(hex_spec=hex_array, d=14.6)
    ants = np.array(list(ant_pos.keys()))
    Nants = len(ants)
    print("Nants =", Nants)

    return array_lat, ant_pos, Nants


def setup_args_dirs(parser):
    args = parser.parse_args()
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

    return args, output_dir

def vis_sim_wrapper(
        args, 
        output_dir, 
        array_lat, 
        ant_pos, 
        Nants, 
        times, 
        freqs, 
        ra, 
        dec, 
        beta_ptsrc, 
        ptsrc_amps, 
        fluxes, 
        beams, 
        ref_beam, 
        ftime
):
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

    np.save(os.path.join(output_dir, "data0"), data)
    if args.perts_only: # Subtract off the vis. made with ref beam
        ref_sim_outpath = os.path.join(output_dir, "model0_ref.npy")
        ref_beams = Nants * [ref_beam]
        ref_beam_vis = run_vis_sim(args, ftime, times, freqs, ant_pos, 
                                  Nants, ra, dec, fluxes, ref_beams, sim_outpath)
        data -= (ref_beam_vis + ref_beam_vis.swapaxes(-1, -2).conj())
        np.save(os.path.join(output_dir, "data_res"), data)

    inv_noise_var = 1/noise_var
    np.save(os.path.join(output_dir, "inv_noise_var.npy"), inv_noise_var)
    
    return flux_inference, unpert_vis, data, inv_noise_var

def prep_beam_Dmatr_items(args, output_dir, array_lat, times, ra, dec, unpert_sb):
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
    return bess_matr, trig_matr, nmodes, mmodes, per_source_Dmatr_out, za, az