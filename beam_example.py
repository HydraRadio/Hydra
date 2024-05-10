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

    description = "Example Gibbs sampling of the joint posterior of beam "  \
                  "parameters from a simulated visibility data set " 
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, action="store", default=0,
                        required=False, dest="seed",
                        help="Set the random seed.")
    
    # Samplers
    parser.add_argument("--recalc-sc-op", action="store_true", required=False,
                        dest="recalc_sc_op", 
                        help="Recalculate a large operator in between iterations" \
                            "despite that it is constant; useful for profiling")
    
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
                        default=10., required=False, dest="sigma_noise",
                        help="Integration time, in seconds")
    parser.add_argument("--ch_wid", type=float, action="store",
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
    parser.add_argument("--Nbasis", type=str, action="store",
                        required=True,
                        help="Number of basis functions to use for beam estimation.")
    parser.add_argument("--nmax", type=int, action="store", default=80,
                        required=False, help="Maximum radial mode for beam modeling.")
    parser.add_argument("--mmax", type=int, action="store", default=45, 
                        required=False, help="Maxmimum azimuthal mode for beam modeling.")
    parser.add_argument("--rho-const", type=float, action="store", 
                        default=np.sqrt(1-np.cos(np.pi * 23 / 45)),
                        required=False, dest="rho_const",
                        help="A constant to define the radial projection for the beam spatial basis")
    
    args = parser.parse_args()

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

    mmodes = np.arange(-args.mmax, args.mmax + 1)
    input_beam = hydra.sparse_beam.sparse_beam(args.beam_file, args.nmax, 
                                               mmodes, alpha=args.rho_const)
    
    
