
import numpy as np

from scipy.signal import blackmanharris
import pyuvsim
from hera_sim.beams import PolyBeam
import time, os, resource
from .vis_simulator import simulate_vis
from .utils import flatten_vector, reconstruct_vector, timing_info, \
                   build_hex_array, get_flux_from_ptsrc_amp, \
                   convert_to_tops, gain_prior_pspec_sqrt, extract_vis_from_sim



def generate_random_ptsrc_catalogue(Nptsrc, ra_bounds, dec_bounds, logflux_bounds=(-1., 2.)):
    """
    xx
    """
    # Get coordinate bounds
    ra_low, ra_high = (min(ra_bounds), max(ra_bounds))
    dec_low, dec_high = (min(dec_bounds), max(dec_bounds))
    logflux_low, logflux_high = (min(logflux_bounds), max(logflux_bounds))
    
    # Generate random point source locations
    # RA goes from [0, 2 pi] and Dec from [-pi / 2, +pi / 2].
    ra = np.random.uniform(low=ra_low, high=ra_high, size=Nptsrc)
    
    # Inversion sample to get them uniform on the sphere, in case wide bounds are used
    U = np.random.uniform(low=0, high=1, size=Nptsrc)
    dsin = np.sin(dec_high) - np.sin(dec_low)
    dec = np.arcsin(U * dsin + np.sin(dec_low)) # np.arcsin returns on [-pi / 2, +pi / 2]

    # Generate fluxes
    ptsrc_amps = 10.**np.random.uniform(low=logflux_low, high=logflux_high, size=Nptsrc)
    return ra, dec, ptsrc_amps


def run_example_simulation(args, times, freqs, output_dir, ra, dec, ptsrc_amps, 
                           array_latitude, verbose=False):
    """
    Run an example visibility simulation for testing purposes.
    """
    # Dimensions of simulation
    hex_array = tuple(args.hex_array)
    assert len(hex_array) == 2, "hex-array argument must have length 2."

    # Set up array and data properties
    ant_pos = build_hex_array(hex_spec=args.hex_array, d=14.6)
    ants = np.array(list(ant_pos.keys()))
    Nants = len(ants)

    # Set up baselines
    antpairs = []
    for i in range(len(ants)):
        for j in range(i, len(ants)):
            if i != j:
                # Exclude autos
                antpairs.append((i,j))

    ants1, ants2 = list(zip(*antpairs))

    beta_ptsrc = -2.7
    fluxes = get_flux_from_ptsrc_amp(ptsrc_amps, freqs, beta_ptsrc)
    ##np.save(os.path.join(output_dir, "ptsrc_amps0"), ptsrc_amps)
    ##np.save(os.path.join(output_dir, "ptsrc_coords0"), np.column_stack((ra, dec)).T)

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
    _sim_vis = simulate_vis(
            ants=ant_pos,
            fluxes=fluxes,
            ra=ra,
            dec=dec,
            freqs=freqs*1e6, # MHz -> Hz
            lsts=times,
            beams=beams,
            polarized=False,
            precision=2,
            latitude=array_latitude,
            use_feed="x"
        )
    #timing_info(ftime, 0, "(0) Simulation", time.time() - t0)
    #print("(0) Simulation", time.time() - t0)

    # Allocate computed visibilities to only the requested baselines (saves memory)
    model0 = extract_vis_from_sim(ants, antpairs, _sim_vis)
    del _sim_vis # save some memory

    # Return
    ant_info = (ants, ant_pos, antpairs, ants1, ants2)
    return model0, fluxes, beams, ant_info

