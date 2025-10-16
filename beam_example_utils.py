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