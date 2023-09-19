#!/usr/bin/env python

import numpy as np
from pyuvdata import UVData, UVBeam
import healpy as hp
import argparse, os, sys, time
import pyuvsim
from hera_sim.beams import PolyBeam

#sys.path.insert(0,'/home/phil/hera/Hydra/')
sys.path.insert(0,'/cosma/home/dp270/dc-bull2/software/Hydra/')
import hydra

# Set up argparser
description = "Precompute visibility response of an array to each spherical harmonic mode."
parser = argparse.ArgumentParser(description=description)
parser.add_argument("--template", type=str, action="store", 
                    required=True, dest="template",
                    help="Path to template UVData file.")
parser.add_argument("--lmax", type=int, action="store", default=4,
                    required=False, dest="lmax",
                    help="Set the random seed.")
parser.add_argument("--nside", type=int, action="store", default=32,
                    required=False, dest="nside",
                    help="Set the healpix resolution (nside).")
parser.add_argument("--outdir", type=str, action="store", 
                    required=True, dest="outdir",
                    help="Path to output directory.")
parser.add_argument("--freqidx", type=int, action="store", 
                    required=True, dest="freqidx",
                    help="Frequency index.")
parser.add_argument("--beam", type=str, action="store", 
                    required=True, dest="beam",
                    help="Beam type ('gaussian' or 'polybeam'), or path to UVBeam file.")
args = parser.parse_args()


# Set-up variables
lmax = args.lmax
nside = args.nside
freqidx = args.freqidx
outdir = args.outdir
template = args.template
beam_str = args.beam

# Check that output directory exists
if not os.path.exists(outdir):
    try:
        os.makedirs(outdir)
    except:
        pass
print("\nOutput directory:", outdir)

# Load template UVData object
print("Template file:", template)
uvd = UVData()
uvd.read_uvh5(template, read_data=False)
print("    Read uvh5 file metadata.")

# Check logfile
logfile = os.path.join(outdir, "logfile%04d" % freqidx)
with open(logfile, 'w+') as f:
    f.write("Starting precompute script\n")

# Get freqs, lsts, ants etc.
freqs = np.unique(uvd.freq_array)
lsts = np.unique(uvd.lst_array)
antpos, antnums = uvd.get_ENU_antpos(center=False, pick_data_ants=True)
ants = {}
for i in range(len(antnums)):
    ants[antnums[i]] = antpos[i]

# Get number of modes
_ell, _m = hp.Alm().getlm(lmax=lmax)
Nmodes = _ell.size

# Print basic info
print("lmax:        %d" % lmax)
print("modes:       %d" % Nmodes)
print("nside:       %d" % nside)
print("Frequencies: %5.1f -- %5.1f MHz (%d channels)" \
      % (freqs.min()/1e6, freqs.max()/1e6, freqs.size))
print("LSTs:        %5.4f -- %5.4f rad (%d times)" \
      % (lsts.min(), lsts.max(), lsts.size))
print("(Identical Gaussian beams)")
print("-"*50)
    
# Load beams and set interpolator properties
if beam_str.lower() == 'polybeam':
    print("PolyBeam")
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

elif beam_str.lower() == 'gaussian':
    beams = [pyuvsim.AnalyticBeam('gaussian', diameter=14.) 
             for ant in ants]
else:
    print("UVBeam:", beam_str)
    uvb = UVBeam.from_file(beam_str)
    uvb.interpolation_function = 'healpix_simple'
    uvb.freq_interp_kind = 'linear'
    beams = [uvb for i in range(len(antnums))]
        

# Output metadata
metafile = os.path.join(outdir, "response_sh_metadata_%04d" % freqidx)
print("Output file:", "response_sh_metadata")
with open(metafile, 'w') as f:
    f.write("template: %s\n" % template)
    f.write("lmax:     %d\n" % lmax)
    f.write("modes:    %d\n" % Nmodes)
    f.write("nside:    %d\n" % nside)
    f.write("freqs:    %s\n" % freqs[freqidx])
    f.write("lsts:     %s\n" % lsts)
    f.write("antnums:  %s\n" % antnums)
    f.write("antpos:   %s\n" % antpos)

# Run calculation on each worker
# (NFREQS, NTIMES, NANTS, NANTS, NMODES) if polarized=False
#v = np.zeros((max_block_size, lsts.size, len(ants), len(ants), Nmodes))

# Run simulation for single frequency
tstart = time.time()
ell, m, vis = hydra.vis_simulator.simulate_vis_per_alm(
                    lmax=lmax,
                    nside=nside,
                    ants=ants,
                    freqs=np.atleast_1d(freqs[freqidx]),
                    lsts=lsts,
                    beams=beams,
                    polarized=False,
                    precision=2,
                    latitude=np.deg2rad(-30.7215),
                    use_feed="x",
                    multiprocess=True,
                    amplitude=1.,
                    logfile=logfile
                )
# vis shape (NAXES, NFEED, NFREQS, NTIMES, NANTS, NANTS, NMODES)
# (NFREQS, NTIMES, NANTS, NANTS, NMODES) if pol False
print("(Freq. idx %04d) Run took %5.1f min" % (freqidx, (time.time() - tstart)/60.))

# Save operator to .npy file for each chunk
outfile = os.path.join(outdir, "response_sh_%04d" % freqidx)
np.save(outfile, vis)
print("Output file:", "response_sh_%04d" % freqidx)

# Output ell, m values
out_lm = os.path.join(outdir, "response_sh_ellm_%04d" % freqidx)
print("Output file:", "response_sh_ellm_%04d" % freqidx)
np.save(out_lm, np.column_stack((ell, m)))
print("Finished")

