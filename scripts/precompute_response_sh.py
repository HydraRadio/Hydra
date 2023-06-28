#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
from pyuvdata import UVData
import healpy as hp
import argparse, os, sys
import pyuvsim

sys.path.insert(0,'/home/phil/hera/Hydra/')
#sys.path.insert(0,'/cosma/home/dp270/dc-bull2/software/Hydra/')
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

args = parser.parse_args()

# Configure mpi
comm = MPI.COMM_WORLD
myid = comm.Get_rank()
nworkers = comm.Get_size()

# Set-up variables
lmax = args.lmax
nside = args.nside
outdir = args.outdir
template = args.template

# Load template UVData object
if myid == 0:
    print("Template file:", template)
uvd = UVData()
uvd.read_uvh5(template, read_data=False)
if myid == 0:
    print("    Read uvh5 file metadata.")

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
if myid == 0:
    print("lmax:        %d" % lmax)
    print("modes:       %d" % Nmodes)
    print("nside:       %d" % nside)
    print("Frequencies: %5.1f -- %5.1f MHz (%d channels)" \
          % (freqs.min()/1e6, freqs.max()/1e6, freqs.size))
    print("LSTs:        %5.4f -- %5.4f rad (%d times)" \
          % (lsts.min(), lsts.max(), lsts.size))
    print("(Identical Gaussian beams)")
    print("-"*50)
    
# Split idxs into ordered blocks per worker
idxs = np.arange(freqs.size)
blocks = np.array_split(idxs, nworkers)
max_block_size = np.max([b.size for b in blocks])

# Simple Gaussian beams for now
beams = [pyuvsim.AnalyticBeam('gaussian', diameter=14.) 
         for i in range(len(antnums))]

# Output metadata
if myid == 0:
    metafile = os.path.join(outdir, "response_sh_metadata")
    print("Output file:", "response_sh_metadata")
    with open(metafile, 'w') as f:
        f.write("template: %s\n" % template)
        f.write("lmax:     %d\n" % lmax)
        f.write("modes:    %d\n" % Nmodes)
        f.write("nside:    %d\n" % nside)
        f.write("freqs:    %s\n" % freqs)
        f.write("lsts:     %s\n" % lsts)
        f.write("blocks:   %s\n" % blocks)
        f.write("antnums:  %s\n" % antnums)
        f.write("antpos:   %s\n" % antpos)

# Run calculation on each worker
# (NFREQS, NTIMES, NANTS, NANTS, NMODES) if polarized=False
#v = np.zeros((max_block_size, lsts.size, len(ants), len(ants), Nmodes))

# Loop over blocks, one block per worker
# Run simulation for each block of frequencies
tstart = time.time()
ell, m, vis = hydra.vis_simulator.simulate_vis_per_alm(
                    lmax=lmax,
                    nside=nside,
                    ants=ants,
                    freqs=freqs[blocks[myid]],
                    lsts=lsts,
                    beams=beams,
                    polarized=False,
                    precision=2,
                    latitude=np.deg2rad(-30.7215),
                    use_feed="x",
                    multiprocess=False,
                    amplitude=1.
                )
# vis shape (NAXES, NFEED, NFREQS, NTIMES, NANTS, NANTS, NMODES)
# (NFREQS, NTIMES, NANTS, NANTS, NMODES) if pol False
print("(Worker %03d) Run took %5.1f min" % (myid, (time.time() - tstart)/60.))

# Save operator to .npy file for each chunk
outfile = os.path.join(outdir, "response_sh_%04d" % myid)
np.save(outfile, vis)
print("Output file:", "response_sh_%04d" % myid)

# Output ell, m values
if myid == 0:
    out_lm = os.path.join(outdir, "response_sh_ellm")
    print("Output file:", "response_sh_ellm")
    np.save(out_lm, np.column_stack((ell, m)))
    
comm.Barrier()
sys.exit(0)

"""
# Allocate receive buffer on root worker and gather values
# (NOTE: I think we need the blocks on each worker to be the same size to avoid 
# weird overlaps happening when we do Gather)
allv = None
if myid == 0:
    allv = np.zeros([nworkers, 
                     max_block_size, 
                     v.shape[-1]], 
                     dtype=float)
comm.Gather(v, allv, root=0)
if myid != 0:
    del v # free some memory

# Concatenate into a single array on root worker with the right shape
if myid == 0:
    allv_flat = np.concatenate([allv[i,:len(blocks[i])] for i in range(len(blocks))])
    del allv # save some memory again
    print(allv_flat)
"""
