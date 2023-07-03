#!/usr/bin/env python
"""
Down-select a data file by compressing by redundancy and selecting a particular 
band by frequency indices.
"""
import numpy as np
from pyuvdata import UVData
import argparse, os, sys

import argparse, os, sys, time
import pyuvsim

#sys.path.insert(0,'/home/phil/hera/Hydra/')
sys.path.insert(0,'/cosma/home/dp270/dc-bull2/software/Hydra/')
import hydra

# Set up argparser
description = "Down-select a data file by compressing by redundancy and selecting a particular band by frequency indices."
parser = argparse.ArgumentParser(description=description)
parser.add_argument("filename", type=str, action="store", 
                    help="Path to input UVData file.")
parser.add_argument("idx_min", type=int, action="store",
                    help="Index of lower bound of frequency range.")
parser.add_argument("idx_max", type=int, action="store",
                    help="Index of upper bound of frequency range.")
parser.add_argument("--compress", type=str, action="store", default="none", 
                    required=False, dest="compress",
                    help="Compress by redundancy. Options are 'none', 'select', or 'average'.")
args = parser.parse_args()

# Extract args
fname = arg.fname
idx_min = arg.idx_min
idx_max = arg.idx_max
compress = arg.compress
assert compress in ['none', 'select', 'average']

print("Input file:", fname)
print("Compress by redundancy:", compress)
print("Freq. indices between %4d -- %4d inclusive" % (idx_min, idx_max))

# Load full data file and select down by redundancy
uvd = UVData()
uvd.read_uvh5(fname)
if compress != 'none':
    uvd.compress_by_redundancy(method=compress, inplace=True)

# Frequency selection
freqs = np.unique(uvd.freq_array)
print("Freq. channels available: %d (%5.2f -- %5.2f MHz)" \
      % (freqs.size, freqs[0]/1e6, freqs[-1]/1e6))
new_chans = np.arange(idx_min, idx_max+1)
print("Selecting %d freq. channels" % new_chans.size)
print("Freq. range: %5.2f -- %5.2f MHz" \
      % (freqs[idx_min]/1e6, freqs[idx_max]/1e6))
uvd.select(freq_chans=new_chans)

outfile = "%s.subband_%d_%d.uvh5" % (fname[:-5], freqs[idx_min]/1e6, freqs[idx_max]/1e6)
print("Saving down-selected file as:", outfile)
uvd.write_uvh5(outfile)
