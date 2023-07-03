#!/usr/bin/env python
"""
Down-select a data file by compressing by redundancy and selecting a particular 
band by frequency indices.
"""
import numpy as np
from pyuvdata import UVData
import argparse, os, sys

print("Down-select a data file by compressing by redundancy and selecting \n"
      "a particular band by frequency indices.")

fname = str(sys.argv[1])
idx_min = float(sys.argv[2])
idx_max = float(sys.argv[3])

print("Input file:", fname)
print("Freq. indices between %4d -- %4d inclusive" % (idx_min, idx_max))

# Load full data file and select down by redundancy
uvd = UVData()
uvd.read_uvh5(fname)
uvd.compress_by_redundancy(method='select', inplace=True)

# Frequency selection
freqs = np.unique(uvd.freq_array)
new_chans = np.arange(idx_min, idx_max+1)
print("Selecting %d freq. channels" % new_chans.size)
print("Freq. range: %5.2f -- %5.2f MHz" % (freqs[idx_min], freqs[idx_max]))
uvd.select(freq_chans=new_chans)

outfile = "%s.subband.uvh5" % fname[:-5]
print("Saving down-selected file as:", outfile)
uvd.write_uvh5(outfile)
