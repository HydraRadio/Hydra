#!/usr/bin/env python
"""
Combine uvh5 files, e.g. containing different LSTs.
"""
import numpy as np
from pyuvdata import UVData
import argparse, os, sys, time, glob

# Set up argparser
description = "Combine uvh5 files, e.g. containing different LSTs."
parser = argparse.ArgumentParser(description=description)
parser.add_argument("template", type=str, action="store", 
                    help="Filename template, including wildcards.")
parser.add_argument("idx_min", type=int, action="store",
                    help="Index of lower bound of frequency range.")
parser.add_argument("idx_max", type=int, action="store",
                    help="Index of upper bound of frequency range.")
parser.add_argument("outname", type=str, action="store",
                    help="Output filename.")
args = parser.parse_args()

# Extract args
template = args.template
idx_min = args.idx_min
idx_max = args.idx_max
files = sorted( glob.glob(template) )
print("Template: %s" % template)
print("Found %d files matching template." % len(files))

# Frequency channel selection
new_chans = np.arange(idx_min, idx_max)

# Load the first file to get the UVData object to have the right shape
uvd = UVData().from_file(files[0]).select(freq_chans=new_chans)

# Loop over files, adding them together
for i in range(1, len(files)):
    print("File:", files[i])
    _uvd = UVData.from_file(files[i]).select(freq_chans=new_chans)
    uvd += _uvd

# Save file
uvd.write_uvh5(outname)
