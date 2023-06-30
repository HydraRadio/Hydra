#!/usr/bin/env python
import numpy as np
from pyuvdata import UVData
import argparse, sys

# Set up argparser
description = "Select only short baselines."
parser = argparse.ArgumentParser(description=description)
parser.add_argument("--filename", type=str, action="store", 
                    required=True, dest="fname",
                    help="Path to input UVData file.")
parser.add_argument("--max-bl-length", type=float, action="store", default=75.,
                    required=False, dest="max_bl_length",
                    help="Max. baseline length, in m.")
args = parser.parse_args()

# Get input arguments
fname = args.fname
max_bl_length = args.max_bl_length
print("Input file:", fname)
print("Max. bl. length (m):", max_bl_length)

# Load data file
uvd = UVData()
uvd.read_uvh5(fname)

# Get baselines and redundancies (assumes data has been squished so only 
# one baseline per red. grp.)
ap, vec, bllens = uvd.get_redundancies()
bls = np.unique(uvd.baseline_array)
print("Total baselines:", len(bls))
print("Total data ants:", uvd.get_ants().size)

# Find short baselines
short_bls = []
for i in range(len(ap)):
    if bllens[i] <= max_bl_length:
        short_bls.append(ap[i][0])
print("Short baselines:", len(short_bls))
print(short_bls)

# Select only short bls
uvd.select(bls=short_bls)
print("Remaining ants:", uvd.get_ants().size)
ap, vec, bllens = uvd.get_redundancies()
print("Remaining baselines:", bllens.size)

# Output modified file
outfile = "%s.shortbls.uvh5" % fname[:-5]
print("Output file:", outfile)
uvd.write_uvh5(outfile)
