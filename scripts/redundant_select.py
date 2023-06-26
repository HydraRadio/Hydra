#!/usr/bin/env python

import numpy as np
from pyuvdata import UVData
import argparse, os, sys

fname = str(sys.argv[1])

# Load full data file and select down
uvd = UVData()
uvd.read_uvh5(fname)
uvd.compress_by_redundancy(method='select', inplace=True)

outfile = "%s.minimised.uvh5" % fname[:-5]
print("Saving minimised file as:", outfile)
uvd.write_uvh5(outfile)
