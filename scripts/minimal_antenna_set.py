#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
from pyuvdata import UVData
import argparse, os, sys

fname = str(sys.argv[1])

# Load data file (metadata only)
uvd = UVData()
uvd.read_uvh5(fname, read_data=False)

# Get all possible redundancies given antenna positions
blgroups, vecs, bllens = uvd.get_redundancies(use_antpos=False)

# Make list of antennas in pair and which red. group. they belong to
ant1, ant2, redgrp = [], [], []
for i, grp in enumerate(blgroups):
    for bl in grp:
        a1, a2 = uvd.baseline_to_antnums(bl)
        if a1 >= a2:
            ant1.append(a1)
            ant2.append(a2)
        else:
            ant1.append(a2)
            ant2.append(a1)
        redgrp.append(i)

ant1 = np.array(ant1)
ant2 = np.array(ant2)
redgrp = np.array(redgrp)

# Unique red. grps.
unique_redgrps = np.unique(redgrp)
unique_ants = np.unique(np.concatenate((ant1, ant2)))

# Antenna appearances (total no. redgrps)
tot = []
for ant in unique_ants:
    t1 = redgrp[ant1 == ant]
    t2 = redgrp[ant2 == ant]
    tot.append( np.unique(np.concatenate((t1, t2))).size )


def count_bls(myants):
    """
    Count how many baselines there are for this set of antennas.
    """
    Nbls = 0
    for i in range(ant1.size):
        if ant1[i] not in myants or ant2[i] not in myants:
            continue
        else:
            Nbls += 1
    return Nbls


def valid_set(myants, verbose=False):
    """
    Check whether all redgrps are covered by this set of antennas.
    """
    grps = []
    for a in myants:
        # Only work with valid pairs where both ants are in myants
        t1 = redgrp[np.logical_and(ant1 == a, np.isin(ant2, myants))]
        t2 = redgrp[np.logical_and(ant2 == a, np.isin(ant1, myants))]
        
        # Add unique groups for this antenna
        grps.append( np.unique(np.concatenate((t1, t2))) )
    
    # Concatenate and get unique groups again
    grps = np.unique( np.concatenate(grps) )
    if np.all(grps == unique_redgrps):
        if verbose:
            print(grps, unique_redgrps)
        return True
    else:
        return False


# Monte Carlo algorithm to look for small subsets
np.random.seed(30)
nthrows = 10
start = 15
best = []
best_count = count_bls(unique_ants)
for n in range(nthrows):
    print("Throw %d / %d" % (n+1, nthrows))
    antlist = unique_ants.copy()
    np.random.shuffle(antlist)
    
    # Loop over possible lengths
    for m in range(2, unique_ants.size):
        valid = valid_set(antlist[:m])
        if not valid:
            continue
        else:
            count = count_bls(antlist[:m])
            if count < best_count:
                print("    Improved to %d ants (%d bls)" % (m, count))
                best_count = count
                best = antlist[:m]

# Report best set of antennas
np.sort(best)
print("Run finished.")
print("Unique red. grps:", unique_redgrps.size, len(blgroups))
print("Best no. baselines:", best_count)
print("Best no. ants:", len(best))
print("Best:", best)
print("Valid:", valid_set(best, verbose=True))

# Load full data file and select down
uvd2 = UVData()
uvd2.read_uvh5(fname)
uvd2.select(antenna_nums=best)
print("New size:", uvd2.get_baseline_nums().size)
outfile = "%s.minimised.uvh5" % fname
print("Saving minimised file as:", outfile)
uvd2.write_uvh5(outfile)
