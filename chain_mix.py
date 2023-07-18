import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
import warnings
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--chdirs", nargs="*", type=str, help="The directories where the chains are",
                    required=True)
parser.add_argument("--burn-in", nargs=1, type=int, help="Number of samples to discard", required=False,
                    default=5000)
parser.add_argument("--step", nargs=1, type=str, help="Which Gibbs step (used as a file name prefix)",
                    required=True)
args = parser.parse_args()

num_chains = len(args.chdirs) 
chains = []
for chdir in args.chdirs:
    fns = glob.glob(f"{chdir}/{args.step}*.npy")
    if len(fns) < burn_in:
        raise ValueError("burn-in period set greater than number of samples")
    elif len(fns) < 2 * burn_in:
        warnings.warn("Throwing away more than half the samples. Consider burn-in length")
    params = []
    Niters = len(fns)
    for fn in fns[burn_in:]:
        params.append(np.load(fn))
    # Split in half to assess within-chain mixing
    chains.extend(params[:Niters // 2], params[Niters//2:])

nchain_use = 2 * num_chains
niter_use = Niters // 2

# Shape (Niters, Nchains, Nants, Ncoeffs, Nfreqs)
chains = np.array(chains).swapaxes(0,1)


chain_means = np.mean(chains, axis=0) # Average over iterations
full_means = np.mean(chain_means, axis=0) # Further average over chains

btw_var = niter_use / (nchain_use - 1) * np.sum((chain_means - full_means)**2, axis=0)

sj = np.sum((chains - chain_means)**2, axis=0)

win_var = np.mean(sj, axis=0)

# Special weighted statistic from Gelman
post_var_wt = win_var * (niter_use - 1) / niter_use + btw_var / niter_use
Rstat = np.sqrt(post_var_wt/win_var) # Should be close to 1 if well-mixed

sys.stdout.write(f"Estimated mixing stat of {Rstat}")
