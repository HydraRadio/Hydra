#!/usr/bin/env python

import copy

default_params = {
    'seed':                     10,
    'calsrc-std':               0.01,
    'sim-gain-amp-std':         0.1,
    'sim-gain-sigma-frate':     0.3,
    'sim-gain-sigma-delay':     60,
    'sim-gain-frate0':          0.1,
    'sim-gain-delay0':          20,
    'gain-prior-amp':           0.15,
    'gain-prior-sigma-frate':   0.5,
    'gain-prior-sigma-delay':   100,
    'ptsrc-amp-prior-level':    0.1,
}


param_ranges = {
    'calsrc-std':               [0.001, 0.005, 0.01, 0.05, 0.1],
    'sim-gain-amp-std':         [0.01, 0.05, 0.1, 0.5],
    'sim-gain-sigma-frate':     [0.1, 0.3, 0.6, 1.2],
    'sim-gain-sigma-delay':     [30, 60, 90, 120],
    'gain-prior-amp':           [0.05, 0.15, 0.25, 0.35, 0.45],
    'gain-prior-sigma-frate':   [0.1, 0.3, 0.5, 0.7],
    'gain-prior-sigma-delay':   [20, 50, 100, 150, 200, 400],
    'ptsrc-amp-prior-level':    [0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
}


def fill_script(taskname, params):
    params['taskname'] = taskname
    script = \
"""#!/bin/bash -l

#SBATCH --ntasks 1
#SBATCH --cpus-per-task=128
#SBATCH -J hydra-{taskname}
#SBATCH -o stdout.%J.out
#SBATCH -e stderr.%J.err
#SBATCH -p cosma8-serial
#SBATCH -A dp270
#SBATCH -t 40:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=phil.bull@manchester.ac.uk

module purge

module load anaconda3/5.2.0
. /cosma/local/anaconda3/5.2.0/etc/profile.d/conda.sh
conda env list
conda activate /cosma/home/dp270/dc-bull2/.conda/envs/hera

echo $PATH

# Testing
/cosma/home/dp270/dc-bull2/.conda/envs/hera/bin/python -c "import numpy; print(numpy.__version__)"
/cosma/home/dp270/dc-bull2/.conda/envs/hera/bin/python -c "import pyuvsim; print(pyuvsim.__version__)"

# Set no. of cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPLBACKEND=Agg

# Run the program
cd /cosma/home/dp270/dc-bull2/software/Hydra
/cosma/home/dp270/dc-bull2/.conda/envs/hera/bin/python \\
       example.py --seed {seed} \\
                  --gains \\
                  --ptsrc \\
                  --stats \\
                  --timing \\
                  --diagnostics \\
                  --hex-array 6 9 \\
                  --Nptsrc 1000 \\
                  --Ntimes 40 \\
                  --Nfreqs 48 \\
                  --Niters 200 \\
                  --output-dir /cosma8/data/dp270/dc-bull2/hydra-{taskname} \\
                  --multiprocess \\
                  --calsrc-std {calsrc-std} \\
                  --dec-bounds -1.583 0.511 \\
                  --freq-bounds 100.0 120.0 \\
                  --lst-bounds 0.8 1.2 \\
                  --ptsrc-amp-prior-level {ptsrc-amp-prior-level} \\
                  --sim-gain-amp-std {sim-gain-amp-std} \\
                  --sim-gain-sigma-frate {sim-gain-sigma-frate} \\
                  --sim-gain-sigma-delay {sim-gain-sigma-delay} \\
                  --sim-gain-frate0 {sim-gain-frate0} \\
                  --sim-gain-delay0 {sim-gain-delay0} \\
                  --gain-prior-amp {gain-prior-amp} \\
                  --gain-prior-sigma-frate {gain-prior-sigma-frate} \\
                  --gain-prior-sigma-delay {gain-prior-sigma-delay}
"""
    script = script.format(**params)
    return script


n = 0
for pname in param_ranges.keys():
    print(pname)
    
    for j, val in enumerate(param_ranges[pname]):
        print("    %s-%d" % (pname, j))
        
        # Check if value is equal to fiducial; skip if so
        if val == default_params[pname]:
            print("    (Skipping fiducial case)")
            continue
        n += 1
        
        # Create updated script
        pp = copy.copy(default_params)
        pp[pname] = val
        txt = fill_script("%s-%d" % (pname, j), pp)
        
        with open("hydra-scan-%s-%d.sh" % (pname, j), 'w') as f:
            f.write(txt)
        
print(n)
