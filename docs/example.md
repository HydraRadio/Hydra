# Example script

Hydra comes with an example script that can generate simulated data and then run an inference on it, all in one go. 
The script is `example.py` in the root directory of the Hydra repository.

For a basic example, simply run:

    python example.py --ptsrc

This will run the simulation and sampler for a basic default setup with a (3, 4, 3) hexagonal grid of 10 antennas, 
100 randomly-placed point sources, 30 time samples, 60 frequency channels, and 100 samples. The samples and 
simulated data will be output into the `./output/` directory.

To run as an MPI job, make sure that `mpi4py` is installed and functioning. You can then run the script like so:

    mpirun -n 16 python example.py --ptsrc

At the moment, the MPI-enabled linear system solver assumes that the matrix operator can be split into square 
blocks, and so will only run with a square number of processes, e.g. 4, 9, 16 and so on.

Run `python example.py --help` to see a full list of command line options.
