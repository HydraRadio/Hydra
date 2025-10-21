import hydra
import numpy as np
import argparse
    
parser = argparse.ArgumentParser()
parser.add_argument("--seed", required=True, type=int,
                    help="Random seed for perturbation")
parser.add_argument("--beam-file", required=True, type=str, dest="beam_file",
                    help="Path to beam file.")
parser.add_argument("--outdir", required=True, type=str, 
                    help="Output directory for the fits.")
parser.add_argument("--trans-std", required=False, type=float,
                    default=1e-2, dest="trans_std", 
                    help="Standard deviation for random tilt of beam")
parser.add_argument("--rot-std-deg", required=False, type=float, 
                    dest="rot_std_deg", default=1., 
                    help="Standard deviation for random beam rotation, in degrees.")
parser.add_argument("--stretch-std", required=False, type=float, 
                    dest="stretch_std", default=1e-2, 
                    help="Standard deviation for random beam stretching.")
parser.add_argument("--nmax", required=False, type=int, default=80,
                    help="The maximum radial mode number to use in the FB basis.")
parser.add_argument("--mmax", required=False, type=int, default=45,
                    help="The maximum azimuthal mode number to use.")
args = parser.parse_args()
_ = hydra.per_ant_beam_sampler.get_pert_beam(args.seed, args.beam_file, 
                                     trans_std=args.trans_std, 
                                     rot_std_deg=args.rot_std_deg,
                                     stretch_std=args.stretch_std, 
                                     mmax=args.mmax, nmax=args.nmax, sqrt=True, 
                                     Nfeeds=2, num_modes_comp=32, save=True,
                                     outdir=args.outdir)



