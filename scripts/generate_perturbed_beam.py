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

np.random.seed(args.seed)
trans_x, trans_y = np.random.normal(scale=args.trans_std, size=2)
rot = np.random.normal(scale=np.deg2rad(args.rot_std_deg))
stretch_x, stretch_y = np.random.normal(scale=args.stretch_std, size=2)
sin_pert_coeffs = np.random.normal(size=8)

mmodes = np.arange(-args.mmax, args.mmax + 1)
pow_sb = hydra.sparse_beam.sparse_beam(args.beam_file, args.nmax, mmodes, 
                                       Nfeeds=2, num_modes_comp=32, sqrt=True, 
                                       perturb=True, trans_x=trans_x, 
                                       trans_y=trans_y, rot=rot, 
                                       stretch_x=stretch_x, stretch_y=stretch_y, 
                                       sin_pert_coeffs=sin_pert_coeffs)

Azg, Zag = np.meshgrid(pow_sb.axis1_array, pow_sb.axis2_array)
pert_beam, _ = pow_sb.interp(az_array=Azg.flatten(), za_array=Zag.flatten())
fit_coeffs, _ = pow_sb.get_fits(data_array=pert_beam.reshape(pow_sb.data_array.shape))

np.save(fit_coeffs, 
        f"{args.outdir}/perturbed_beam_fit_coeffs_seed_{args.seed}.npy")



