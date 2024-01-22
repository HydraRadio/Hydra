
import argparse
import numpy as np

def get_config():
    """
    Parse commandline arguments to get configuration settings.
    """
    description = "Example Gibbs sampling of the joint posterior of several analysis " \
                  "parameters in 21-cm power spectrum estimation from a simulated " \
                  "visibility data set"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, action="store", default=0,
                        required=False, dest="seed",
                        help="Set the random seed.")

    # Samplers
    parser.add_argument("--gains", action="store_true",
                        required=False, dest="sample_gains",
                        help="Sample gains.")
    #parser.add_argument("--vis", action="store_true",
    #                    required=False, dest="sample_vis",
    #                    help="Sample visibilities in general.")
    parser.add_argument("--ptsrc", action="store_true",
                        required=False, dest="sample_ptsrc",
                        help="Sample point source amplitudes.")
    parser.add_argument("--beam", action="store_true",
                        required=False, dest="sample_beam",
                        help="Sample beams.")
    parser.add_argument("--sh", action="store_true",
                        required=False, dest="sample_sh",
                        help="Sample spherical harmonic modes.")
    parser.add_argument("--pspec", action="store_true",
                        required=False, dest="sample_pspec",
                        help="Sample 21cm power spectrum.")

    # Output options
    parser.add_argument("--stats", action="store_true",
                        required=False, dest="calculate_stats",
                        help="Calcultae statistics about the sampling results.")
    parser.add_argument("--diagnostics", action="store_true",
                        required=False, dest="output_diagnostics",
                        help="Output diagnostics.") # This will be ignored
    parser.add_argument("--timing", action="store_true", required=False,
                        dest="save_timing_info", help="Save timing info.")
    parser.add_argument("--plotting", action="store_true",
                        required=False, dest="plotting",
                        help="Output plots.")

    # Array and data shape options
    parser.add_argument('--hex-array', type=int, action="store", default=(3,4),
                        required=False, nargs='+', dest="hex_array",
                        help="Hex array layout, specified as the no. of antennas "
                             "in the 1st and middle rows, e.g. '--hex-array 3 4'.")
    parser.add_argument("--Nptsrc", type=int, action="store", default=100,
                        required=False, dest="Nptsrc",
                        help="Number of point sources to use in simulation (and model).")
    parser.add_argument("--Ntimes", type=int, action="store", default=30,
                        required=False, dest="Ntimes",
                        help="Number of times to use in the simulation.")
    parser.add_argument("--Nfreqs", type=int, action="store", default=60,
                        required=False, dest="Nfreqs",
                        help="Number of frequencies to use in the simulation.")
    parser.add_argument("--Niters", type=int, action="store", default=100,
                        required=False, dest="Niters",
                        help="Number of joint samples to gather.")

    # Noise level
    parser.add_argument("--sigma-noise", type=float, action="store",
                        default=0.05, required=False, dest="sigma_noise",
                        help="Standard deviation of the noise, in the same units "
                             "as the visibility data.")

    parser.add_argument("--solver", type=str, action="store",
                        default='cg', required=False, dest="solver_name",
                        help="Which linear solver to use ('cg' or 'gmres' or 'mpicg').")
    #parser.add_argument("--mpicg-split", type=int, action="store",
    #                    default=1, required=False, dest="mpicg_split",
    #                    help="If the MPI CG solver is being used, how many blocks to split the linear system into.")
    parser.add_argument("--output-dir", type=str, action="store",
                        default="./output", required=False, dest="output_dir",
                        help="Output directory.")
    #parser.add_argument("--multiprocess", action="store_true", dest="multiprocess",
    #                    required=False,
    #                    help="Whether to use multiprocessing in vis sim calls.")

    # Point source sim params
    parser.add_argument("--ra-bounds", type=float, action="store", default=(0, 1),
                        nargs=2, required=False, dest="ra_bounds",
                        help="Bounds for the Right Ascension of the randomly simulated sources")
    parser.add_argument("--dec-bounds", type=float, action="store", default=(-0.6, 0.4),
                        nargs=2, required=False, dest="dec_bounds",
                        help="Bounds for the Declination of the randomly simulated sources")
    parser.add_argument("--lst-bounds", type=float, action="store", default=(0.2, 0.5),
                        nargs=2, required=False, dest="lst_bounds",
                        help="Bounds for the LST range of the simulation, in radians.")
    parser.add_argument("--freq-bounds", type=float, action="store", default=(100., 120.),
                        nargs=2, required=False, dest="freq_bounds",
                        help="Bounds for the frequency range of the simulation, in MHz.")
    parser.add_argument("--ptsrc-amp-prior-level", type=float, action="store", default=0.1,
                        required=False, dest="ptsrc_amp_prior_level",
                        help="Fractional prior on point source amplitudes")
    #parser.add_argument("--vis-prior-level", type=float, action="store", default=0.1,
    #                    required=False, dest="vis_prior_level",
    #                    help="Prior on visibility values")

    parser.add_argument("--calsrc-std", type=float, action="store", default=-1.,
                        required=False, dest="calsrc_std",
                        help="Define a different std. dev. for the amplitude prior of a calibration source. If -1, do not use a calibration source.")
    parser.add_argument("--calsrc-radius", type=float, action="store", default=10.,
                        required=False, dest="calsrc_radius",
                        help="Radius around declination of the zenith in which to search for brightest source, which is then identified as the calibration source.")
                        
    # Gain prior
    parser.add_argument("--gain-prior-amp", type=float, action="store", default=0.1,
                        required=False, dest="gain_prior_amp",
                        help="Overall amplitude of gain prior.")
    parser.add_argument("--gain-nmax-freq", type=int, action="store", default=2,
                        required=False, dest="gain_nmaxfreq",
                        help="Max. Fourier mode index for gain perturbations (freq. direction).")
    parser.add_argument("--gain-nmax-time", type=int, action="store", default=2,
                        required=False, dest="gain_nmaxtime",
                        help="Max. Fourier mode index for gain perturbations (time direction).")
    parser.add_argument("--gain-only-positive-modes", type=bool, action="store", default=False,
                        required=False, dest="gain_only_positive_modes",
                        help="Whether to only permit positive wavenumber gain modes.")

    # parser.add_argument("--gain-prior-sigma-frate", type=float, action="store", default=None,
    #                     required=False, dest="gain_prior_sigma_frate",
    #                     help="Width of a Gaussian prior in fringe rate, in units of mHz.")
    # parser.add_argument("--gain-prior-sigma-delay", type=float, action="store", default=None,
    #                     required=False, dest="gain_prior_sigma_delay",
    #                     help="Width of a Gaussian prior in delay, in units of ns.")
    # parser.add_argument("--gain-prior-zeropoint-std", type=float, action="store", default=None,
    #                     required=False, dest="gain_prior_zeropoint_std",
    #                     help="If specified, fix the std. dev. of the (0,0) mode to some value.")
    # parser.add_argument("--gain-prior-frate0", type=float, action="store", default=0.,
    #                     required=False, dest="gain_prior_frate0",
    #                     help="The central fringe rate of the Gaussian taper (mHz).")
    # parser.add_argument("--gain-prior-delay0", type=float, action="store", default=0.,
    #                     required=False, dest="gain_prior_delay0",
    #                     help="The central delay of the Gaussian taper (ns).")
    # parser.add_argument("--gain-mode-cut-level", type=float, action="store", default=None,
    #                     required=False, dest="gain_mode_cut_level",
    #                     help="If specified, gain modes with (prior power spectrum) < (gain-mode-cut-level) * max(prior power spectrum) will be excluded from the linear solve (i.e. set to zero).")
    # parser.add_argument("--gain-always-linear", type=bool, action="store", default=False,
    #                     required=False, dest="gain_always_linear",
    #                     help="If True, the gain perturbations are always applied under the linear approximation (the x_i x_j^* term is neglected everywhere)")

    # Gain simulation
    parser.add_argument("--sim-gain-amp-std", type=float, action="store", default=0.05,
                        required=False, dest="sim_gain_amp_std",
                        help="Std. dev. of amplitude of simulated gain.")
    # parser.add_argument("--sim-gain-sigma-frate", type=float, action="store", default=None,
    #                     required=False, dest="sim_gain_sigma_frate",
    #                     help="Width of a Gaussian in fringe rate, in units of mHz.")
    # parser.add_argument("--sim-gain-sigma-delay", type=float, action="store", default=None,
    #                     required=False, dest="sim_gain_sigma_delay",
    #                     help="Width of a Gaussian in delay, in units of ns.")
    # parser.add_argument("--sim-gain-frate0", type=float, action="store", default=0.,
    #                     required=False, dest="sim_gain_frate0",
    #                     help="The central fringe rate of the Gaussian taper (mHz).")
    # parser.add_argument("--sim-gain-delay0", type=float, action="store", default=0.,
    #                     required=False, dest="sim_gain_delay0",
    #                     help="The central delay of the Gaussian taper (ns).")

    # Beam parameters
    parser.add_argument("--beam-sim-type", type=str, action="store", default="gaussian",
                        required=False, dest="beam_sim_type",
                        help="Which type of beam to use for the simulation. ['gaussian', 'polybeam']")
    parser.add_argument("--beam-prior-std", type=float, action="store", default=1,
                        required=False, dest="beam_prior_std",
                        help="Std. dev. of beam coefficient prior, in units of Zernike coefficient")
    parser.add_argument("--beam-nmax", type=int, action="store",
                        default=16, required=False, dest="beam_nmax",
                        help="Maximum radial degree of the Fourier-Bessel basis for the beams.")
    parser.add_argument("--beam-mmax", type=int, action="store",
                        default=0, required=False, dest="beam_mmax",
                        help="Maximum azimuthal degree of the Fourier-Bessel basis for the beams.")
    parser.add_argument("--rho-const", type=float, action="store", 
                        default=np.sqrt(1-np.cos(np.pi * 23 / 45)),
                        required=False, dest="rho_const",
                        help="A constant to define the radial projection for the beam spatial basis")

    # Spherical harmonic parameters
    parser.add_argument("--sim-sh-lmax", type=int, action="store", default=8,
                        required=False, dest="sim_sh_lmax",
                        help="Maximum ell value to include for spherical harmonic simulation.")
    parser.add_argument("--sim-sh-nside", type=int, action="store",
                        default=16, required=False, dest="sim_sh_nside",
                        help="Healpix nside used to construct simulated spherical harmonic response.")
    parser.add_argument("--sh-lmax", type=int, action="store", default=8,
                        required=False, dest="sh_lmax",
                        help="Maximum ell value to include for spherical harmonic sampler.")
    parser.add_argument("--sh-nside", type=int, action="store",
                        default=16, required=False, dest="sh_nside",
                        help="Healpix nside used to construct spherical harmonic response function.")


    args = parser.parse_args()
    return args