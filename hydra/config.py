
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
    parser.add_argument("--cosmo", action="store_true",
                        required=False, dest="sample_cosmo_field",
                        help="Sample cosmo field.")
    #parser.add_argument("--vis", action="store_true",
    #                    required=False, dest="sample_vis",
    #                    help="Sample visibilities in general.")
    parser.add_argument("--ptsrc", action="store_true",
                        required=False, dest="sample_ptsrc",
                        help="Sample point source amplitudes.")
    parser.add_argument("--regions", action="store_true",
                        required=False, dest="sample_regions",
                        help="Sample amplitudes of regions of a diffuse map.")
    parser.add_argument("--beam", action="store_true",
                        required=False, dest="sample_beam",
                        help="Sample beams.")
    parser.add_argument("--sh", action="store_true",
                        required=False, dest="sample_sh",
                        help="Sample spherical harmonic modes.")
    parser.add_argument("--cl", action="store_true",
                        required=False, dest="sample_sh_pspec",
                        help="Sample spherical harmonic angular power spectrum.")
    parser.add_argument("--pspec", action="store_true",
                        required=False, dest="sample_pspec",
                        help="Sample 21cm power spectrum.")

    # Debug mode
    parser.add_argument("--debug", action="store_true",
                        required=False, dest="debug",
                        help="Whether to enable debug mode, which shows extra output.")

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
    parser.add_argument("--latitude", type=float, action="store", default=-30.7215,
                        required=False, dest="latitude",
                        help="Latitude of the array, in degrees.")
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
    

    # Point source sampler parameters
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
    
    # Diffuse model simulation parameters
    parser.add_argument("--sim-diffuse-sky-model", type=str, action="store",
                        default="none", required=False, dest="sim_diffuse_sky_model",
                        help="Which global sky model to use to define diffuse model. "
                             "The options all come from pyGDSM: 'gsm2008', 'gsm2016', 'haslam', "
                             "'lfss', or 'none'.")
    parser.add_argument("--sim-diffuse-nside", type=int, action="store",
                        default=16, required=False, dest="sim_diffuse_nside",
                        help="Healpix nside to use for the region maps.")


    # Cosmo field parameters
    parser.add_argument("--cosmo-ra-bounds", type=float, action="store", default=(0., 60.),
                        nargs=2, required=False, dest="cosmo_field_ra_bounds",
                        help="Bounds for the RA of the cosmo field sample points (in degrees).")
    parser.add_argument("--cosmo-dec-bounds", type=float, action="store", default=(-40., -20.),
                        nargs=2, required=False, dest="cosmo_field_dec_bounds",
                        help="Bounds for the Dec of the cosmo field sample points (in degrees).")
    parser.add_argument("--cosmo-ra-ngrid", type=int, action="store", default=10,
                        required=False, dest="cosmo_field_ra_ngrid",
                        help="Number of cosmo field sample points in the RA direction.")
    parser.add_argument("--cosmo-dec-ngrid", type=int, action="store", default=10,
                        required=False, dest="cosmo_field_dec_ngrid",
                        help="Number of cosmo field sample points in the Dec direction.")

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
    parser.add_argument("--gain-prior-zero-mode-std", type=float, action="store", default=None,
                        required=False, dest="gain_prior_zero_mode_std",
                        help="Separately specify the gain prior standard deviaiton for the zero mode.")
    parser.add_argument("--gain-only-positive-modes", type=bool, action="store", default=True,
                        required=False, dest="gain_only_positive_modes",
                        help="Whether to only permit positive wavenumber gain modes.")

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
    parser.add_argument("--sh-prior-std", type=float, action="store",
                        default=0.1, required=False, dest="sh_prior_std",
                        help="Prior standard deviation for spherical harmonic modes.")
    parser.add_argument("--sh-ref-freq", type=float, action="store",
                        default=100., required=False, dest="sh_ref_freq",
                        help="Reference frequency for the SH spectral dependence, in MHz.")
    parser.add_argument("--sh-spectral-idx", type=float, action="store",
                        default=0., required=False, dest="sh_spectral_idx",
                        help="Spectral index for the SH power law spectral dependence.")

    # Region parameters
    parser.add_argument("--region-nregions", type=int, action="store", default=10,
                        required=False, dest="region_nregions",
                        help="No. of regions to break up a diffuse map into.")
    parser.add_argument("--region-smoothing-fwhm", type=float, action="store",
                        default=None, required=False, dest="region_smoothing_fwhm",
                        help="Smoothing FWHM to apply to segmented diffuse map, to smooth "
                        "sharp edges.")
    parser.add_argument("--region-sky-model", type=str, action="store",
                        default="gsm2016", required=False, dest="region_sky_model",
                        help="Which global sky model to use to define diffuse regions. "
                             "The options all come from pyGDSM: 'gsm2008', 'gsm2016', 'haslam', "
                             "'lfss'.")
    parser.add_argument("--region-nside", type=int, action="store",
                        default=16, required=False, dest="region_nside",
                        help="Healpix nside to use for the region maps.")
    parser.add_argument("--region-amp-prior-level", type=float, action="store", default=0.1,
                        required=False, dest="region_amp_prior_level",
                        help="Fractional prior on diffuse region amplitudes")

    args = parser.parse_args()
    return args