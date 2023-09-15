#!/usr/bin/env python

from mpi4py import MPI

import numpy as np
import hydra
import pyuvsim

import astropy
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.coordinates.builtin_frames import AltAz
from astropy.time import Time
import astropy.units as u

import time, os, resource
from hydra.utils import flatten_vector, reconstruct_vector, timing_info, \
                            build_hex_array, get_flux_from_ptsrc_amp, \
                            convert_to_tops, gain_prior_pspec_sqrt


import argparse



def load_ptsrc_catalogue_random(Nptsrc, freqs, ra_low, ra_high, dec_low, dec_high):
    # Generate random point source locations
    # RA goes from [0, 2 pi] and Dec from [-pi / 2, +pi / 2].
    ra = np.random.uniform(low=ra_low, high=ra_high, size=Nptsrc)
    
    # inversion sample to get them uniform on the sphere, in case wide bounds are used
    U = np.random.uniform(low=0, high=1, size=Nptsrc)
    dsin = np.sin(dec_high) - np.sin(dec_low)
    dec = np.arcsin(U * dsin + np.sin(dec_low)) # np.arcsin returns on [-pi / 2, +pi / 2]

    # Generate fluxes
    beta_ptsrc = -2.7
    ptsrc_amps = 10.**np.random.uniform(low=-1., high=2., size=Nptsrc)
    fluxes = get_flux_from_ptsrc_amp(ptsrc_amps, freqs, beta_ptsrc)
    return ra, dec, beta_ptsrc, fluxes


def equatorial_to_altaz(ra, dec, lsts, obstime_ref, location, unit="rad", frame="icrs"):
    """Convert RA and Dec coordinates into alt/az.

    Parameters
    ----------
    ra, dec : array_like
        Input RA and Dec positions. The units and reference frame of these
        positions can be set using the ``unit`` and ``frame`` kwargs.
    
    lsts : array_like
        Local sidereal time, in radians.

    obstime_ref : astropy.Time object
        ``Time`` object specifying the reference time.

    location : astropy.EarthLocation
        ``EarthLocation`` object specifying the location of the reference
        observation.

    unit : str, optional
        Which units the input RA and Dec values are in, using names intelligible
        to ``astropy.SkyCoord``. Default: 'rad'.

    frame : str, optional
        Which frame that input RA and Dec positions are specified in. Any
        system recognized by ``astropy.SkyCoord`` can be used. Default: 'icrs'.

    Returns
    -------
    alt, az : array_like
        Arrays of RA and Dec coordinates with respect to the ECI system used
        by vis_cpu.
    """
    if not isinstance(obstime_ref, Time):
        raise TypeError("obstime must be an astropy.Time object")
    if not isinstance(location, EarthLocation):
        raise TypeError("location must be an astropy.EarthLocation object")

    # Local sidereal time at this obstime and location
    lst_ref = obstime_ref.sidereal_time("apparent", longitude=location.lon).rad
    times = obstime_ref + (lsts - lst_ref)*24./(2.*np.pi) * u.hour

    # Create Astropy SkyCoord object
    skycoords = SkyCoord(ra, dec, unit=unit, frame=frame)

    # Get AltAz and ENU coords of sources at reference time and location. Ref:
    # https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
    alt_az = [skycoords.transform_to(AltAz(obstime=t, location=location)) for t in times]
    alt = np.array([aa.alt.rad for aa in alt_az])
    az = np.array([aa.az.rad for aa in alt_az])
    return alt, az


def precompute_proj_operators(comm, components):
    """
    
    """
    # Loop over components
    for i, comp in enumerate(components):

        # Get component details
        cname, proj_fn, proj_kwargs = comp
        print("Computing proj. operator for '%s' (%d / %d)" % (cname, i+1, len(components)))

        # 


def precompute_linear_operator(comm):
    """
    Precompute the blocks of the linear operator matrix.
    """
    pass


if __name__ == '__main__':

    #--------------------------------------------------------------------------
    # Parse arguments
    #--------------------------------------------------------------------------

    description = "Example Gibbs sampling of the joint posterior of several analysis " \
                  "parameters in 21-cm power spectrum estimation from a simulated " \
                  "visibility data set"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, action="store", default=0,
                        required=False, dest="seed",
                        help="Set the random seed.")
    
    
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
    
    # Ptsrc sim parameters
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

    # Noise level
    parser.add_argument("--sigma-noise", type=float, action="store",
                        default=0.05, required=False, dest="sigma_noise",
                        help="Standard deviation of the noise, in the same units "
                             "as the visibility data.")
    args = parser.parse_args()

    #--------------------------------------------------------------------------
    # MPI setup
    #--------------------------------------------------------------------------

    # Setup MPI
    comm = MPI.COMM_WORLD
    nworkers = comm.Get_size()
    myid = comm.Get_rank()

    #--------------------------------------------------------------------------
    # Settings
    #--------------------------------------------------------------------------

    # Simulation settings
    Nptsrc = args.Nptsrc
    Ntimes = args.Ntimes
    Nfreqs = args.Nfreqs
    Niters = args.Niters
    hex_array = tuple(args.hex_array)
    assert len(hex_array) == 2, "hex-array argument must have length 2."
    if myid == 0:
        print("Nptsrc:   ", Nptsrc)
        print("Ntimes:   ", Ntimes)
        print("Nfreqs:   ", Nfreqs)
        print("Niters:   ", Niters)
        print("hex_array:", hex_array)
    comm.barrier()

    # Noise specification
    sigma_noise = args.sigma_noise

    # Source position and LST/frequency ranges
    ra_low, ra_high = (min(args.ra_bounds), max(args.ra_bounds))
    dec_low, dec_high = (min(args.dec_bounds), max(args.dec_bounds))
    lst_min, lst_max = (min(args.lst_bounds), max(args.lst_bounds))
    freq_min, freq_max = (min(args.freq_bounds), max(args.freq_bounds))
    
    # Array latitude
    hera_latitude = np.deg2rad(-30.7215)

    # Random seed
    np.random.seed(args.seed)
    if myid == 0:
        print("    Seed:    %d" % args.seed)

    #--------------------------------------------------------------------------
    # Set up data dimensions
    #--------------------------------------------------------------------------

    # Time and frequency arrays
    lsts = np.linspace(lst_min, lst_max, Ntimes)
    freqs = np.linspace(freq_min, freq_max, Nfreqs)

    # Antenna positions and pairs
    ant_pos = build_hex_array(hex_spec=hex_array, d=14.6)
    ants = np.array(list(ant_pos.keys()))
    Nants = len(ants)
    if myid == 0:
        print("Nants =", Nants)

    antpairs = []
    for i in range(len(ants)):
        for j in range(i, len(ants)):
            if i != j:
                # Exclude autos
                antpairs.append((i,j))
    ants1, ants2 = list(zip(*antpairs))


    tstart = time.time()



    # Get point sources (random for now)
    ra, dec, beta_ptsrc, ptsrc_fluxes = load_ptsrc_catalogue_random(Nptsrc, freqs, ra_low, ra_high, dec_low, dec_high)

    # Assign blocks of sources to workers (this is the biggest dimension)
    ptsrc_idxs = np.array_split(np.arange(ra.size), nworkers)[myid]

    #----------------
    # HERA location
    location = EarthLocation.from_geodetic(lat=np.rad2deg(hera_latitude),
                                           lon=21.4283,
                                           height=1073.)
    # Observation time
    obstime_ref = Time('2023-09-30T01:00:00.00', format='isot', scale='utc')
    #----------------

    # Beams
    beams = [pyuvsim.analyticbeam.AnalyticBeam('gaussian', diameter=14.)
             for ant in ants]
    beams = [
        hydra.conversions.prepare_beam(beam, polarized=False, use_feed='x')
        for beam in beams ]
    

    # Get source alt/az for each LST
    # alt, az have shape (Ntimes, Nptsrc)
    t0 = time.time()
    alt, az = equatorial_to_altaz(ra[ptsrc_idxs], 
                                  dec[ptsrc_idxs], 
                                  lsts, 
                                  obstime_ref, 
                                  location, 
                                  unit="rad", 
                                  frame="icrs")
    #print("Worker %03d, coordinate precomp. took %5.3f sec" % (myid, time.time() - t0))
    

    #--------------------------------------------------------------------------
    #  Calculate visibilities for each source assigned to each worker
    #--------------------------------------------------------------------------
    antpos = np.array([ant_pos[k] for k in ant_pos.keys()])
    ants_list = list(ants)
    t0 = time.time()
    vv = []
    for j, freq in enumerate(freqs):
        _vv = hydra.vis_simulator.vis_sim_per_source_new(
                                        antpos=antpos,
                                        freq=freq * 1e6,
                                        lsts=lsts,
                                        alt=alt,
                                        az=az,
                                        I_sky=ptsrc_fluxes[ptsrc_idxs, j], 
                                        beam_list=beams,
                                        precision=2,
                                        polarized=False
                                        ) # (Ntimes, Nants, Nants, Nsrcs)
        
        # Allocate computed visibilities to only available baselines (saves memory)
        _vis = []
        for i, bl in enumerate(antpairs):
            idx1 = ants_list.index(bl[0])
            idx2 = ants_list.index(bl[1])
            _vis.append(_vv[:, idx1, idx2, :])
        vv.append(np.array(_vis))

    vv = np.array(vv)
    #print("Worker %03d, vis. sim. took %5.3f sec" % (myid, time.time() - t0))
    #print("Worker %03d" % myid, np.sum(np.abs(vv)), vv.shape)

    """
    # Send all the data to the root worker
    if myid == 0:
        vis = np.zeros((Nfreqs, len(antpairs), Ntimes, Nptsrc), dtype=vv.dtype)

        # Receive operator for each point source
        for i in range(vis.shape[-1]):
            if i in ptsrc_idxs:
                # Local to root worker
                print("SHAPE1a:", vv[:,:,:,i].shape, vis[:,:,:,i].shape)
                vis[:,:,:,i] = vv[:,:,:,i].copy()
            else:
                buf = np.zeros_like(vis[:,:,:,i])
                print("SHAPE1b:", buf.shape, vis[:,:,:,i].shape)
                comm.Irecv(buf, tag=i)
                vis[:,:,:,i] = buf

        print("Receive complete:", vis.shape, np.sum(np.abs(vis)))
    else:
        # Send operator for each point source
        for i, pidx in enumerate(ptsrc_idxs):
            print("SHAPE2:", vv[:,:,:,i].shape)
            comm.Isend(vv[:,:,:,i].copy(), dest=0, tag=pidx) # copy needed to make it contiguous
        print("Send complete")

    """

    # Send all the data to the root worker
    if myid == 0:
        vis = np.zeros((Nfreqs, Ntimes, len(antpairs), Nptsrc), dtype=vv.dtype)

    # Gather all
    vvall = comm.gather(vv, root=0) # gather on to root worker only; ordered by myid
    if myid == 0:
        vis = np.concatenate(vvall, axis=-1)
        print("Total time: %5.3f sec" % (time.time() - tstart))
    
    """
    # Define list of components for joint GCR system
    components = [
        ('ptsrc',   ptsrc_op,    ptsrc_kwargs),
        ('diffuse', diffuse_op,  diffuse_kwargs),
        ('eor',     eor_op,      eor_kwargs),
    ]
    """
    # Get freqs and LSTs for this worker

    comm.Barrier()
    #print(myid, "shutting down")
    #comm.Disconnect()