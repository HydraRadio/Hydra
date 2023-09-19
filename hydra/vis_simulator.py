"""
Trimmed-down version of vis_cpu that can be modified to return fragments of the
visibilities.
"""
import numpy as np
import warnings
from astropy.constants import c
from pyuvdata import UVBeam
from typing import Optional, Sequence
from vis_cpu import conversions
import healpy as hp
from multiprocessing import Pool, cpu_count
import os, warnings, datetime, time
from . import utils

def run_checks(
    antpos: np.ndarray,
    freq: float,
    eq2tops: np.ndarray,
    crd_eq: np.ndarray,
    I_sky: np.ndarray,
    beam_list: Sequence[UVBeam],
    precision: int = 1,
    polarized: bool = False,
    beam_idx: Optional[np.ndarray] = None,
):
    """
    Check that the inputs to vis_sim() are valid.
    """
    # Check precision
    assert precision in {1, 2}
    if precision == 1:
        real_dtype = np.float32
        complex_dtype = np.complex64
    else:
        real_dtype = np.float64
        complex_dtype = np.complex128

    # Specify number of polarizations (axes/feeds)
    if polarized:
        nax = nfeed = 2
    else:
        nax = nfeed = 1

    nant, ncrd = antpos.shape
    assert ncrd == 3, "antpos must have shape (NANTS, 3)."
    ntimes, ncrd1, ncrd2 = eq2tops.shape
    assert ncrd1 == 3 and ncrd2 == 3, "eq2tops must have shape (NTIMES, 3, 3)."
    ncrd, nsrcs = crd_eq.shape
    assert ncrd == 3, "crd_eq must have shape (3, NSRCS)."
    assert (
        I_sky.ndim == 1 and I_sky.shape[0] == nsrcs
    ), "I_sky must have shape (NSRCS,)."

    # Get the number of unique beams
    nbeam = len(beam_list)

    # Check the beam indices
    if beam_idx is None:
        if nbeam == 1:
            beam_idx = np.zeros(nant, dtype=int)
        elif nbeam == nant:
            beam_idx = np.arange(nant, dtype=int)
        else:
            raise ValueError(
                "If number of beams provided is not 1 or nant, beam_idx must be provided."
            )
    else:
        assert beam_idx.shape == (nant,), "beam_idx must be length nant"
        assert all(
            0 <= i < nbeam for i in beam_idx
        ), "beam_idx contains indices greater than the number of beams"

    if beam_list is not None:
        # make sure we interpolate to the right frequency first.
        beam_list = [
            bm.interp(freq_array=np.array([freq]), new_object=True, run_check=False)
            if isinstance(bm, UVBeam)
            else bm
            for bm in beam_list
        ]

    if polarized and any(b.beam_type != "efield" for b in beam_list):
        raise ValueError("beam type must be efield if using polarized=True")
    elif not polarized and any(
        (
            b.beam_type != "power"
            or getattr(b, "Npols", 1) > 1
            or b.polarization_array[0] not in [-5, -6]
        )
        for b in beam_list
    ):
        raise ValueError(
            "beam type must be power and have only one pol (either xx or yy) if polarized=False"
        )
    return beam_idx


def vis_sim_per_source(
    antpos: np.ndarray,
    freq: float,
    eq2tops: np.ndarray,
    crd_eq: np.ndarray,
    I_sky: np.ndarray,
    beam_list: Sequence[UVBeam],
    precision: int = 2,
    polarized: bool = False,
    beam_idx: Optional[np.ndarray] = None,
    subarr_ant=None
):
    """
    Calculate visibility from an input intensity map and beam model. This is
    a trimmed-down version of vis_cpu that only uses UVBeam beams (not gridded
    beams).

    Parameters:
        antpos (array_like):
            Antenna position array. Shape=(NANT, 3).
        freq (float):
            Frequency to evaluate the visibilities at [GHz].
        eq2tops (array_like):
            Set of 3x3 transformation matrices to rotate the RA and Dec
            cosines in an ECI coordinate system (see `crd_eq`) to
            topocentric ENU (East-North-Up) unit vectors at each
            time/LST/hour angle in the dataset.
            Shape=(NTIMES, 3, 3).
        crd_eq (array_like):
            Cartesian unit vectors of sources in an ECI (Earth Centered
            Inertial) system, which has the Earth's center of mass at
            the origin, and is fixed with respect to the distant stars.
            The components of the ECI vector for each source are:
            (cos(RA) cos(Dec), sin(RA) cos(Dec), sin(Dec)).
            Shape=(3, NSRCS).
        I_sky (array_like):
            Intensity distribution of sources/pixels on the sky, assuming
            intensity (Stokes I) only. The Stokes I intensity will be split
            equally between the two linear polarization channels, resulting in
            a factor of 0.5 from the value inputted here. This is done even if
            only one polarization channel is simulated. Shape=(NSRCS,).
        beam_list (list of UVBeam, optional):
            If specified, evaluate primary beam values directly using UVBeam
            objects. Note that if `polarized` is True, these beams must be
            efield beams, and conversely if `polarized` is False they
            must be power beams with a single polarization (either XX or YY).
        precision (int):
            Which precision level to use for floats and complex numbers.
            Allowed values:
            - 1: float32, complex64
            - 2: float64, complex128
        polarized (bool):
            Whether to simulate a full polarized response in terms of nn, ne,
            en, ee visibilities. See Eq. 6 of Kohn+ (arXiv:1802.04151) for
            notation.
        beam_idx (array_like):
            Optional length-NANT array specifying a beam index for each antenna.
            By default, either a single beam is assumed to apply to all
            antennas or each antenna gets its own beam.
        subarr_ant (int): Used to calculate only those visibilities associated
            with a particular antenna.

    Returns:
        vis (array_like):
            Simulated visibilities. If `polarized = True`, the output will have
            shape (NAXES, NFEED, NTIMES, NANTS, NANTS, NSRCS), otherwise it
            will have shape (NTIMES, NANTS, NANTS, NSRCS).
    """
    if precision == 1:
        real_dtype = np.float32
        complex_dtype = np.complex64
    else:
        real_dtype = np.float64
        complex_dtype = np.complex128

    # Specify number of polarizations (axes/feeds)
    if polarized:
        nax = nfeed = 2
    else:
        nax = nfeed = 1

    nant, ncrd = antpos.shape
    assert ncrd == 3, "antpos must have shape (NANTS, 3)."
    ntimes, ncrd1, ncrd2 = eq2tops.shape
    assert ncrd1 == 3 and ncrd2 == 3, "eq2tops must have shape (NTIMES, 3, 3)."
    ncrd, nsrcs = crd_eq.shape
    assert ncrd == 3, "crd_eq must have shape (3, NSRCS)."
    assert (
        I_sky.ndim == 1 and I_sky.shape[0] == nsrcs
    ), "I_sky must have shape (NSRCS,)."

    # Get the number of unique beams
    nbeam = len(beam_list)

    # Run checks and get standardised beam_idx array
    beam_idx = run_checks(
        antpos, freq, eq2tops, crd_eq, I_sky, beam_list, precision, polarized, beam_idx
    )

    # Intensity distribution (sqrt) and antenna positions. Does not support
    # negative sky. Factor of 0.5 accounts for splitting Stokes I between
    # polarization channels
    Isqrt = np.sqrt(0.5 * I_sky).astype(real_dtype)
    antpos = antpos.astype(real_dtype)

    ang_freq = 2.0 * np.pi * freq

    # Zero arrays: beam pattern, visibilities, delays, complex voltages
    if subarr_ant is None:
        vis_shape = (nfeed, nfeed, ntimes, nant, nant, nsrcs)
    else:
        vis_shape = (nfeed, nfeed, ntimes, nant, nsrcs)
    vis = np.zeros(vis_shape, dtype=complex_dtype)
    crd_eq = crd_eq.astype(real_dtype)

    # Loop over time samples
    im = 0
    for t, eq2top in enumerate(eq2tops.astype(real_dtype)):
        # Dot product converts ECI cosines (i.e. from RA and Dec) into ENU
        # (topocentric) cosines, with (tx, ty, tz) = (e, n, u) components
        # relative to the center of the array
        tx, ty, tz = crd_top = np.dot(eq2top, crd_eq)

        # Simulate even if sources are below the horizon, since we need a
        # visibility per source regardless
        above_horizon = tz > 0
        #tx = tx[above_horizon]
        #ty = ty[above_horizon]
        nsrcs_up = len(tx)

        A_s = np.zeros((nax, nfeed, nbeam, nsrcs_up), dtype=complex_dtype)
        tau = np.zeros((nant, nsrcs_up), dtype=real_dtype)
        v = np.zeros((nant, nsrcs_up), dtype=complex_dtype)

        # Primary beam pattern using direct interpolation of UVBeam object
        az, za = conversions.enu_to_az_za(enu_e=tx, enu_n=ty, orientation="uvbeam")
        for i, bm in enumerate(beam_list):
            spw_axis_present = utils.get_beam_interp_shape(bm)
            kw = (
                {"reuse_spline": True, "check_azza_domain": False}
                if isinstance(bm, UVBeam)
                else {}
            )

            interp_beam = bm.interp(
                az_array=az, za_array=za, freq_array=np.atleast_1d(freq), **kw
            )[0]

            if polarized:
                if spw_axis_present:
                    interp_beam = interp_beam[:, 0, :, 0, :]
                else:
                    interp_beam = interp_beam[:, :, 0, :]
            else:
                # Here we have already asserted that the beam is a power beam and
                # has only one polarization, so we just evaluate that one.
                if spw_axis_present:
                    interp_beam = np.sqrt(interp_beam[0, 0, 0, 0, :])
                else:
                    interp_beam = np.sqrt(interp_beam[0, 0, 0, :])

            A_s[:, :, i] = interp_beam

        # Check for invalid beam values
        if np.any(np.isinf(A_s)) or np.any(np.isnan(A_s)):
            raise ValueError("Beam interpolation resulted in an invalid value")

        # Calculate delays, where tau = (b * s) / c
        #np.dot(antpos, crd_top[:, above_horizon], out=tau)
        np.dot(antpos, crd_top[:,:], out=tau)
        tau /= c.value

        # Component of complex phase factor for one antenna
        # (actually, b = (antpos1 - antpos2) * crd_top / c; need dot product
        # below to build full phase factor for a given baseline)
        np.exp(1.0j * (ang_freq * tau), out=v)

        # Complex voltages.
        #v *= Isqrt[above_horizon]
        v *= Isqrt[:]
        v[:,~above_horizon] *= 0. # zero-out sources below the horizon

        # Compute visibilities using product of complex voltages (upper triangle).
        # Input arrays have shape (Nax, Nfeed, [Nants], Nsrcs
        v = A_s[:, :, beam_idx] * v[np.newaxis, np.newaxis, :]



        if subarr_ant is None:
            for i in range(len(antpos)):
                # We want to take an outer product over feeds/antennas, contract over
                # E-field components, and integrate over the sky.
                vis[:, :, t, i : i + 1, i:, :] = np.einsum(
                    "jiln,jkmn->iklmn",
                    v[:, :, i : i + 1, :].conj(),
                    v[:, :, i:, :],
                    optimize=True,
                )
        else:
            # Get the ones where the antenna in question is not conjugated
            vis[:, :, t] = np.einsum(
                "jiln,jkmn->ikln", # summing over m just sums over one antenna (squeezes an axis)
                v[:, :, :, :].conj(),
                v[:, :, subarr_ant: subarr_ant + 1, :],
                optimize=True,
            )

    # Return visibilities with or without multiple polarization channels
    return vis if polarized else vis[0, 0]


def simulate_vis_per_source(
    ants,
    fluxes,
    ra,
    dec,
    freqs,
    lsts,
    beams,
    polarized=False,
    precision=2,
    latitude=-30.7215 * np.pi / 180.0,
    use_feed="x",
    multiprocess=True,
    subarr_ant=None,
    mpi_comm=None
):
    """
    Run a basic simulation, returning the visibility for each source
    separately. Based on ``vis_cpu``.

    This wrapper handles the necessary coordinate conversions etc.

    Parameters:
        ants (dict):
            Dictionary of antenna positions. The keys are the antenna names
            (integers) and the values are the Cartesian x,y,z positions of the
            antennas (in meters) relative to the array center.
        fluxes (array_like):
            2D array with the flux of each source as a function of frequency,
            of shape (NSRCS, NFREQS).
        ra, dec (array_like):
            Arrays of source RA and Dec positions in radians. RA goes from
            [0, 2 pi] and Dec from [-pi, +pi].
        freqs (array_like):
            Frequency channels for the simulation, in Hz.
        lsts (array_like):
            Local sidereal times for the simulation, in radians. Range is
            [0, 2 pi].
        beams (list of ``UVBeam`` objects):
            Beam objects to use for each antenna.
        polarized (bool):
            If True, use polarized beams and calculate all available linearly-
            polarized visibilities, e.g. V_nn, V_ne, V_en, V_ee.
            Default: False (only uses the 'ee' polarization).
        precision (int):
            Which precision setting to use for :func:`~vis_cpu`. If set to
            ``1``, uses the (``np.float32``, ``np.complex64``) dtypes. If set
            to ``2``, uses the (``np.float64``, ``np.complex128``) dtypes.
        latitude (float):
            The latitude of the center of the array, in radians. The default is
            the HERA latitude = -30.7215 * pi / 180.
        multiprocess (bool): Whether to use multiprocessing to speed up the
            calculation.
        subarr_ant (int): Used to calculate only those visibilities associated
            with a particular antenna.
        mpi_comm (MPI.Comm):
            MPI comm object. If given, MPI will be used for parallelisation. 
            Note that this will set `multiprocess = False`.

    Returns:
        vis (array_like):
            Complex, shape (NAXES, NFEED, NFREQS, NTIMES, NANTS, NANTS, NSRCS)
            if ``polarized == True``, or (NFREQS, NTIMES, NANTS, NANTS, NSRCS)
            otherwise.
    """
    nsrcs = ra.size
    
    # Disable multiprocess if mpi_comm is specified
    if mpi_comm is not None:
        multiprocess = False
        myid = comm.Get_rank()
        nworkers = comm.Get_size()

    assert len(ants) == len(
        beams
    ), "The `beams` list must have as many entries as the ``ants`` dict."

    assert fluxes.shape == (
        ra.size,
        freqs.size,
    ), "The `fluxes` array must have shape (NSRCS, NFREQS)."

    # Check RA and Dec ranges
    if np.any(ra < 0.) or np.any(ra > 2.*np.pi):
        warnings.warn("One or more ra values is outside the allowed range (0, 2 pi)", RuntimeWarning)
    if np.any(dec < -np.pi) or np.any(dec > np.pi):
        warnings.warn("One or more dec values is outside the allowed range (-pi, +pi)", RuntimeWarning)

    # Determine precision
    if precision == 1:
        complex_dtype = np.complex64
    else:
        complex_dtype = np.complex128

    # Get polarization information from beams
    if polarized:
        try:
            naxes = beams[0].Naxes_vec
            nfeeds = beams[0].Nfeeds
        except AttributeError:
            # If Naxes_vec and Nfeeds properties aren't set, assume all pol.
            naxes = nfeeds = 2

    # Antenna x,y,z positions
    antpos = np.array([ants[k] for k in ants.keys()])
    nants = antpos.shape[0]

    # Source coordinate transform, from equatorial to Cartesian
    crd_eq = conversions.point_source_crd_eq(ra, dec)

    # Get coordinate transforms as a function of LST
    eq2tops = np.array([conversions.eci_to_enu_matrix(lst, latitude) for lst in lsts])

    beams = [
        conversions.prepare_beam(beam, polarized=polarized, use_feed=use_feed)
        for beam in beams
    ]

    # Initialise output array
    if polarized:
        vis_shape = (naxes, nfeeds, freqs.size, lsts.size, nants, nants, nsrcs)
    elif subarr_ant is not None:
    # When polarized beams implemented, need to have similar block in  polarized case above
        vis_shape = (freqs.size, lsts.size, nants, nsrcs)
    else:
        vis_shape = (freqs.size, lsts.size, nants, nants, nsrcs)
    vis = np.zeros(vis_shape, dtype=complex_dtype)

    # Parallel loop over frequencies that calls vis_cpu for UVBeam
    # The `global` declaration is needed so that multiprocessing can handle
    # the input args correctly
    if multiprocess:
        global _sim_fn_simulate_vis_per_source
        def _sim_fn_simulate_vis_per_source(i):
            return vis_sim_per_source(
                                  antpos,
                                  freqs[i],
                                  eq2tops,
                                  crd_eq,
                                  fluxes[:, i],
                                  beam_list=beams,
                                  precision=precision,
                                  polarized=polarized,
                                  subarr_ant=subarr_ant,
                                 )
    
        # Set up parallel loop
        try:
            Nthreads = int(os.environ['OMP_NUM_THREADS'])
        except:
            Nthreads = cpu_count()
        with Pool(Nthreads) as pool:
            vv = pool.map(_sim_fn_simulate_vis_per_source, range(freqs.size))
    else:
        vv = np.zeros_like(vis)
        for i in range(len(freqs)):
            vv[i] = vis_sim_per_source(
                                  antpos,
                                  freqs[i],
                                  eq2tops,
                                  crd_eq,
                                  fluxes[:, i],
                                  beam_list=beams,
                                  precision=precision,
                                  polarized=polarized,
                                  subarr_ant=subarr_ant,
                                 )

    # Assign returned values to array
    for i in range(freqs.size):
        if polarized:
            vis[:, :, i] = vv[i]  # v.shape: (nax, nfeed, ntimes, nant, nant, nsrcs)
        else:
            vis[i] = vv[i]  # v.shape: (ntimes, nant, nant, nsrcs) (unless subarr_ant is not None)

    return vis


def simulate_vis(
    *args, **kwargs
):
    """
    Run a basic simulation, based on ``vis_cpu``.

    This wrapper handles the necessary coordinate conversions etc.

    Parameters:
        ants (dict):
            Dictionary of antenna positions. The keys are the antenna names
            (integers) and the values are the Cartesian x,y,z positions of the
            antennas (in meters) relative to the array center.
        fluxes (array_like):
            2D array with the flux of each source as a function of frequency,
            of shape (NSRCS, NFREQS).
        ra, dec (array_like):
            Arrays of source RA and Dec positions in radians. RA goes from
            [0, 2 pi] and Dec from [-pi, +pi].
        freqs (array_like):
            Frequency channels for the simulation, in Hz.
        lsts (array_like):
            Local sidereal times for the simulation, in radians. Range is
            [0, 2 pi].
        beams (list of ``UVBeam`` objects):
            Beam objects to use for each antenna.
        polarized (bool):
            If True, use polarized beams and calculate all available linearly-
            polarized visibilities, e.g. V_nn, V_ne, V_en, V_ee.
            Default: False (only uses the 'ee' polarization).
        precision (int):
            Which precision setting to use for :func:`~vis_cpu`. If set to
            ``1``, uses the (``np.float32``, ``np.complex64``) dtypes. If set
            to ``2``, uses the (``np.float64``, ``np.complex128``) dtypes.
        latitude (float):
            The latitude of the center of the array, in radians. The default is
            the HERA latitude = -30.7215 * pi / 180.
        multiprocess (bool): Whether to use multiprocessing to speed up the
            calculation

    Returns:
        vis (array_like):
            Complex, shape (NAXES, NFEED, NFREQS, NTIMES, NANTS, NANTS)
            if ``polarized == True``, or (NFREQS, NTIMES, NANTS, NANTS)
            otherwise.
    """
    # Run simulation using the per-source simulation function
    vis = simulate_vis_per_source(*args, **kwargs)

    # Sum over sources and return
    return np.sum(vis, axis=-1)


def simulate_vis_per_alm(
    lmax,
    nside,
    ants,
    freqs,
    lsts,
    beams,
    polarized=False,
    precision=2,
    latitude=-30.7215 * np.pi / 180.0,
    use_feed="x",
    multiprocess=True,
    amplitude=1.,
    logfile=None
):
    """
    Run a basic simulation, returning the visibility for each spherical harmonic mode
    separately. Based on ``vis_cpu``.

    This wrapper handles the necessary coordinate conversions etc.

    NOTE: The spherical harmonic modes are defined in the equatorial (RA, Dec)
    coordinate system.

    Parameters:
        lmax (int):
            Maximum ell value to simulate.
        nside (int):
            Healpix map nside used to generate the simulations. Higher values
            will give more accurate results.
        ants (dict):
            Dictionary of antenna positions. The keys are the antenna names
            (integers) and the values are the Cartesian x,y,z positions of the
            antennas (in meters) relative to the array center.
        freqs (array_like):
            Frequency channels for the simulation, in Hz.
        lsts (array_like):
            Local sidereal times for the simulation, in radians. Range is
            [0, 2 pi].
        beams (list of ``UVBeam`` objects):
            Beam objects to use for each antenna.
        polarized (bool):
            If True, use polarized beams and calculate all available linearly-
            polarized visibilities, e.g. V_nn, V_ne, V_en, V_ee.
            Default: False (only uses the 'ee' polarization).
        precision (int):
            Which precision setting to use for :func:`~vis_cpu`. If set to
            ``1``, uses the (``np.float32``, ``np.complex64``) dtypes. If set
            to ``2``, uses the (``np.float64``, ``np.complex128``) dtypes.
        latitude (float):
            The latitude of the center of the array, in radians. The default is
            the HERA latitude = -30.7215 * pi / 180.
        multiprocess (bool): Whether to use multiprocessing to speed up the
            calculation.
        amplitude (float):
            The amplitude to use for the spherical harmonic modes when running
            the simulation.
        logfile (str):
            Path to log file.

    Returns:
        ell, m (array_like):
            Arrays of integer values of ell, m modes, with the same ordering as
            the modes in the last dimensions of `vis`. This uses the default
            healpy ordering and convention (+ve m modes only).

        vis (array_like):
            Complex, shape (NAXES, NFEED, NFREQS, NTIMES, NANTS, NANTS, NMODES)
            if ``polarized == True``, or (NFREQS, NTIMES, NANTS, NANTS, NMODES)
            otherwise. This is the visibility response of the interferometer
            to each spherical harmonic (ell, m) mode.

            The final dimension has size `NMODES == 2*ell.size`. The first half
            of this dimensions (i.e. of length `ell.size`) corresponds to real
            spherical harmonic coefficients, with the same ordering as the `ell`
            and `m` arrays. The second set is for imaginary SH coefficients, again
            with the same ordering.
    """
    # Make sure these are array_like
    freqs = np.atleast_1d(freqs)
    lsts = np.atleast_1d(lsts)

    # Array of ell, m values in healpy ordering
    ell, m = hp.Alm().getlm(lmax=lmax)

    # Get Healpix pixel coords
    npix = hp.nside2npix(nside)
    pix_area = 4.*np.pi / npix # steradians per pixel
    dec, ra = hp.pix2ang(nside=nside, ipix=np.arange(npix), lonlat=False)
    # RA must be in range [0, 2 pi] and Dec in range [-pi, +pi]
    dec = dec - 0.5*np.pi # shift Dec coords

    # Dummy fluxes (one everywhere)
    fluxes = np.ones((npix, freqs.size))

    # Run simulation using the per-source simulation function, to get
    # visibility contrib. from each pixel
    if logfile is not None:
        t0 = time.time()
        with open(logfile, 'a') as f:
            f.write("%s Starting simulate_vis_per_source\n" % (datetime.now()))

    vis_pix = simulate_vis_per_source(ants=ants,
                                      fluxes=fluxes,
                                      ra=ra,
                                      dec=dec,
                                      freqs=freqs,
                                      lsts=lsts,
                                      beams=beams,
                                      polarized=polarized,
                                      precision=precision,
                                      latitude=latitude,
                                      use_feed=use_feed,
                                      multiprocess=multiprocess)

    if logfile is not None:
        with open(logfile, 'a') as f:
            f.write("%s Finished simulate_vis_per_source in %5.2f sec\n" \
                     % (datetime.now(), time.time() - t0))

    # Empty array with the right shape (no. visibilities times no. l,m modes)
    shape = list(vis_pix.shape)
    shape[-1] = 2*ell.size # replace last dim. with Nmodes (real + imag.)
    vis = np.zeros(shape, dtype=np.complex128)

    # Loop over (ell, m) modes, weighting the precomputed visibility sim
    # by the value of each spherical harmonic mode in each pixel
    alm = np.zeros(ell.size, dtype=np.complex128)
    for n in range(ell.size):

        if logfile is not None:
            with open(logfile, 'a') as f:
                f.write("%s ell %d / %d\n" % (datetime.now(), n, ell.size))

        # Start with zero vector for all modes
        alm *= 0

        # Loop over real, imaginary values for this mode only
        for j, val in enumerate([1., 1.j]):

            # Make healpix map for this mode only
            alm[n] = val
            skymap = hp.alm2map(alm, nside=nside) * pix_area * amplitude
            # multiply by pixel area to get 'integrated' quantity
            # (results will be in Jy units)

            # Multiply visibility for each pixel by the pixel value for this mode
            if polarized:
                # vis_pix: (NAXES, NFEED, NFREQS, NTIMES, NANTS, NANTS, NSRCS)
                vis[:,:,:,:,:,:,n + j*ell.size] = np.sum(vis_pix * skymap, axis=-1)
                # Last dim. of vis is in blocks of real (first ell.size modes) and
                # imaginary (last ell.size modes)
            else:
                # vis_pix: (NFREQS, NTIMES, NANTS, NANTS, NSRCS)
                vis[:,:,:,:,n + j*ell.size] = np.sum(vis_pix * skymap, axis=-1)

    if logfile is not None:
        with open(logfile, 'a') as f:
            f.write("%s Finished all.\n" % (datetime.now()))

    return ell, m, vis


def vis_sim_per_source_new(
    antpos: np.ndarray,
    freq: float,
    lsts: np.ndarray,
    alt: np.ndarray,
    az: np.ndarray,
    I_sky: np.ndarray,  
    beam_list: Sequence[UVBeam],
    precision: int = 2,
    polarized: bool = False,
    subarr_ant=None
):
    """
    Calculate visibility from an input intensity map and beam model. This is
    a trimmed-down version of vis_cpu that only uses UVBeam beams (not gridded
    beams).

    Parameters:
        antpos (array_like):
            Antenna position array. Shape=(NANT, 3).
        freq (float):
            Frequency to evaluate the visibilities at [Hz].
        lsts (array_like):
            LSTs.
        alt, az (array_like):
            Alt and az of sources at each time.
        I_sky (array_like):
            Intensity distribution of sources/pixels on the sky, assuming
            intensity (Stokes I) only. The Stokes I intensity will be split
            equally between the two linear polarization channels, resulting in
            a factor of 0.5 from the value inputted here. This is done even if
            only one polarization channel is simulated. Shape=(NSRCS,).
        beam_list (list of UVBeam, optional):
            If specified, evaluate primary beam values directly using UVBeam
            objects. Note that if `polarized` is True, these beams must be
            efield beams, and conversely if `polarized` is False they
            must be power beams with a single polarization (either XX or YY).
        precision (int):
            Which precision level to use for floats and complex numbers.
            Allowed values:
            - 1: float32, complex64
            - 2: float64, complex128
        polarized (bool):
            Whether to simulate a full polarized response in terms of nn, ne,
            en, ee visibilities. See Eq. 6 of Kohn+ (arXiv:1802.04151) for
            notation.
        subarr_ant (int): Used to calculate only those visibilities associated
            with a particular antenna.

    Returns:
        vis (array_like):
            Simulated visibilities. If `polarized = True`, the output will have
            shape (NAXES, NFEED, NTIMES, NANTS, NANTS, NSRCS), otherwise it
            will have shape (NTIMES, NANTS, NANTS, NSRCS).
    """
    if precision == 1:
        real_dtype = np.float32
        complex_dtype = np.complex64
    else:
        real_dtype = np.float64
        complex_dtype = np.complex128

    # Specify number of polarizations (axes/feeds)
    if polarized:
        nax = nfeed = 2
    else:
        nax = nfeed = 1

    nant, ncrd = antpos.shape
    assert ncrd == 3, "antpos must have shape (NANTS, 3)."
    ntimes, nsrcs = alt.shape
    assert (
        I_sky.ndim == 1 and I_sky.shape[0] == nsrcs
    ), "I_sky must have shape (NSRCS,)."

    # Get the number of unique beams
    nbeam = len(beam_list)

    # Intensity distribution (sqrt) and antenna positions. Does not support
    # negative sky. Factor of 0.5 accounts for splitting Stokes I between
    # polarization channels
    Isqrt = np.sqrt(0.5 * I_sky).astype(real_dtype)
    antpos = antpos.astype(real_dtype)
    ang_freq = 2.0 * np.pi * freq

    # Zero arrays: beam pattern, visibilities, delays, complex voltages
    if subarr_ant is None:
        vis_shape = (nfeed, nfeed, ntimes, nant, nant, nsrcs)
    else:
        vis_shape = (nfeed, nfeed, ntimes, nant, nsrcs)
    vis = np.zeros(vis_shape, dtype=complex_dtype)

    # Loop over time samples
    im = 0
    for tidx in range(len(lsts)):

        # Simulate even if sources are below the horizon, since we need a
        # visibility per source regardless
        above_horizon = alt[tidx] > 0.
        nsrcs_up = nsrcs

        A_s = np.zeros((nax, nfeed, nbeam, nsrcs_up), dtype=complex_dtype)
        tau = np.zeros((nant, nsrcs_up), dtype=real_dtype)
        v = np.zeros((nant, nsrcs_up), dtype=complex_dtype)

        # Primary beam pattern using direct interpolation of UVBeam object
        za = 0.5*np.pi - alt
        for i, bm in enumerate(beam_list):
            spw_axis_present = utils.get_beam_interp_shape(bm)
            kw = (
                {"reuse_spline": True, "check_azza_domain": False}
                if isinstance(bm, UVBeam)
                else {}
            )

            interp_beam = bm.interp(
                az_array=az[tidx], za_array=za[tidx], freq_array=np.atleast_1d(freq), **kw
            )[0]

            if polarized:
                if spw_axis_present:
                    interp_beam = interp_beam[:, 0, :, 0, :]
                else:
                    interp_beam = interp_beam[:, :, 0, :]
            else:
                # Here we have already asserted that the beam is a power beam and
                # has only one polarization, so we just evaluate that one.
                if spw_axis_present:
                    interp_beam = np.sqrt(interp_beam[0, 0, 0, 0, :])
                else:
                    interp_beam = np.sqrt(interp_beam[0, 0, 0, :])

            A_s[:, :, i] = interp_beam

        # Check for invalid beam values
        if np.any(np.isinf(A_s)) or np.any(np.isnan(A_s)):
            raise ValueError("Beam interpolation resulted in an invalid value")

        # Calculate delays, where tau = (b * s) / c
        #enu_e, enu_n, enu_u = crd_top
        crd_top = np.array([np.sin(za[tidx])*np.cos(az[tidx]), 
                            np.sin(za[tidx])*np.sin(az[tidx]), 
                            np.cos(za[tidx])])
        np.dot(antpos, crd_top[:,:], out=tau)
        tau /= c.value

        # Component of complex phase factor for one antenna
        # (actually, b = (antpos1 - antpos2) * crd_top / c; need dot product
        # below to build full phase factor for a given baseline)
        np.exp(1.0j * (ang_freq * tau), out=v)

        # Complex voltages.
        #v *= Isqrt[above_horizon]
        v *= Isqrt[:]
        v[:,~above_horizon] *= 0. # zero-out sources below the horizon

        # Compute visibilities using product of complex voltages (upper triangle).
        # Input arrays have shape (Nax, Nfeed, [Nants], Nsrcs
        v = A_s[:, :, :] * v[np.newaxis, np.newaxis, :]

        #print(">>>", A_s)
        #print("\n\n\n")
        #print("***", v)

        if subarr_ant is None:
            for i in range(len(antpos)):
                # We want to take an outer product over feeds/antennas, contract over
                # E-field components, and integrate over the sky.
                vis[:, :, tidx, i : i + 1, i:, :] = np.einsum(
                    "jiln,jkmn->iklmn",
                    v[:, :, i : i + 1, :].conj(),
                    v[:, :, i:, :],
                    optimize=True,
                )
        else:
            # Get the ones where the antenna in question is not conjugated
            vis[:, :, tidx] = np.einsum(
                "jiln,jkmn->ikln", # summing over m just sums over one antenna (squeezes an axis)
                v[:, :, :, :].conj(),
                v[:, :, subarr_ant: subarr_ant + 1, :],
                optimize=True,
            )

    # Return visibilities with or without multiple polarization channels
    return vis if polarized else vis[0, 0]