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
    precision: int = 1,
    polarized: bool = False,
    beam_idx: Optional[np.ndarray] = None,
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
    vis = np.zeros((nfeed, nfeed, ntimes, nant, nant, nsrcs), dtype=complex_dtype)
    crd_eq = crd_eq.astype(real_dtype)

    # Loop over time samples
    im = 0
    for t, eq2top in enumerate(eq2tops.astype(real_dtype)):
        # Dot product converts ECI cosines (i.e. from RA and Dec) into ENU
        # (topocentric) cosines, with (tx, ty, tz) = (e, n, u) components
        # relative to the center of the array
        tx, ty, tz = crd_top = np.dot(eq2top, crd_eq)
        above_horizon = tz > 0
        tx = tx[above_horizon]
        ty = ty[above_horizon]
        nsrcs_up = len(tx)

        A_s = np.zeros((nax, nfeed, nbeam, nsrcs_up), dtype=complex_dtype)
        tau = np.zeros((nant, nsrcs_up), dtype=real_dtype)
        v = np.zeros((nant, nsrcs_up), dtype=complex_dtype)

        # Primary beam pattern using direct interpolation of UVBeam object
        az, za = conversions.enu_to_az_za(enu_e=tx, enu_n=ty, orientation="uvbeam")
        for i, bm in enumerate(beam_list):
            kw = (
                {"reuse_spline": True, "check_azza_domain": False}
                if isinstance(bm, UVBeam)
                else {}
            )

            interp_beam = bm.interp(
                az_array=az, za_array=za, freq_array=np.atleast_1d(freq), **kw
            )[0]

            if polarized:
                interp_beam = interp_beam[:, 0, :, 0, :]
            else:
                # Here we have already asserted that the beam is a power beam and
                # has only one polarization, so we just evaluate that one.
                interp_beam = np.sqrt(interp_beam[0, 0, 0, 0, :])

            A_s[:, :, i] = interp_beam

        # Check for invalid beam values
        if np.any(np.isinf(A_s)) or np.any(np.isnan(A_s)):
            raise ValueError("Beam interpolation resulted in an invalid value")

        # Calculate delays, where tau = (b * s) / c
        np.dot(antpos, crd_top[:, above_horizon], out=tau)
        tau /= c.value

        # Component of complex phase factor for one antenna
        # (actually, b = (antpos1 - antpos2) * crd_top / c; need dot product
        # below to build full phase factor for a given baseline)
        np.exp(1.0j * (ang_freq * tau), out=v)

        # Complex voltages.
        v *= Isqrt[above_horizon]

        # Compute visibilities using product of complex voltages (upper triangle).
        # Input arrays have shape (Nax, Nfeed, [Nants], Nsrcs
        v = A_s[:, :, beam_idx] * v[np.newaxis, np.newaxis, :]

        for i in range(len(antpos)):
            # We want to take an outer product over feeds/antennas, contract over
            # E-field components, and integrate over the sky.
            vis[:, :, t, i : i + 1, i:, :] = np.einsum(
                "jiln,jkmn->iklmn",
                v[:, :, i : i + 1, :].conj(),
                v[:, :, i:, :],
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
    precision=1,
    latitude=-30.7215 * np.pi / 180.0,
    use_feed="x",
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

    Returns:
        vis (array_like):
            Complex, shape (NAXES, NFEED, NFREQS, NTIMES, NANTS, NANTS, NSRCS)
            if ``polarized == True``, or (NFREQS, NTIMES, NANTS, NANTS, NSRCS)
            otherwise.
    """
    nsrcs = ra.size

    assert len(ants) == len(
        beams
    ), "The `beams` list must have as many entries as the ``ants`` dict."

    assert fluxes.shape == (
        ra.size,
        freqs.size,
    ), "The `fluxes` array must have shape (NSRCS, NFREQS)."

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
        vis = np.zeros(
            (naxes, nfeeds, freqs.size, lsts.size, nants, nants, nsrcs),
            dtype=complex_dtype,
        )
    else:
        vis = np.zeros(
            (freqs.size, lsts.size, nants, nants, nsrcs), dtype=complex_dtype
        )

    # Loop over frequencies and call vis_cpu for UVBeam
    for i in range(freqs.size):

        v = vis_sim_per_source(
            antpos,
            freqs[i],
            eq2tops,
            crd_eq,
            fluxes[:, i],
            beam_list=beams,
            precision=precision,
            polarized=polarized,
        )
        if polarized:
            vis[:, :, i] = v  # v.shape: (nax, nfeed, ntimes, nant, nant, nsrcs)
        else:
            vis[i] = v  # v.shape: (ntimes, nant, nant, nsrcs)

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
