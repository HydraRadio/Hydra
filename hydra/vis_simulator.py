"""
Trimmed-down version of vis_cpu that can be modified to return fragments of the 
visibilities.
"""

import numpy as np
import warnings
from astropy.constants import c
from pyuvdata import UVBeam
from scipy.interpolate import RectBivariateSpline
from typing import Optional, Sequence


def enu_to_az_za(enu_e, enu_n, orientation="astropy", periodic_azimuth=True):
    """Convert angle cosines in ENU coordinates into azimuth and zenith angle.

    For a pointing vector in East-North-Up (ENU) coordinates vec{p}, the input
    arguments are ``enu_e = vec{p}.hat{e}`` and ``enu_n = vec{p}.hat{n}`, where
    ``hat{e}`` is a unit vector in ENU coordinates etc.

    For a drift-scan telescope pointing at the zenith, the ``hat{e}`` direction
    is aligned with the ``U`` direction (in the UVW plane), which means that we
    can identify the direction cosines ``l = enu_e`` and ``m = enu_n``.

    Azimuth is oriented East of North, i.e. Az(N) = 0 deg, Az(E) = +90 deg in
    the astropy convention, and North of East, i.e. Az(N) = +90 deg, and
    Az(E) = 0 deg in the UVBeam convention.

    Parameters
    ----------
    enu_e, enu_n : array_like
        Normalized angle cosine coordinates on the interval (-1, +1).

    orientation : str, optional
        Orientation convention used for the azimuth angle. The default is
        ``'astropy'``, which uses an East of North convention (Az(N) = 0 deg,
        Az(E) = +90 deg). Alternatively, the ``'uvbeam'`` convention uses
        North of East (Az(N) = +90 deg, Az(E) = 0 deg).

    periodic_azimuth : bool, optional
        if True, constrain az to be betwee 0 and 2 * pi
        This avoids the issue that arctan2 outputs angles between -pi and pi
        while most CST beam formats store azimuths between 0 and 2pi which leads
        interpolation domain mismatches.

    Returns
    -------
    az, za : array_like
        Corresponding azimuth and zenith angles (in radians).
    """
    assert orientation in [
        "astropy",
        "uvbeam",
    ], "orientation must be either 'astropy' or 'uvbeam'"

    lsqr = enu_n**2.0 + enu_e**2.0
    mask = lsqr < 1
    zeta = np.zeros_like(lsqr)
    zeta[mask] = np.sqrt(1 - lsqr[mask])

    az = np.arctan2(enu_e, enu_n)
    za = 0.5 * np.pi - np.arcsin(zeta)

    # Flip and rotate azimuth coordinate if uvbeam convention is used
    if orientation == "uvbeam":
        az = 0.5 * np.pi - az
    if periodic_azimuth:
        az = np.mod(az, 2 * np.pi)
    return az, za


def run_checks(
    antpos: np.ndarray,
    freq: float,
    eq2tops: np.ndarray,
    crd_eq: np.ndarray,
    I_sky: np.ndarray,
    beam_list: Sequence[UVBeam]
    precision: int = 1,
    polarized: bool = False,
    beam_idx: Optional[np.ndarray] = None
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


def vis_sim(
    antpos: np.ndarray,
    freq: float,
    eq2tops: np.ndarray,
    crd_eq: np.ndarray,
    I_sky: np.ndarray,
    beam_list: Sequence[UVBeam]
    precision: int = 1,
    polarized: bool = False,
    beam_idx: Optional[np.ndarray] = None,
):
    """
    Calculate visibility from an input intensity map and beam model. This is 
    a trimmed-down version of vis_cpu that only uses UVBeam beams (not gridded 
    beams).

    Parameters
    ----------
    antpos : array_like
        Antenna position array. Shape=(NANT, 3).
    freq : float
        Frequency to evaluate the visibilities at [GHz].
    eq2tops : array_like
        Set of 3x3 transformation matrices to rotate the RA and Dec
        cosines in an ECI coordinate system (see `crd_eq`) to
        topocentric ENU (East-North-Up) unit vectors at each
        time/LST/hour angle in the dataset.
        Shape=(NTIMES, 3, 3).
    crd_eq : array_like
        Cartesian unit vectors of sources in an ECI (Earth Centered
        Inertial) system, which has the Earth's center of mass at
        the origin, and is fixed with respect to the distant stars.
        The components of the ECI vector for each source are:
        (cos(RA) cos(Dec), sin(RA) cos(Dec), sin(Dec)).
        Shape=(3, NSRCS).
    I_sky : array_like
        Intensity distribution of sources/pixels on the sky, assuming intensity
        (Stokes I) only. The Stokes I intensity will be split equally between
        the two linear polarization channels, resulting in a factor of 0.5 from
        the value inputted here. This is done even if only one polarization
        channel is simulated.
        Shape=(NSRCS,).
    beam_list : list of UVBeam, optional
        If specified, evaluate primary beam values directly using UVBeam
        objects instead of using pixelized beam maps. Only one of ``bm_cube`` and
        ``beam_list`` should be provided.Note that if `polarized` is True,
        these beams must be efield beams, and conversely if `polarized` is False they
        must be power beams with a single polarization (either XX or YY).
    precision : int, optional
        Which precision level to use for floats and complex numbers.
        Allowed values:
        - 1: float32, complex64
        - 2: float64, complex128
    polarized : bool, optional
        Whether to simulate a full polarized response in terms of nn, ne, en,
        ee visibilities. See Eq. 6 of Kohn+ (arXiv:1802.04151) for notation.
        Default: False.
    beam_idx
        Optional length-NANT array specifying a beam index for each antenna.
        By default, either a single beam is assumed to apply to all antennas or
        each antenna gets its own beam.

    Returns
    -------
    vis : array_like
        Simulated visibilities. If `polarized = True`, the output will have
        shape (NAXES, NFEED, NTIMES, NANTS, NANTS), otherwise it will have
        shape (NTIMES, NANTS, NANTS).
    """
    # Run checks and get standardised beam_idx array
    beam_idx = run_checks(antpos, freq, eq2tops, crd_eq, I_sky, beam_list, 
                          precision, polarized, beam_idx)

    # Intensity distribution (sqrt) and antenna positions. Does not support
    # negative sky. Factor of 0.5 accounts for splitting Stokes I between
    # polarization channels
    Isqrt = np.sqrt(0.5 * I_sky).astype(real_dtype)
    antpos = antpos.astype(real_dtype)

    ang_freq = 2.0 * np.pi * freq

    # Zero arrays: beam pattern, visibilities, delays, complex voltages
    vis = np.zeros((nfeed, nfeed, ntimes, nant, nant), dtype=complex_dtype)
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
        az, za = enu_to_az_za(enu_e=tx, enu_n=ty, orientation="uvbeam")
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
            vis[:, :, t, i : i + 1, i:] = np.einsum(
                "jiln,jkmn->iklm", v[:, :, i : i + 1].conj(), v[:, :, i:], optimize=True
            )

    # Return visibilities with or without multiple polarization channels
    return vis if polarized else vis[0, 0]

