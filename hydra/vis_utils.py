
import numpy as np
from vis_cpu import conversions


def flatten_vector(v):
    """
    Flatten a complex vector with shape (N, Ntimes, Nfreq) into a block
    vector of shape (N x Ntimes x Nfreqs x 2), i.e. the real and imaginary
    blocks stuck together.
    """
    return np.concatenate((v.real.flatten(), v.imag.flatten()))


def reconstruct_vector(v, shape):
    """
    Undo the flattening of a complex vector.
    """
    y = v[: v.size // 2] + 1.0j * v[v.size // 2 :]
    return y.reshape(shape)


def apply_gains(v, gains, ants, antpairs, inline=False):
    """
    Apply gain factors to an input array of complex visibility values.

    Parameters:
        v (array_like):
            Input visibility array to which gains will be applied. This must
            have shape (Nbaselines, Nfreqs, Ntimes). The ordering of the 0th
            (baseline) dimension is assumed to be the same ordering as
            `antpairs`.
        gains (array_like):
            Complex gains, with the same ordering as `ants`. Expected shape is
            (Nants, Nfreqs, Ntimes).
        ants (array_like):
            Array of antenna IDs.
        antpairs (list of tuples):
            List of antenna pair tuples.
        inline (bool):
            If True, apply the gains to the input array directly. If False,
            return a copy of the input array with the gains applied.

    Returns:
        ggv (array_like):
            A copy of the input `v` array with gains applied.
    """
    if inline:
        ggv = v
    else:
        ggv = v.copy()
    assert v.shape[0] == len(antpairs), \
        "Input array `v` has shape that is incompatible with `antpairs`"

    # Apply gains
    for k, bl in enumerate(antpairs):
        ant1, ant2 = bl
        i1 = np.where(ants == ant1)[0][0]
        i2 = np.where(ants == ant2)[0][0]
        ggv[k,:,:] *= gains[i1] * gains[i2].conj()
    return ggv


def extract_vis_from_sim(ants, antpairs, sim_vis):
    """
    Extract only the desired set of visibilities from a vis_cpu simulation, in
    the desired order.

    vis_cpu outputs a full set of visibilities formed from all antenna pairs,
    but only a subset of these is needed.

    Parameters:
        ants (array_like):
            Ordered list of antenna IDs from the last two dimensions of the
            `sim_vis` array.
        antpairs (list of tuples):
            List of antenna pair tuples. The output ordering will match the
            ordering of this list.
        sim_vis (array_like):
            Array of complex visibility values output by vis_cpu, with expected
            shape (Nfreqs, Ntimes, Nants, Nants).

    Returns:
        vis (array_like):
            Array of complex visibilities with shape (Nbls, Nfreqs, Ntimes).
    """
    Nfreqs, Ntimes, _, _ = sim_vis.shape
    # Allocate computed visibilities to only the requested baselines (saves memory)
    vis = np.zeros((len(antpairs), Nfreqs, Ntimes), dtype=sim_vis.dtype)
    for i, bl in enumerate(antpairs):
        idx1 = np.where(ants == bl[0])[0][0]
        idx2 = np.where(ants == bl[1])[0][0]
        vis[i,:,:] = sim_vis[:,:,idx1,idx2]
    return vis


def timing_info(fname, iter, task, duration, verbose=True):
    """
    Append timing information to a file.

    Parameters:
        fname (str):
            Filename used to save timing info.
        iter (int):
            ID of this iteration.
        task (str):
            Name/description of the task.
        duration (float):
            Duration of task, in seconds.
        verbose (bool):
            If True, print a timing message to stdout too.
    """
    with open("output/timing.dat", "a") as f:
        f.write("%05d %7.5f %s\n" % (iter, duration, task))
        if verbose:
            print("%s took %3.2f sec" % (task, duration))


def build_hex_array(hex_spec=(3,4), ants_per_row=None, d=14.6):
    """
    Build an antenna position dict for a hexagonally close-packed array.

    Parameters:
        hex_spec (tuple):
            If `ants_per_row = None`, this is used to specify a hex array as
            `hex_spec = (nmin, nmax)`, where `nmin` is the number of antennas
            in the bottom and top rows, and `nmax` is the number in the middle
            row. The number per row increases by 1 until the middle row is
            reached. Default: (3,4) [a hex with 3,4,3 antennas per row]

        ants_per_row (array_like):
            Number of antennas per row.

        d (float):
            Minimum baseline length between antennas in the hex array, in
            metres.

    Returns:
        ants (dict):
            Dictionary with antenna IDs as the keys, and tuples with antenna
            (x, y, z) position values (with respect to the array center) as the
            values. Units: metres.
    """
    ants = {}

    # If ants_per_row isn't given, build it from hex_spec
    if ants_per_row is None:
        r = np.arange(hex_spec[0], hex_spec[1]+1).tolist()
        ants_per_row = r[:-1] + r[::-1]

    # Assign antennas
    k = -1
    y = 0.
    dy = d * np.sqrt(3) / 2. # delta y = d sin(60 deg)
    for j, r in enumerate(ants_per_row):

        # Calculate y coord and x offset
        y = -0.5 * dy * (len(ants_per_row)-1) + dy * j
        x = np.linspace(-d*(r-1)/2., d*(r-1)/2., r)
        for i in range(r):
            k += 1
            ants[k] = (x[i], y, 0.)

    return ants


def convert_to_tops(ra, dec, lsts, latitude, precision=1):
    """
    Go to topocentric co-ordinates from sequences of ra, dec at certain lsts,
    observing from a given latitude. Ripped this out of vis_sim_per_source.

    Parameters:
        ra (array_like): Right ascensions of interest.
        dec (array_like): Declinations of interest.
        lsts (array_like): Local sidereal times of observation.
        latitude (float): Latidude of observing, in radians.
        precision (int): If 1, use 32-bit floats, else use 64-bit floats

    Returns:
        txs, tys, tzs: The topocentric co-ordinates (radio astronomers' l, m, n)
    """
    if precision == 1:
        real_dtype = np.float32
    else:
        real_dtype = np.float64
    # Source coordinate transform, from equatorial to Cartesian
    crd_eq = conversions.point_source_crd_eq(ra, dec)

    # Get coordinate transforms as a function of LST
    eq2tops = np.array([conversions.eci_to_enu_matrix(lst, latitude) for lst in lsts])

    txs, tys, tzs = [], [], []

    for t, eq2top in enumerate(eq2tops.astype(real_dtype)):
        # Dot product converts ECI cosines (i.e. from RA and Dec) into ENU
        # (topocentric) cosines, with (tx, ty, tz) = (e, n, u) components
        # relative to the center of the array
        tx, ty, tz = crd_top = np.dot(eq2top, crd_eq)
        txs.append(tx)
        tys.append(ty)
        tzs.append(tz)

    return(txs, tys, tzs)


def get_flux_from_ptsrc_amp(ptsrc_amps, freqs, beta_ptsrc):
    fluxes = ptsrc_amps[:,np.newaxis] * ((freqs / 100.)**beta_ptsrc)[np.newaxis,:]
    return(fluxes)
