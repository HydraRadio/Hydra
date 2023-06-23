
import numpy as np
from vis_cpu import conversions
import pyuvdata
from pyuvsim import AnalyticBeam


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


def load_gain_model(gain_model_file, lst_pad=[0,0], freq_pad=[0,0], pad_value=1.):
    """
    Load complex gain model for each antenna into a single array, and
    zero-pad the edges of the array in the time and frequency dimensions
    if needed.

    Parameters:
        gain_model_file (str):
            Path to file containing a numpy array of shape
            `(Nants, Nfreqs, Ntimes)`. The antenna ordering is assumed to
            be correct; the ordering is not checked.
        lst_pad (list of int):
            How many time samples to add as zero-padding to the data arrays.
            The list should have two entries, with the number of channels to
            add before and after the existing ones.
        freq_pad (list of int):
            Same as `lst_pad`, but for frequency channels.
        pad_value (float):
            Value to use for the gain model in the padded regions.

    Returns:
        gain_model (array_like):
            Model for each antenna gain, of shape `(Nants, Nfreqs, Ntimes)`,
            where the frequency and time dimensions have now been zero-padded
            if requested.
    """
    # Load file
    orig_model = np.load(gain_model_file)

    # Pad the array as requested
    padded_shape = (orig_model.shape[0],
                    orig_model.shape[1] + freq_pad[0] + freq_pad[1],
                    orig_model.shape[2] + lst_pad[0] + lst_pad[1],)
    gain_model = np.zeros(padded_shape, dtype=orig_model.dtype) + pad_value

    # Put gain model into padded array
    gain_model[:,
               freq_pad[0]:orig_model.shape[1]+freq_pad[0],
               lst_pad[0]:orig_model.shape[2]+lst_pad[0]] = orig_model[:,:,:]
    return gain_model


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
        ant1, ant2 = bl

        # Ensure not conjugated
        if ant1 > ant2:
            ant1 = bl[1]
            ant2 = bl[0]

        # Extract data for this antenna pair
        idx1 = np.where(ants == ant1)[0][0]
        idx2 = np.where(ants == ant2)[0][0]
        vis[i,:,:] = sim_vis[:,:,idx1,idx2]
    return vis


def extract_vis_from_uvdata(uvd, exclude_autos=True, lst_pad=[0,0], freq_pad=[0,0]):
    """
    Extract only the desired set of visibilities from a UVData object, in
    the desired order.

    Parameters:
        uvd (array_like):
            UVData object containing the visibility data.
        exclude_autos (bool):
            Whether to exclude auto (zero-spacing) baselines.
        lst_pad (list of int):
            How many time samples to add as zero-padding to the data arrays.
            The list should have two entries, with the number of channels to
            add before and after the existing ones.
        freq_pad (list of int):
            Same as `lst_pad`, but for frequency channels.

    Returns:
        vis (array_like):
            Array of complex visibilities with shape (Nbls, Nfreqs, Ntimes).
        antpair (list of tuple):
            Ordered list of antenna pairs, with the same ordering as the
            first dimension of `vis`.
        ants (list):
            List of unique antenna IDs.
    """
    ants = []
    antpairs = []
    vis = []
    uvd.conjugate_bls(convention='ant1<ant2') # conjugate baselines
    for antpair, bl_data in uvd.antpairpol_iter(squeeze='full'):
        ant1 = antpair[0]
        ant2 = antpair[1]
        Nfreqs = bl_data.shape[1]
        Ntimes = bl_data.shape[0]

        # Add antpair to the list
        if ant1 == ant2 and exclude_autos:
            continue # exclude autos
        antpairs.append((ant1, ant2))

        # Add data for this bl to array (with zero-padding if necessary)
        dat = np.zeros((freq_pad[0] + Nfreqs + freq_pad[1],
                        lst_pad[0] + Ntimes + lst_pad[1]),
                       dtype=bl_data.dtype)
        dat[freq_pad[0]:Nfreqs+freq_pad[0],
            lst_pad[0]:Ntimes+lst_pad[0]] = bl_data[:,:].T
        vis.append(dat)

        # Add antenna to list if not there already
        if ant1 not in ants:
            ants.append(ant1)
        if ant2 not in ants:
            ants.append(ant2)

    return np.array(vis), antpairs, np.array(ants)


def extend_coords_with_padding(arr, pad=[0,0]):
    """
    Take an array with (e.g.) frequencies or times and extend it by a number
    of elements of padding on either side. The values are extrapolated into
    the padded region.

    Parameters:
        arr (array_like):
            Array of coordinates. Expected to be monotonic and equally-spaced.
        pad (list):
            How many array elements of padding to add on either side of the
            array. For example, if `arr` has size `10` and `pad=[2,3]`, the
            new array will have `15` elements, with `2` added to the start
            and `3` added to the end.

    Returns:
        arr_new (array_like):
            New array with coordinates extrapolated into the padded ends.
    """
    # Create new copy of the array with zero padding around it
    arr_new = np.zeros(pad[0] + arr.size + pad[1], dtype=arr.dtype)
    arr_new[pad[0]:arr_new.size-pad[1]] = arr[:]

    # Extrapolate into the padded regions
    diff = arr[1] - arr[0]
    arr_new[arr_new.size-pad[1]:] = arr[-1] + diff * (1.+np.arange(pad[1]))
    arr_new[:pad[0]] = arr[0] - diff * (np.arange(pad[0]) + 1.)[::-1]
    return arr_new


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

    return(np.array(txs), np.array(tys), np.array(tzs))


def get_flux_from_ptsrc_amp(ptsrc_amps, freqs, beta_ptsrc, curv_ptsrc=None, ref_freq=100.):
    """
    Calculate flux as a function of frequency for each point source, assuming
    a power-law spectrum.

    Parameters:
        pstsrc_amps (array_like):
            Amplitude of each point source at the reference frequency (in Jy).
        freqs (array_like):
            Frequencies, in MHz.
        beta_ptsrc (float or array_like):
            Power law spectral index for each source.
        curv_ptsrc (float or array_like):
            Spectral index curvature. If None, no curvature is added.
        ref_freq (float):
            Reference frequency for the power-law spectrum, in MHz.

    Returns:
        fluxes (array_like):
            Flux of each source at each frequency, shape `(Nsrc, Nfreqs)`.
    """
    # Check input arguments
    if isinstance(beta_ptsrc, float):
        beta_ptsrc = beta_ptsrc * np.ones_like(ptsrc_amps)
    assert ptsrc_amps.size == beta_ptsrc.size

    if curv_ptsrc is not None and isinstance(curv_ptsrc, float):
        curv_ptsrc = curv_ptsrc * np.ones_like(ptsrc_amps)
    if curv_ptsrc is None:
        curv_ptsrc = np.zeros_like(ptsrc_amps)
    assert ptsrc_amps.size == curv_ptsrc.size

    # Make SEDs
    spec_idx = beta_ptsrc[:,np.newaxis] + curv_ptsrc[:,np.newaxis] \
                                          * np.log(freqs / ref_freq)[np.newaxis,:]
    fluxes = ptsrc_amps[:,np.newaxis] * (freqs / ref_freq)[np.newaxis,:]**spec_idx
    return fluxes


def antenna_dict_from_uvd(uvd):
    """
    Construct an antenna dictionary (keys are antenna IDs, values are ENU
    x/y/z coords of the antennas) from a UVData object.

    Parameters:
        uvd (UVData object):
            UVData object that contains necessary metadata about antenna
            and telescope position.

    Returns:
        ants (dict):
            Dictionary containing antenna IDs (keys) and x/y/z positions
            in ENU coordinates local to the array (values).
    """
    # Get ENU positions of antennas
    enu = pyuvdata.utils.ENU_from_ECEF(uvd.antenna_positions + uvd.telescope_location,
                                       *uvd.telescope_location_lat_lon_alt)

    # Get antenna IDs
    ant_ids = uvd.antenna_numbers

    assert ant_ids.size == enu.shape[0], \
        "antenna_number and antenna_positions are mis-matched"

    # Output dict
    ants = {}
    for i, ant in enumerate(ant_ids):
        ants[ant] = enu[i]
    return ants


def get_beam_interp_shape(beam):
    """
    Determine whether the spw axis is present in a beam object.

    Parameters:
        beam (AnalyticBeam or UVBeam): The beam object in question. Must be an
            instance of either AnalyticBeam or UVBeam.
    Returns:
        spw_axis_present (bool): Whether the spw axis is present.
    """
    # Check beam type to see shape for return of interp method
    if isinstance(beam, AnalyticBeam):
        spw_axis_present = False

    elif isinstance(beam, pyuvdata.UVBeam):
        spw_axis_present = not beam.future_array_shapes
    else:
        raise ValueError("beam object is not AnalyticBeam or UVBeam object.")

    return spw_axis_present
