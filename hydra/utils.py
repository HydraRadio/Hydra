import numpy as np
from matvis import coordinates as conversions
import pyuvdata
#from pyuvsim import AnalyticBeam

"""
# Terminal colour codes
#terminal_ = '\033[95m'
terminal_blue = '\033[94m'
terminal_cyan = '\033[96m'
terminal_green = '\033[92m'
terminal_yellow = '\033[93m'
#FAIL = '\033[91m'
terminal_endc = '\033[0m'
terminal_bold = '\033[1m'
terminal_ul = '\033[4m'
"""


def flatten_vector(v, reduced_idxs=None):
    """
    Flatten a complex vector with shape (N, Ntimes, Nfreq) into a block
    vector of shape (N x Ntimes x Nfreqs x 2), i.e. the real and imaginary
    blocks stuck together.

    Parameters:
        reduced_idxs (array_like):
            If specified, this is an array of indices that maps a reduced x
            vector to the full x vector. The unspecified modes are set to zero.
    """
    # If only certain indices were kept, remove others
    if reduced_idxs is not None:
        return np.concatenate((v.real.flatten(), v.imag.flatten()))[reduced_idxs]
    else:
        return np.concatenate((v.real.flatten(), v.imag.flatten()))


def reconstruct_vector(v, shape, reduced_idxs=None):
    """
    Undo the flattening of a complex vector.

    Parameters:
        reduced_idxs (array_like):
            If specified, this is an array of indices that maps a reduced x
            vector to the full x vector expected by the linear operator. The
            unspecified modes are set to zero.
    """
    # If only certain indices were kept, populate those into a vector of zeros
    if reduced_idxs is not None:
        vv = np.zeros(
            2 * np.prod(shape), dtype=v.dtype
        )  # required size = product of shape tuple
        vv[reduced_idxs] = v[:]
    else:
        vv = v

    # Now unpack the full-sized array
    y = vv[: vv.size // 2] + 1.0j * vv[vv.size // 2 :]
    return y.reshape(shape)


def apply_gains(v, gains, ants, antpairs, perturbation=None, inline=False):
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
        perturbation (array_like):
            Linear perturbations to the gains. If set, this will apply these
            perturbations under the linear approximation, i.e.
            `g_i g_j* \approx \bar{g}_i \bar{g}_j^* (1 + x_i + x_j^*)`.
            Expected shape is (Nants, Nfreqs, Ntimes).
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
    assert v.shape[0] == len(
        antpairs
    ), "Input array `v` has shape that is incompatible with `antpairs`"

    # Apply gains
    for k, bl in enumerate(antpairs):
        ant1, ant2 = bl
        i1 = np.where(ants == ant1)[0][0]
        i2 = np.where(ants == ant2)[0][0]
        fac = 1.0
        if perturbation is not None:
            fac = 1.0 + perturbation[i1] + perturbation[i2].conj()
        ggv[k, :, :] *= gains[i1] * gains[i2].conj() * fac
    return ggv


def load_gain_model(gain_model_file, lst_pad=[0, 0], freq_pad=[0, 0], pad_value=1.0):
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
    padded_shape = (
        orig_model.shape[0],
        orig_model.shape[1] + freq_pad[0] + freq_pad[1],
        orig_model.shape[2] + lst_pad[0] + lst_pad[1],
    )
    gain_model = np.zeros(padded_shape, dtype=orig_model.dtype) + pad_value

    # Put gain model into padded array
    gain_model[
        :,
        freq_pad[0] : orig_model.shape[1] + freq_pad[0],
        lst_pad[0] : orig_model.shape[2] + lst_pad[0],
    ] = orig_model[:, :, :]
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
        vis[i, :, :] = sim_vis[:, :, idx1, idx2]
    return vis


def extract_vis_from_uvdata(uvd, exclude_autos=True, lst_pad=[0, 0], freq_pad=[0, 0]):
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
    uvd.conjugate_bls(convention="ant1<ant2")  # conjugate baselines
    for antpair, bl_data in uvd.antpairpol_iter(squeeze="full"):
        ant1 = antpair[0]
        ant2 = antpair[1]
        Nfreqs = bl_data.shape[1]
        Ntimes = bl_data.shape[0]

        # Add antpair to the list
        if ant1 == ant2 and exclude_autos:
            continue  # exclude autos
        antpairs.append((ant1, ant2))

        # Add data for this bl to array (with zero-padding if necessary)
        dat = np.zeros(
            (freq_pad[0] + Nfreqs + freq_pad[1], lst_pad[0] + Ntimes + lst_pad[1]),
            dtype=bl_data.dtype,
        )
        dat[freq_pad[0] : Nfreqs + freq_pad[0], lst_pad[0] : Ntimes + lst_pad[0]] = (
            bl_data[:, :].T
        )
        vis.append(dat)

        # Add antenna to list if not there already
        if ant1 not in ants:
            ants.append(ant1)
        if ant2 not in ants:
            ants.append(ant2)

    return np.array(vis), antpairs, np.array(ants)


def extend_coords_with_padding(arr, pad=[0, 0]):
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
    arr_new[pad[0] : arr_new.size - pad[1]] = arr[:]

    # Extrapolate into the padded regions
    diff = arr[1] - arr[0]
    arr_new[arr_new.size - pad[1] :] = arr[-1] + diff * (1.0 + np.arange(pad[1]))
    arr_new[: pad[0]] = arr[0] - diff * (np.arange(pad[0]) + 1.0)[::-1]
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
    with open(fname, "a") as f:
        f.write("%05d %7.5f %s\n" % (iter, duration, task))
        if verbose:
            print("%s took %3.2f sec" % (task, duration))


def freqs_times_for_worker(comm, freqs, times, fchunks, tchunks=1):
    """
    Get the unique frequency and time chunk for a worker. The workers
    are arranged onto an `fchunks * tchunks` grid, and each worker is
    given the corresponding chunk of the 2D `freqs * times` grid.

    Parameters:
        comm (MPI Communicator):
            MPI communicator.
        freqs (array_like):
            Array of all frequencies in the data.
        times (array_like):
            Array of all times (LSTs) in the data.
        fchunks (int):
            Number of chunks to divide the frequency array into.
        tchunks (int):
            Number of chunks to divide the time array into.

    Returns:
        freq_idx_chunk (array_like):
            The indices of the frequency array belonging to this worker.
        time_idx_chunk (array_like):
            The indices of the time array belonging to this worker.
        worker_map (dict):
            A dictionary containing the chunk indices and frequency and
            time indices for each worker. This is useful for cases where
            individual workers have to communicate with each other. The
            format of the dictionary entries is:
            `worker_id: (freq_chunk_idx, time_chunk_idx, freq_idxs, time_idxs)`
            where `freq_chunk_idx` and `time_chunk_idx` are indices in the
            grid of chunks for this worker, and `freq_idxs` and `time_idxs`
            are the actual indices in the `freqs` and `times` arrays belonging
            to this worker.
    """
    myid = comm.Get_rank()
    nworkers = comm.Get_size()
    assert (
        nworkers <= fchunks * tchunks
    ), "There are more workers than time and frequency chunks"

    # Get chunk ID for this worker
    allidxs = np.arange(fchunks * tchunks).reshape((fchunks, tchunks))
    fidx, tidx = np.where(allidxs == myid)
    fidx, tidx = int(fidx[0]), int(tidx[0])

    # Get chunk of freq/time idxs for each worker
    freq_idxs = np.arange(freqs.size)
    time_idxs = np.arange(times.size)
    freq_idx_chunks = np.array_split(freq_idxs, fchunks)
    time_idx_chunks = np.array_split(time_idxs, tchunks)

    # Make map of freq. and time idxs for each worker
    worker_map = {}
    for i in range(nworkers):
        _fidx, _tidx = np.where(allidxs == i)
        _fidx, _tidx = int(_fidx[0]), int(_tidx[0])
        worker_map[i] = (_fidx, _tidx, freq_idx_chunks[_fidx], time_idx_chunks[_tidx])

    return freq_idx_chunks[fidx], time_idx_chunks[tidx], worker_map


def build_hex_array(hex_spec=(3, 4), ants_per_row=None, d=14.6):
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
        r = np.arange(hex_spec[0], hex_spec[1] + 1).tolist()
        ants_per_row = r[:-1] + r[::-1]

    # Assign antennas
    k = -1
    y = 0.0
    dy = d * np.sqrt(3) / 2.0  # delta y = d sin(60 deg)
    for j, r in enumerate(ants_per_row):

        # Calculate y coord and x offset
        y = -0.5 * dy * (len(ants_per_row) - 1) + dy * j
        x = np.linspace(-d * (r - 1) / 2.0, d * (r - 1) / 2.0, r)
        for i in range(r):
            k += 1
            ants[k] = (x[i], y, 0.0)

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

    return (np.array(txs), np.array(tys), np.array(tzs))


def get_flux_from_ptsrc_amp(
    ptsrc_amps, freqs, beta_ptsrc, curv_ptsrc=None, ref_freq=100.0
):
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
    spec_idx = (
        beta_ptsrc[:, np.newaxis]
        + curv_ptsrc[:, np.newaxis] * np.log(freqs / ref_freq)[np.newaxis, :]
    )
    fluxes = ptsrc_amps[:, np.newaxis] * (freqs / ref_freq)[np.newaxis, :] ** spec_idx
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
    lat, lon, alt = uvd.telescope_location_lat_lon_alt
    enu = pyuvdata.utils.ENU_from_ECEF(
        uvd.antenna_positions + uvd.telescope_location,
        latitude=lat,
        longitude=lon,
        altitude=alt
    )

    # Get antenna IDs
    ant_ids = uvd.antenna_numbers

    assert (
        ant_ids.size == enu.shape[0]
    ), "antenna_number and antenna_positions are mis-matched"

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
    if isinstance(beam, pyuvdata.UVBeam):
        spw_axis_present = not beam.future_array_shapes
    else:
        spw_axis_present = False
        #raise ValueError("beam object is not AnalyticBeam or UVBeam object.")

    return spw_axis_present


def gain_prior_pspec_sqrt(
    lsts,
    freqs,
    gain_prior_amp,
    gain_prior_sigma_frate=None,
    gain_prior_sigma_delay=None,
    gain_prior_zeropoint_std=None,
    frate0=0.0,
    delay0=0.0,
):
    """
    Construct a gain prior power spectrum, which can be uniform or have a
    Gaussian taper in the delay and/or fringe rate directions.

    Parameters:
        lsts (array_like):
            LST grid (same as for the data), in radians.
        freqs (array_like):
            Frequency grid (same as for the data), in MHz.
        gain_prior_amp (float):
            Amplitude of the prior power spectrum.
        gain_prior_sigma_frate (float):
            Width of a Gaussian prior in fringe rate, in units of mHz. If
            `None`, the prior will be uniform in fringe rate.
        gain_prior_sigma_delay (float):
            Width of a Gaussian prior in delay, in units of ns. If `None`, the
            prior will be uniform in delay.
        gain_prior_zeropoint_std (float):
            If not `None`, fix the std. dev. of the (0,0) mode to some value.
        frate0 (float):
            The central fringe rate of the Gaussian taper (mHz).
        delay0 (float):
            The central delay of the Gaussian taper (ns).

    Returns:
        gain_pspec_sqrt (array_like):
            Square-root of the gain prior power spectrum, in delay-fringe rate
            space (FFT ordering). Shape (Nfreqs, Ntimes).
    """
    times = 24.0 * 60.0 * 60.0 * lsts / (2.0 * np.pi)  # seconds
    frate = 1e3 * np.fft.fftfreq(times.size, d=times[1] - times[0])  # mHz
    delay = 1e3 * np.fft.fftfreq(freqs.size, d=freqs[1] - freqs[0])  # ns

    # Start with flat prior and then multiply by
    # Ordering is usual FFT ordering
    gain_pspec_sqrt = gain_prior_amp * np.ones((freqs.size, times.size))
    if gain_prior_sigma_frate is not None:
        xt = (frate[np.newaxis, :] - frate0) / gain_prior_sigma_frate
        gain_pspec_sqrt *= np.exp(-0.5 * xt**2.0)
    if gain_prior_sigma_delay is not None:
        xf = (delay[:, np.newaxis] - delay0) / gain_prior_sigma_delay
        gain_pspec_sqrt *= np.exp(-0.5 * xf**2.0)

    # Zero-point prior
    if gain_prior_zeropoint_std is not None:
        gain_pspec_sqrt[delay == 0.0, frate == 0.0] = gain_prior_zeropoint_std

    return gain_pspec_sqrt


def partial_fourier_basis_2d(
    freqs, times, nfreq, ntime, Lfreq, Ltime, freq0=None, time0=None, shape0=None
):
    """
    Construct a set of 2D Fourier modes from a list of wavenumber integers,
    to form an incomplete set of 2D Fourier modes.
    """
    # Decide on origin of frequency axis for FT
    if time0 is None:
        time0 = times[0]
    if freq0 is None:
        freq0 = freqs[0]

    # Determine normalising factors. If being used as a standalone Fourier operator,
    # these are just the lengths of the freq and time arrays. If being used as a
    # chunk of a Fourier operator across multiple workers, use the overall shape
    # from 'shape0'
    if shape0 is None:
        Nfreqs = freqs.size
        Ntimes = times.size
    else:
        Nfreqs, Ntimes = shape0

    # Build grid of freqs and times
    nfreq = np.atleast_1d(nfreq)
    ntime = np.atleast_1d(ntime)
    assert len(nfreq.shape) == 1
    assert len(ntime.shape) == 1
    assert len(freqs.shape) == 1
    assert len(times.shape) == 1
    t2d, f2d = np.meshgrid(times - time0, freqs - freq0)

    # Calculate wavenumbers for each mode
    kfreq = 2.0 * np.pi * nfreq / Lfreq  # inverse freq. units
    ktime = 2.0 * np.pi * ntime / Ltime  # inverse time units

    # Shape: (Nmodes, Nfreqs, Ntimes)
    basis_fns = np.exp(
        1.0j
        * (
            (kfreq[:, np.newaxis, np.newaxis] * f2d[np.newaxis, :, :])
            + (ktime[:, np.newaxis, np.newaxis] * t2d[np.newaxis, :, :])
        )
    ) / np.sqrt(Nfreqs * Ntimes)
    return basis_fns, kfreq, ktime


def partial_fourier_basis_2d_from_nmax(
    freqs,
    times,
    nmaxfreq,
    nmaxtime,
    Lfreq,
    Ltime,
    freq0=None,
    time0=None,
    shape0=None,
    positive_only=False,
):
    """
    Convenience function to construct a set of 2D Fourier modes with wavenumbers
    between -nmax <= 0 < nmax.
    """
    # Make grid of wavenumber values for both frequency and time, in the range
    # -nmax <= 0 < nmax (matches the modes you get from an FFT for 2*nmax points)
    if positive_only:
        _nfreq = np.arange(0, nmaxfreq)
        _ntime = np.arange(0, nmaxtime)
    else:
        _nfreq = np.arange(-nmaxfreq, nmaxfreq)
        _ntime = np.arange(-nmaxtime, nmaxtime)
    nfreq, ntime = np.meshgrid(_nfreq, _ntime)

    # Use generic function to construct Fourier basis functions
    basis_fns, kfreq, ktime = partial_fourier_basis_2d(
        freqs=freqs,
        times=times,
        nfreq=nfreq.flatten(),
        ntime=ntime.flatten(),
        Lfreq=Lfreq,
        Ltime=Ltime,
        freq0=freq0,
        time0=time0,
        shape0=shape0,
    )
    return basis_fns, kfreq, ktime


def status(myid, message, colour=None):
    """
    Print a status message.
    """
    # terminal_ = '\033[95m'
    endchar = "\033[0m"
    colours = {
        "r": "\033[91m",
        "g": "\033[92m",
        "y": "\033[93m",
        "b": "\033[94m",
        "m": "\033[95m",
        "c": "\033[96m",
        "bold": "\033[1m",
        "ul": "\033[4m",
    }

    # Worker ID, if present
    if myid is None:
        myid_str = ""
    else:
        myid_str = "[%d]" % myid

    if colour is not None:
        print("%s%s %s%s" % (colours[colour], myid_str, message, endchar))
    else:
        print("%s %s" % (myid_str, message))
