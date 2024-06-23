
try:
    from mpi4py.MPI import SUM as MPI_SUM
except:
    pass
    
import numpy as np
import numpy.fft as fft

import matvis
import pyuvsim
import time

from .vis_simulator import simulate_vis_per_source


def precompute_mpi(
    comm,
    ants,
    antpairs,
    freq_chunk,
    time_chunk,
    proj_chunk,
    data_chunk,
    inv_noise_var_chunk,
    gain_chunk,
    amp_prior_std,
    realisation=True,
):
    """
    Precompute the projection operator and matrix operator in parallel.

    The projection operator is computed in chunks in time and frequency.
    The overall matrix operator can be computed by summing the matrix
    operator for the time and frequency chunks.
    """
    myid = comm.Get_rank()

    # Check input dimensions
    assert data_chunk.shape == (len(antpairs), freq_chunk.size, time_chunk.size)
    assert data_chunk.shape == inv_noise_var_chunk.shape
    proj = proj_chunk.copy()  # make a copy so we don't alter the original proj!

    # FIXME: Check for unused args!

    # Apply gains to projection operator
    for k, bl in enumerate(antpairs):
        ant1, ant2 = bl
        i1 = np.where(ants == ant1)[0][0]
        i2 = np.where(ants == ant2)[0][0]
        proj[k, :, :, :] *= (
            gain_chunk[i1, :, :, np.newaxis] * gain_chunk[i2, :, :, np.newaxis].conj()
        )

    # (2) Precompute linear system operator
    nsrcs = proj.shape[-1]
    my_linear_op = np.zeros((nsrcs, nsrcs), dtype=proj.real.dtype)

    # inv_noise_var has shape (Nbls, Nfreqs, Ntimes)
    v_re = (proj.real * np.sqrt(inv_noise_var_chunk[..., np.newaxis])).reshape(
        (-1, nsrcs)
    )
    v_im = (proj.imag * np.sqrt(inv_noise_var_chunk[..., np.newaxis])).reshape(
        ((-1, nsrcs))
    )

    # Treat real and imaginary separately, and get copies, to massively
    # speed-up the matrix multiplication!
    my_linear_op[:, :] = v_re.T @ v_re + v_im.T @ v_im
    del v_re, v_im

    # Do Reduce (sum) operation to get total operator on root node
    linear_op = np.zeros(
        (1, 1), dtype=my_linear_op.dtype
    )  # dummy data for non-root workers
    if myid == 0:
        linear_op = np.zeros_like(my_linear_op)

    comm.Reduce(my_linear_op, linear_op, op=MPI_SUM, root=0)

    # Include prior and identity terms to finish constructing LHS operator on root worker
    if myid == 0:
        linear_op = np.eye(linear_op.shape[0]) + np.diag(
            amp_prior_std
        ) @ linear_op @ np.diag(amp_prior_std)

    # (3) Calculate linear system RHS
    proj = proj.reshape((-1, nsrcs))
    realisation_switch = (
        1.0 if realisation else 0.0
    )  # Turn random realisations on or off

    # Calculate residual of data vs fiducial model (residual from amplitudes = 1)
    # (proj now includes gains)
    resid_chunk = data_chunk.copy() - (
        proj.reshape((-1, nsrcs)) @ np.ones_like(amp_prior_std)
    ).reshape(data_chunk.shape)

    # (Terms 1+3): S^1/2 A^\dagger [ N^{-1} r + N^{-1/2} \omega_r ]
    omega_n = (
        realisation_switch
        * (
            1.0 * np.random.randn(*resid_chunk.shape)
            + 1.0j * np.random.randn(*resid_chunk.shape)
        )
        / np.sqrt(2.0)
    )

    # Separate complex part of RHS into real and imaginary parts, and apply
    # the real and imaginary parts of the projection operator separately.
    # This is necessary to get a real RHS vector
    y = (
        (resid_chunk * inv_noise_var_chunk) + (omega_n * np.sqrt(inv_noise_var_chunk))
    ).flatten()
    b = amp_prior_std * (proj.T.real @ y.real + proj.T.imag @ y.imag)

    # Reduce (sum) operation on b
    linear_rhs = np.zeros((1,), dtype=b.dtype)  # dummy data for non-root workers
    if myid == 0:
        linear_rhs = np.zeros_like(b)
    comm.Reduce(b, linear_rhs, op=MPI_SUM, root=0)

    # (Term 2): \omega_a
    if myid == 0:
        linear_rhs += realisation_switch * np.random.randn(nsrcs)  # real vector

    return linear_op, linear_rhs


def calc_proj_operator(
    ra,
    dec,
    fluxes,
    ant_pos,
    antpairs,
    freqs,
    times,
    beams,
    latitude=-0.5361913261514378,
):
    """
    Calculate a visibility vector for each point source, as a function of
    frequency, time, and baseline. This is the projection operator from point
    source amplitude to visibilities. Gains are not included.

    Parameters:
        ra, dec (array_like):
            RA and Dec of each source, in radians.
        fluxes (array_like):
            Flux for each point source as a function of frequency.
        ant_pos (dict):
            Dictionary of antenna positions, [x, y, z], in m. The keys should
            be the numerical antenna IDs.
        antpairs (list of tuple):
            List of tuples containing pairs of antenna IDs, one for each
            baseline.
        freqs (array_like):
            Frequencies, in MHz.
        times (array_like):
            LSTs, in radians.
        beams (list of UVBeam):
            List of UVBeam objects, one for each antenna.
        latitude (float):
            Latitude of the observing site, in radians.

    Returns:
        vis_proj_operator (array_like):
            The projection operator from source amplitudes to visibilities,
            from `calc_proj_operator`. This is an array of the visibility
            values for each source.
    """
    Nptsrc = ra.size
    Nants = len(ant_pos)
    Nvis = len(antpairs)

    # Empty array of per-point source visibilities
    vis_ptsrc = np.zeros((Nvis, freqs.size, times.size, Nptsrc), dtype=np.complex128)

    # Get visibility for each point source
    # Returns shape (NFREQS, NTIMES, NANTS, NANTS, NSRCS)
    vis = simulate_vis_per_source(
        ants=ant_pos,
        fluxes=fluxes,
        ra=ra,
        dec=dec,
        freqs=freqs * 1e6,
        lsts=times,
        beams=beams,
        polarized=False,
        precision=2,
        latitude=latitude,
        use_feed="x",
    )

    # Allocate computed visibilities to only available baselines (saves memory)
    ants = list(ant_pos.keys())
    for i, bl in enumerate(antpairs):
        idx1 = ants.index(bl[0])
        idx2 = ants.index(bl[1])
        vis_ptsrc[i, :, :, :] = vis[:, :, idx1, idx2, :]

    return vis_ptsrc