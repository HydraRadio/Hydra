import numpy as np


def make_cosmo_field_grid(args):
    """
    Make a regular Cartesian grid of points in RA and Dec that sample a
    cosmological 21cm field.

    Parameters:
        args (argparse object):
            An argparse object containing the following settings:
            `cosmo_field_ra_bounds`, `cosmo_field_dec_bounds`,
            `cosmo_field_ra_ngrid`, `cosmo_field_dec_ngrid`.

    Returns:
        ra_grid, dec_grid (array_like):
            RA and Dec values of the sample points, in radians.
    """
    # Define sample points
    ra = np.linspace(
        min(args.cosmo_field_ra_bounds),
        max(args.cosmo_field_ra_bounds),
        args.cosmo_field_ra_ngrid,
    )
    dec = np.linspace(
        min(args.cosmo_field_dec_bounds),
        max(args.cosmo_field_dec_bounds),
        args.cosmo_field_dec_ngrid,
    )

    # Define 2D grid
    ra_grid, dec_grid = np.meshgrid(ra, dec)
    return ra_grid.flatten(), dec_grid.flatten()


def precompute_mpi(
    comm,
    ants,
    antpairs,
    freqs,
    freq_chunk,
    time_chunk,
    proj_chunk,
    data_chunk,
    inv_noise_var_chunk,
    current_data_model_chunk,
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
    assert data_chunk.shape == current_data_model_chunk.shape
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

    # (2) Precompute linear system operator for each frequency (for the
    # likelihood part of the operator, the freqs. don't talk to each other)
    Npix = proj.shape[-1]
    my_linear_op = np.zeros((freqs.size, Npix, Npix), dtype=proj.real.dtype)

    # inv_noise_var has shape (Nbls, Nfreqs, Ntimes); proj has shape (Nbls, Nfreqs, Ntimes, Npix)
    v_re = (proj.real * np.sqrt(inv_noise_var_chunk[..., np.newaxis])).reshape(
        (-1, Npix)
    )
    v_im = (proj.imag * np.sqrt(inv_noise_var_chunk[..., np.newaxis])).reshape(
        ((-1, Npix))
    )

    # Treat real and imaginary separately; treat frequencies separately
    # FIXME: Is this neglecting real/imag cross-terms?
    for j in range(freq_chunk.size):
        i = np.where(freqs == freq_chunk[j])[0]
        my_linear_op[i, :, :] = (
            v_re[:, j, :, :].T @ v_re[:, j, :, :]
            + v_im[:, j, :, :].T @ v_im[:, j, :, :]
        )
    del v_re, v_im

    # Do Reduce (sum) operation to get total operator on root node
    linear_op = np.zeros(
        (1, 1, 1), dtype=my_linear_op.dtype
    )  # dummy data for non-root workers
    if myid == 0:
        linear_op = np.zeros_like(my_linear_op)

    comm.Reduce(my_linear_op, linear_op, op=MPI_SUM, root=0)

    #######################
    ### FIXME: Got to here

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

    # Construct current state of model (residual from amplitudes = 1)
    # (proj now includes gains)
    resid_chunk = data_chunk.copy() - (
        proj.reshape((-1, nsrcs)) @ np.ones_like(amp_prior_std)
    ).reshape(current_data_model_chunk.shape)

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


def apply_operator(x, freqs, ra_pix, dec_pix, linear_op_term, pspec):
    """

    Parameters:
        x (array_like):
            1D array of cosmo field values that can be reshaped to `(Nfreqs, Npix)`.
    """
    # Get 3D pixel grid shape
    Nfreqs = freqs.size
    Nx = ra_pix.size
    Ny = dec_pix.size
    Npix = Nx * Ny

    # Reshape x and apply A^T N^-1 A term to x
    x_vec = x.reshape((Nfreqs, Npix))
    y_vec = np.zeros_like(x_vec)
    for j in range(Nfreqs):
        y_vec[j] = linear_op_term[j] @ x_vec[j]

    # Apply prior term to x vector
    x_arr = x.reshape((Nfreqs, Nx, Ny))
    y_vecnp.fft.ifftn(pspec * np.fft.fftn(x_arr)).reshape(y_vec.shape)
