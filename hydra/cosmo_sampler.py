
from mpi4py.MPI import SUM as MPI_SUM

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid

C = 299792.458 # speed of light, km/s
FREQ_21CM = 1420.405751768 # MHz


def calculate_cosmo_fns(h=0.69, omega_m=0.31):
    """
    Calculate H(z) and D_A(z) for a given cosmology and return interpolation 
    functions. Assumes a flat LambdaCDM cosmology.

    Parameters:
        h (float):
            Dimensionless Hubble parameter, defined through:
            `H_0 = (100 h)` km/s/Mpc.
        omega_m (float):
            Total fractional density of matter components.
    
    Returns:
        Hz (func):
            Interpolation function to calculate H(z) in units of km/s/Mpc.
        dA_comoving (func):
            Interpolation function to calculate comoving angular diameter 
            distance in units of Mpc.
    """
    # Calculate H(z)
    zz = np.concatenate(([0.,], np.logspace(-4., 2.)))
    _Hz = 100.*h*np.sqrt(omega_m*(1.+zz)**3. + (1. - omega_m)) # in km/s/Mpc
    Hz = interp1d(zz, _Hz, kind='quadratic')
    
    # Calculate d_A(z) = r(z) [comoving angular diameter distance]
    _dAc = cumulative_trapezoid(C / Hz(zz), zz, initial=0.)
    dAc = interp1d(zz, _dAc, kind='quadratic')

    return Hz, dAc


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
    ra = np.linspace(min(args.cosmo_field_ra_bounds), 
                     max(args.cosmo_field_ra_bounds), 
                     args.cosmo_field_ra_ngrid)
    dec = np.linspace(min(args.cosmo_field_dec_bounds), 
                      max(args.cosmo_field_dec_bounds), 
                      args.cosmo_field_dec_ngrid)

    # Define 2D grid
    ra_grid, dec_grid = np.meshgrid(ra, dec)
    return np.deg2rad(ra_grid.flatten()), np.deg2rad(dec_grid.flatten()) 


def comoving_fourier_modes(x, y, freqs, **cosmo_params):
    """
    Calculate the comoving Fourier wavenumbers for a given grid of points 
    in RA, Dec, and frequency. The units are Mpc^-1.

    Parameters:
        x, y (array_like):
            Coordinates of angular grid points on the sky. Each is a 1D array. 
            Sky curvature is ignored, and the units are degrees.
        freqs (array_like):
            Coordinates of radial (frequency) grid points. This is a 1D array 
            in units of MHz.
        cosmo_params (dict):
            Cosmological background model parameters that are passed to 
            `calculate_cosmo_fns()`.

    Returns:
        kx, ky, knu (array_like):
            Arrays of comoving wavenumber values, in Mpc^-1, in the x, y, and 
            frequency directions. The ordering of the wavenumbers is the FFT 
            ordering given by ``fftfreq()``.
    """
    # Calculate cosmo params
    # FIXME: No need to recalculate this each time
    Hz, dA_comoving = calculate_cosmo_fns(**cosmo_params)

    # Calculate centre frequency
    centre_freq = 0.5 * (freqs.min() + freqs.max())
    zc = (FREQ_21CM / centre_freq) - 1.

    # Calculate 3D Fourier modes
    dAc = dA_comoving(zc)
    dx_perp = np.deg2rad(x[1] - x[0]) * dAc # Mpc
    dy_perp = np.deg2rad(y[1] - y[0]) * dAc # Mpc
    dnu = (C * (1. + zc)**2. / Hz(zc)) * (freqs[1] - freqs[0]) / FREQ_21CM # Mpc
    kx = 2.*np.pi*np.fft.fftfreq(n=x.size, d=dx_perp) # fftfreq outputs per-cycle units
    ky = 2.*np.pi*np.fft.fftfreq(n=y.size, d=dy_perp)
    knu = 2.*np.pi*np.fft.fftfreq(n=freqs.size, d=dnu)

    return kx, ky, knu


def calculate_pspec_on_grid(kbins, pspec, x, y, freqs, **cosmo_params):
    """
    Calculate 1D power spectrum values on a grid of 3D Fourier modes.

    Parameters:
        kbins (array_like):
            Array of |k| bin centres for 1D power spectrum.
        pspec (array_like):
            Array of binned 1D power spectrum values, `P(|k|)`. The 1D 
            wavenumbers are defined as `|k| = sqrt(kx^2 + ky^2 + kz^2)`.
        x, y (array_like):
            Coordinates of angular grid points on the sky. Each is a 1D array. 
            Sky curvature is ignored, and the units are degrees.
        freqs (array_like):
            Coordinates of radial (frequency) grid points. This is a 1D array 
            in units of MHz.
        cosmo_params (dict):
            Cosmological background model parameters that are passed to 
            `calculate_cosmo_fns()`.

    Returns:
        pspec (array_like):
            Power spectrum values on a 3D grid.
    """
    # Check inputs are 1D
    assert len(np.shape(x)) == len(np.shape(y)) == len(np.shape(freqs)) == 1, \
        "x, y, and freqs coordinate arrays must be 1D"

    # Check that grids are regular (difference between grid points is a constant)
    assert np.unique(np.diff(x)).size == np.unique(np.diff(y)).size \
                                      == np.unique(np.diff(freqs)).size \
                                      == 1, \
           "x, y, and freqs grids must each be uniformly spaced"

    # Check that kbins and pspec have the same length
    assert kbins.size == pspec.size, "kbins and pspec must have the same length"

    # Get comoving Fourier modes
    kx, ky, knu = comoving_fourier_modes(x, y, freqs, **cosmo_params)

    # Get 3D Fourier grid
    kx3d, ky3d, knu3d = np.meshgrid(kx, ky, knu)
    k = np.sqrt(kx3d**2. + ky3d**2. + knu3d**2.)

    # Make interpolation function for power spectrum bandpowers
    pspec_fn = interp1d(kbins, pspec, kind='nearest', 
                        fill_value=(pspec[0], pspec[-1]))

    # Interpolate and return
    pspec = pspec_fn(k)
    return pspec


def apply_S(x, pspec, exponent=1):
    """
    Apply the prior covariance matrix to a vector by doing a multiplication 
    in Fourier space.

    The calculation is: `FFT^-1( (pspec)^exponent * FFT(x) )`.

    Parameters:
        x (array_like):
            Input vector in configuration space, with dimensions 
            `(Nx, Ny, Nfreqs)`. A 3D FFT will be applied.
        pspec (array_like):
            A vector of power spectrum values, `P(|k|)`, on the 3D Fourier 
            grid. This can be calculated by `calculate_pspec_on_grid()`.
        exponent (float):
            Exponent to apply to the power spectrum when performing the 
            multiplication.

    Returns:
        y (array_like):
            Result of applying the prior covariance matrix to the input 
            vector. This will be returned with shape `(Nx, Ny, Nfreqs)`.
    """
    assert len(x.shape) == 3, \
        "Input vector x must be a 3D array with shape (Nx, Ny, Nfreqs)"

    return np.fft.ifftn((pspec)**exponent * np.fft.fftn(x))


def apply_lhs_operator(x, lhs_Ninv_operator, pspec):
    """
    
    Parameters:
        x (array_like):
            Set of field values on a 3D grid. Has shape (Nfreqs, Nx, Ny).
        lhs_Ninv_operator (array_like):
            xx
        pspec (function):

    """



def precompute_mpi(comm,
                   ants, 
                   antpairs,
                   freqs,
                   freq_chunk, 
                   time_chunk,
                   proj_chunk,
                   data_chunk,
                   inv_noise_var_chunk,
                   gain_chunk, 
                   pspec3d, 
                   grid_shape,
                   realisation=True):
    """
    Precompute the projection operator and matrix operator in parallel. 

    The projection operator is computed in chunks in time and frequency. 
    The overall matrix operator can be computed by summing the matrix 
    operator for the time and frequency chunks.

    Parameters:
        x

    Returns:
        x
    """
    myid = comm.Get_rank()

    # Check input dimensions
    assert data_chunk.shape == (len(antpairs), freq_chunk.size, time_chunk.size)
    assert data_chunk.shape == inv_noise_var_chunk.shape
    proj = proj_chunk.copy() # make a copy so we don't alter the original proj!

    # Apply gains to projection operator
    for k, bl in enumerate(antpairs):
        ant1, ant2 = bl
        i1 = np.where(ants == ant1)[0][0]
        i2 = np.where(ants == ant2)[0][0]
        proj[k,:,:,:] *= gain_chunk[i1,:,:,np.newaxis] \
                       * gain_chunk[i2,:,:,np.newaxis].conj()

    # (2) Precompute linear system operator for each frequency (for the 
    # likelihood part of the operator, the freqs. don't talk to each other)
    Nbls, Nfreqs_chunk, Ntimes_chunk, Npix = proj.shape
    my_linear_op = np.zeros((freqs.size, Npix, Npix), dtype=proj.real.dtype)

    # inv_noise_var has shape (Nbls, Nfreqs, Ntimes); proj has shape (Nbls, Nfreqs, Ntimes, Npix) 
    v_re = (proj.real * np.sqrt(inv_noise_var_chunk[...,np.newaxis]))
    v_im = (proj.imag * np.sqrt(inv_noise_var_chunk[...,np.newaxis]))

    # Treat real and imaginary separately; treat frequencies separately
    # FIXME: Is this neglecting real/imag cross-terms?
    for j in range(freq_chunk.size):
        # Get frequency index of locally-held frequency channels
        i = np.where(freqs == freq_chunk[j])[0]
        _vre = v_re[:,j,:,:].reshape((-1, Npix))
        _vim = v_im[:,j,:,:].reshape((-1, Npix))
        my_linear_op[i,:,:] = _vre.T @ _vre + _vim.T @ _vim
    del v_re, v_im

    # Do Reduce (sum) operation to get total operator on root node
    linear_op = np.zeros((1,1,1), dtype=my_linear_op.dtype) # dummy data for non-root workers
    if myid == 0:
        linear_op = np.zeros_like(my_linear_op)

    comm.Reduce(my_linear_op,
                linear_op,
                op=MPI_SUM,
                root=0)

    # (3) Calculate linear system RHS
    realisation_switch = 1.0 if realisation else 0.0 # Turn random realisations on or off

    # (Terms 1+3): A^\dagger [ N^{-1} r + N^{-1/2} \omega_r ]
    # FIXME: random seed?
    omega_n = (
        realisation_switch
        * (1.0 * np.random.randn(*data_chunk.shape) + 1.0j * np.random.randn(*data_chunk.shape))
        / np.sqrt(2.0)
    )

    # Calculate data/noise-dependent terms of RHS for this chunk
    # y has shape (Nbls, Nfreqs, Ntimes)
    y = data_chunk * inv_noise_var_chunk + omega_n * np.sqrt(inv_noise_var_chunk)

    # Separate complex part of RHS into real and imaginary parts, and apply
    # the real and imaginary parts of the projection operator separately.
    # Do this for each frequency separately
    b = np.zeros((freqs.size, Npix), dtype=np.float64)
    for j in range(freq_chunk.size):
        # Get frequency index of locally-held frequency channels
        i = np.where(freqs == freq_chunk[j])[0]

        # Reshape proj from (Nbls, Nfreqs, Ntimes, Npix) -> (Nbls*Ntimes, Npix)
        _proj = proj[:,j,:,:].reshape((-1, Npix))

        # Add contribution to b vector for this frequency
        b[i,:] = (  _proj.T.real @ y[:,j,:].real.flatten() \
                  + _proj.T.imag @ y[:,j,:].imag.flatten())

    # Reduce (sum) operation on b (sums over all chunks in freq and time)
    linear_rhs = np.zeros((1,1), dtype=b.dtype) # dummy data for non-root workers
    if myid == 0:
        linear_rhs = np.zeros_like(b)
    comm.Reduce(b, linear_rhs, op=MPI_SUM, root=0)

    # (Term 2): \omega_a
    if myid == 0:
        omega_s = realisation_switch * np.random.randn(*grid_shape) # real vector
        bs = apply_S(omega_s, pspec3d, exponent=-0.5)
        for i in range(b.shape[0]):
            linear_rhs[i] += bs[:,:,i].flatten() # values per pixel at each frequency

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