import numpy as np
import pylab as plt
import scipy.sparse
import numpy.fft as fft
#import pyfftw.interfaces.numpy_fft as fft

from scipy.sparse import dok_matrix
from .utils import flatten_vector, reconstruct_vector

r"""
# Mathematical representation of linear system for gain GCR

The visibility model can be written as

$d_{ij} \approx \bar{g}_i \bar{g}_j^* \left ( 1 + x_i + x_j^* \right ) V_{ij}$.

We will work with the residual of the data with respect to the model,

$r_{ij} \approx d_{ij} - \bar{g}_i \bar{g}_j^* V_{ij} = \bar{g}_i \bar{g}_j^* \left ( x_i + x_j^* \right ) V_{ij}$.

We can write the constrained realisation system (e.g. see arXiv:0709.1058) as

$\left ( 1 + S^\frac{1}{2} [A^\dagger N^{-1} A] S^\frac{1}{2} \right ) y = S^\frac{1}{2} A^\dagger N^{-1} r + \omega_y + S^\frac{1}{2} A^\dagger N^{-\frac{1}{2}}\omega_r$,

where $y = S^{-\frac{1}{2}}x$, so we can trivially solve to get
$x = S^\frac{1}{2}y$. Here, the $x$ vector is a set of Fourier coefficients for
the gain fluctuation terms; $S$ is the prior covariance of the gain parameters,
$S_{ij} = \langle x_i x^\dagger_j \rangle$, where $i,j$ label antennas; the
noise covariance is $N_{ab} = \sigma_{a}\delta_{ab}$, where $a,b$ label
baselines (antenna pairs), and $d$ is a vector of measured visibilities.

Each of the objects will have the following shape:

 * $d$ is a vector of $(N_{\rm bl}, N_\nu \times N_t)$ complex visibilities.
   It is a fixed quantity.
 * $N$ is a matrix of $(N_d, N_d)$ real noise variance values. Because it's
   diagonal, we can perform some simplifications.
 * $s$ is a vector of $(N_{\rm ant}, N_\tau \times N_{\rm fr})$ complex Fourier
   coefficients.
 * $S$ is a matrix of $(N_s, N_s)$ prior variance values. It can be simplified
   if there is no mode coupling.
 * $A$ is a linear operator that projects and combines the Fourier coefficients
   for each antenna into a visibility vector.
"""


def proj_operator(ants, antpairs):
    """
    Construct two sparse projection operators, one that goes from the real part
    of x_i and the other that goes from the imaginary part.

    To apply the projection operator, just do:
        dot(A, x) = A_real.dot(x.real) + 1.j*A_imag.dot(x.imag)
    """
    Nvis = len(antpairs)
    Nants = len(ants)

    # Get ant1 and ant2 from each pair
    ants1, ants2 = list(zip(*antpairs))
    A_real = dok_matrix((Nvis, Nants), dtype=int)
    A_imag = dok_matrix((Nvis, Nants), dtype=int)

    # Build dict of indices in antenna array
    ants = list(ants)
    ant_idx = {ant: ants.index(ant) for ant in ants}

    # Construct projection operator
    for i, antpair in enumerate(antpairs):
        ant1, ant2 = antpair
        A_real[i, ant_idx[ant1]] += 1  # x_i.real
        A_real[i, ant_idx[ant2]] += 1  # x_j.conj().real
        A_imag[i, ant_idx[ant1]] += 1  # x_i.imag
        A_imag[i, ant_idx[ant2]] += -1  # x_j.conj().imag
    return A_real, A_imag


def apply_proj(x, A_real, A_imag, model_vis):
    """
    Apply the projection operator to multi-dimensional arrays of gain
    fluctuations.

    dot(A, x) = A_real.dot(x.real) + 1.j*A_imag.dot(x.imag)

    Parameters:
        x (array_like):
            Array with shape of vector of gain fluctuations (real space).
            Shape (Nants, Nfreqs, Ntimes).

        A_real, A_imag (sparse array):
            Shape (Nvis, Nants).

        model_vis (array_like):
            Array of complex model visibilities. Shape (Nvis, Ntimes, Nfreqs).
    """
    Nvis, Ntimes, Nfreqs = model_vis.shape
    v = np.zeros((Nvis, Ntimes, Nfreqs), dtype=np.complex128)
    v[:, :, :] = (
        1.0 * A_real.dot(x.real.reshape((x.shape[0], -1)))
        + 1.0j * A_imag.dot(x.imag.reshape((x.shape[0], -1)))
    ).reshape((-1, x.shape[1], x.shape[2]))
    v *= model_vis  # Apply \bar{g}_i \bar{g}_j V_ij factor
    return v


def apply_proj_conj(v, A_real, A_imag, model_vis, gain_shape):
    """
    Apply the conjugate transpose of the projection operator to
    multi-dimensional arrays.

    dot(A, x) = A_real.dot(x.real) + 1.j*A_imag.dot(x.imag)

    Parameters:
        v (array_like):
            Shape (Nvis, Nfreqs, Ntimes)

        A_real, A_imag (sparse array):
            Shape (Nvis, Nants)

        model_vis (array_like):
            Array of complex model visibilities. Shape (Nvis, Ntimes, Nfreqs).

        gain_shape (tuple):
            Expected shape of the gain model.
    """
    Nants, Nfrate, Ntau = gain_shape
    g = np.zeros((Nants, Nfrate, Ntau), dtype=np.complex128)

    # Apply conjugate transpose of model visibility,
    # (\bar{g}_i \bar{g}_j V_ij)^\dagger
    vv = v * model_vis.conj()

    # If we split the visibilities etc. into real and imaginary parts, the
    # A operator is block diagonal, and so its transpose is also block diagonal
    g[:, :, :] = (
        1.0 * A_real.T.dot(vv.real.reshape((vv.shape[0], -1)))
        + 1.0j * A_imag.T.dot(vv.imag.reshape((vv.shape[0], -1)))
    ).reshape((-1, vv.shape[1], vv.shape[2]))
    return g


def apply_sqrt_pspec(sqrt_pspec, x):
    """
    Apply the square root of the power spectrum to a set of complex Fourier
    coefficients. This is a way of implementing the operation "S^1/2 x" if S is
    diagonal, represented only by a 2D power spectrum.

    Parameters:
        sqrt_pspec (array_like):
            If given as a 3D array, apply a different power spectrum to each antenna.
            Otherwise, apply the same (2D) power spectrum to each antenna. The
            power spectrum should have shape (Ntau, Nfrate).

        x (array_like):
            Array of complex Fourier coefficients.

    Returns:
        z (array_like):
            Array of complex Fourier coefficients that have been multiplied by
            the sqrt of the power spectrum. Same shape as x.
    """
    assert len(x.shape) == 3, "x must have shape (Nants, Ntau, Nfrate)"
    if len(sqrt_pspec.shape) == 3:
        return sqrt_pspec * x
    else:
        return sqrt_pspec[np.newaxis, :, :] * x


def apply_operator(x, inv_noise_var, pspec_sqrt, A_real, A_imag, model_vis):
    r"""
    Apply LHS operator to a vector of Fourier coefficients.

    Parameters:
        x (array_like):
            Array of complex values in Fourier space, of shape
            (Nants, Nfrate, Ntau).

        inv_noise_var (array_like):
            Array of real values in visibility space, with the inverse noise
            variance per baseline, time, and frequency. Shape
            (Nvis, Ntimes, Nfreqs).

        pspec_sqrt (array_like):
            Array of (real) power spectrum values, square-rooted, modelling
            the prior covariance of the gain perturbations.
            Shape (Nants, Nfrate, Ntau).

        A_real, A_imag (sparse array):
            Sparse arrays that project from a vector of gain perturbations to
            a vector of visibilities that they belong to. Shape (Nvis, Nants).

        model_vis (array_like):
            Array of complex visibility model values, of shape (Nvis, Ntimes,
            Nfreqs).
            $m_{ij} = \bar{g}_i \bar{g}_j^\dagger V_{ij}$.
    """
    gain_shape = x.shape
    vis_shape = inv_noise_var.shape
    assert inv_noise_var.shape == model_vis.shape

    # Multiply Fourier x values by S^1/2 and FT
    sqrtSx = apply_sqrt_pspec(pspec_sqrt, x)
    for k in range(sqrtSx.shape[0]):
        sqrtSx[k, :, :] = fft.ifft2(sqrtSx[k, :, :])

    # Apply projection operator to real-space sqrt(S)-weighted x values,
    # weight by inverse noise variance, then apply (conjugated) projection
    # operator
    y = apply_proj_conj(apply_proj(sqrtSx,
                                   A_real,
                                   A_imag,
                                   model_vis) \
                          * inv_noise_var,
                        A_real,
                        A_imag,
                        model_vis,
                        gain_shape)

    # Do inverse FT and multiply by S^1/2 again
    for k in range(y.shape[0]):
        y[k, :, :] = fft.fft2(y[k, :, :])
    return x + apply_sqrt_pspec(pspec_sqrt, y)


def construct_rhs(
    resid, inv_noise_var, pspec_sqrt, A_real, A_imag, model_vis, realisation=True
):
    """
    Construct the RHS vector of the linear system. This will have shape
    (Nants, Ntau, Nfrate).

    Parameters:
        resid (array_like, complex):
            Residual of the observed visibilities (data minus fiducial input
            model).

        inv_noise_var (array_like):
            Inverse noise variance, of shape (Nbl, Nfreqs, Ntimes). This is
            used because we have assumed that the noise is diagonal
            (uncorrelated) between baselines, times, and frequencies.

        pspec_sqrt (array_like):
            Array of 2D (sqrt) power spectra used to construct the prior
            covariance S.

        A_real, A_imag (sparse array):
            Shape (Nvis, Nants)

        model_vis (array_like):
            Array of complex model visibilities. Shape (Nvis, Ntimes, Nfreqs).

        realisation (bool):
            Whether to include Gaussian random realisation terms (True) or just
            the Wiener filter terms (False).
    """
    # fft: data -> fourier
    # ifft: fourier -> data
    Nvis, Ntimes, Nfreqs = resid.shape
    Nants = A_real.shape[-1]
    Nfrate = Ntimes
    Ntau = Nfreqs

    # Switch to turn random realisations on or off
    realisation_switch = 1.0 if realisation else 0.0

    # (Term 2): \omega_y
    b = (
        realisation_switch
        * (
            1.0 * np.random.randn(Nants, Nfrate, Ntau)
            + 1.0j * np.random.randn(Nants, Nfrate, Ntau)
        )
        / np.sqrt(2.0)
    )

    # (Terms 1+3): S^1/2 F^dagger A^\dagger [ N^{-1} r + N^{-1/2} \omega_r ]
    omega_r = (
        realisation_switch
        * (1.0 * np.random.randn(*resid.shape) + 1.0j * np.random.randn(*resid.shape))
        / np.sqrt(2.0)
    )
    gain_shape = (Nants, Nfrate, Ntau)

    # Apply inverse noise (or its sqrt) to data/random vector terms, and do
    # transpose projection operation, all in real space
    yy = apply_proj_conj(
        resid * inv_noise_var + omega_r * np.sqrt(inv_noise_var),
        A_real,
        A_imag,
        model_vis,
        gain_shape,
    )

    # Do FT to go into Fourier space again
    for k in range(Nants):
        yy[k,:,:] = fft.fft2(yy[k,:,:])

    # Apply sqrt(S) operator
    yy = apply_sqrt_pspec(pspec_sqrt, yy)

    # Add the transformed Terms 1+3 to b vector
    return b + yy
