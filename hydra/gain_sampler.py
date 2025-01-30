import numpy as np
import pylab as plt
import scipy.sparse
import numpy.fft as fft

try:
    from mpi4py.MPI import SUM as MPI_SUM
except:
    pass

from scipy.sparse import dok_matrix
from .utils import flatten_vector, reconstruct_vector, freqs_times_for_worker

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


def construct_rhs_mpi(
    comm,
    resid,
    inv_noise_var,
    pspec_sqrt,
    A_real,
    A_imag,
    model_vis,
    Fbasis,
    realisation=True,
    seed=None,
):
    """
    MPI version of RHS constructor.
    """
    myid = comm.Get_rank()

    # The random seed should depend on the worker ID to avoid duplication
    np.random.seed(seed)

    Nvis, Nfreqs, Ntimes = resid.shape
    Nmodes = Fbasis.shape[0]
    Nants = A_real.shape[-1]
    assert pspec_sqrt.shape == (Nmodes,)

    # Switch to turn random realisations on or off
    realisation_switch = 1.0 if realisation else 0.0

    # (Term 2): \omega_y
    if myid == 0:
        # Random realisation for prior
        b = (
            realisation_switch
            * (
                1.0 * np.random.randn(Nants, Nmodes)
                + 1.0j * np.random.randn(Nants, Nmodes)
            )
            / np.sqrt(2.0)
        )
    else:
        b = np.zeros((Nants, Nmodes), dtype=np.complex128)

    # Broadcast this random realisation
    comm.Bcast(b, root=0)

    # The following quantities are calculated on each worker; each worker holds
    # a chunk of the data, and the calculated quantities are reduced/summed across
    # the full set of workers at the end

    # (Terms 1+3): S^1/2 F^dagger A^\dagger [ N^{-1} r + N^{-1/2} \omega_r ]
    omega_r = (
        realisation_switch
        * (1.0 * np.random.randn(*resid.shape) + 1.0j * np.random.randn(*resid.shape))
        / np.sqrt(2.0)
    )
    gain_shape = (Nants, Nfreqs, Ntimes)

    # Apply inverse noise (or its sqrt) to data/random vector terms, and do
    # transpose projection operation, all in real space
    yy = apply_proj_conj(
        resid * inv_noise_var + omega_r * np.sqrt(inv_noise_var),
        A_real,
        A_imag,
        model_vis,
        gain_shape,
    )

    # Conjugate basis
    FFc = Fbasis.conj().reshape((Fbasis.shape[0], -1))

    # Do FT to go into Fourier space again; also apply sqrtS operator
    my_y = np.zeros_like(b)
    for k in range(Nants):
        my_y[k, :] = pspec_sqrt * np.tensordot(
            FFc, yy[k, :, :].flatten(), axes=((1,), (0,))
        )

    # Do reduce (sum) operation for all workers' my_y arrays, which contain the
    # projection of that worker's chunk of the result onto a chunk of the Fourier
    # basis. The result of the reduce/sum gives the full FT over *all* data points
    total_y = np.zeros_like(my_y)
    comm.Allreduce(my_y.flatten(), total_y, op=MPI_SUM)

    # Add the transformed Terms 1+3 to b vector
    # Result should be the same for all MPI workers
    bvec = np.concatenate(((b + total_y).flatten().real, (b + total_y).flatten().imag))
    return bvec


def apply_operator_mpi(
    comm, x, inv_noise_var, pspec_sqrt, A_real, A_imag, model_vis, Fbasis
):
    """
    MPI version of gain linear operator.
    """
    myid = comm.Get_rank()

    assert inv_noise_var.shape == model_vis.shape

    # Reshape input vector
    Nmodes, Nfreqs, Ntimes = Fbasis.shape
    Nants = A_real.shape[1]
    # vec = x.reshape((Nants, Nmodes))
    assert pspec_sqrt.shape == (Nmodes,)

    # Broadcast this input vector to all workers, to make sure it's synced
    vec = np.zeros(2 * Nants * Nmodes, dtype=x.dtype)
    if myid == 0:
        vec[:] = x
    comm.Bcast(vec, root=0)

    # Extract real and imaginary parts, reshape, and multiply by sqrt of prior var
    xre = pspec_sqrt[np.newaxis, :] * vec[: vec.size // 2].reshape((Nants, Nmodes))
    xim = pspec_sqrt[np.newaxis, :] * vec[vec.size // 2 :].reshape((Nants, Nmodes))

    # Conjugate basis
    FFc = Fbasis.conj().reshape((Fbasis.shape[0], -1))

    # The following quantities are calculated on each worker; each worker holds
    # a chunk of the data, and the calculated quantities are reduced/summed across
    # the full set of workers at the end

    # Multiply Fourier x values by S^1/2 and FT
    sqrtSx = np.zeros((Nants, Fbasis.shape[1], Fbasis.shape[2]), dtype=np.complex128)
    for k in range(sqrtSx.shape[0]):
        sqrtSx[k, :, :] = np.tensordot(
            Fbasis, xre[k, :] + 1.0j * xim[k, :], axes=((0,), (0,))
        )
    gain_shape = (Nants, Nfreqs, Ntimes)

    # Apply projection operator to real-space sqrt(S)-weighted x values,
    # weight by inverse noise variance, then apply (conjugated) projection
    # operator
    y = apply_proj_conj(
        apply_proj(sqrtSx, A_real, A_imag, model_vis) * inv_noise_var,
        A_real,
        A_imag,
        model_vis,
        gain_shape,
    )

    # Do inverse FT and multiply by S^1/2 again
    yy = np.zeros((Nants, Nmodes), dtype=np.complex128)
    for k in range(y.shape[0]):
        yy[k, :] = pspec_sqrt * np.tensordot(
            FFc, y[k, :, :].flatten(), axes=((1,), (0,))
        )

    # Expand into 1D vector of real and imag. parts
    my_y = np.concatenate((yy.real.flatten(), yy.imag.flatten()))

    # Do Allreduce/sum to sum over all chunks and distribute result to all workers
    total_y = np.zeros_like(my_y)
    comm.Allreduce(my_y.flatten(), total_y, op=MPI_SUM)

    # Add identity and return
    return x + total_y
