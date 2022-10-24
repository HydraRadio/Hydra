import numpy as np
import pylab as plt
import scipy.sparse
import numpy.fft as fft

from .utils import flatten_vector, reconstruct_vector


def apply_sqrt_pspec(sqrt_pspec, v, group_id, ifft=False):
    """
    Apply the square root of the power spectrum to a set of complex Fourier
    coefficients. This is a way of implementing the operation "S^1/2 x" if S is
    diagonal, represented only by a 2D power spectrum in delay and fringe rate.

    Parameters:
        sqrt_pspec (dict of array_like):
            Dictionary of power spectra. The key would normally be the ID of
            the redundant group that a baseline belongs to.

        v (array_like):
            Array of complex Fourier coefficients, of shape (Nvis, Ntau, Nfrate).

        group_id (array_like):
            Integer group ID of each baseline.

        ifft (bool):
            Whether to apply a 2D inverse FFT to transform "S^1/2 x" to real
            space.

    Returns:
        z (array_like):
            Array of complex Fourier coefficients that have been multiplied by
            the sqrt of the power spectrum. Same shape as v.
    """
    assert group_id.size == v.shape[0], "Must have a group_id for each visibility"

    # Set up inverse FFT if requested
    if ifft:
        transform = lambda x: fft.ifft2(x)
    else:
        transform = lambda x: x

    # Loop through visibilities and apply sqrt of power spectrum
    z = v.copy()
    for i in range(z.shape[0]):
        z[i] = transform(z[i] * sqrt_pspec[group_id[i]])
    return z


def apply_operator(v, inv_noise_var, sqrt_pspec, group_id, gains, ants, antpairs):
    """
    Apply LHS operator to a vector of complex visibility Fourier coefficients.

    Parameters:
        v (array_like):
            Vector of model visibility Fourier modes to apply the operator to.
            Shape (Nvis, Ntau, Nfrate).

        inv_noise_var (array_like):
            Inverse noise variance array, with the same shape as the visibility
            data.

        sqrt_pspec (dict of array_like):
            Dictionary of power spectra. The key would normally be the ID of
            the redundant group that a baseline belongs to.

        group_id (array_like):
            Integer group ID of each baseline.

        gains (array_like):
            Complex gains, with the same ordering as `ants`. Expected shape is
            (Nants, Nfreqs, Ntimes).

        ants (array_like):
            Array of antenna IDs.

        antpairs (list of tuples):
            List of antenna pair tuples.

    Returns:
        lhs (array_like):
            Result of applying the LHS operator to the input vector, v.
    """
    assert len(v.shape) == 3, "v must have shape (Nvis, Ntau, Nfrate)"
    assert len(antpairs) == v.shape[0]

    # Apply sqrt power spectrum, inverse FFT, and divide by noise variance
    y = apply_sqrt_pspec(sqrt_pspec=sqrt_pspec,
                         v=v,
                         group_id=group_id,
                         ifft=True) * inv_noise_var

    # Multiply by gains, then FFT back
    for k, bl in enumerate(antpairs):

        # Get antenna indices
        ant1, ant2 = bl
        i1 = np.where(ants == ant1)[0][0]
        i2 = np.where(ants == ant2)[0][0]

        # Apply S^1/2 to input vector, transform to data space, multiply by
        # gain product, and divide by noise variance
        y[k, :, :] = fft.fft2(  y[k,:,:]
                              * gains[i1] * gains[i2].conj()
                              * gains[i2] * gains[i1].conj()
                              )

    # Return full application of the LHS operator
    return v + apply_sqrt_pspec(sqrt_pspec=sqrt_pspec,
                                v=y,
                                group_id=group_id,
                                ifft=False)


def construct_rhs(
    data, inv_noise_var, sqrt_pspec, group_id, gains, ants, antpairs, realisation=False
):
    """
    Construct the RHS vector of the linear system. This will have shape
    (2*Nvis), as the real and imaginary parts are separated.

    Parameters:
        data (array_like):
            Observed visibility data.

        inv_noise_var (array_like):
            Inverse noise variance array, with the same shape as the visibility
            data.

        sqrt_pspec (dict of array_like):
            Dictionary of power spectra. The key would normally be the ID of
            the redundant group that a baseline belongs to.

        group_id (array_like):
            Integer group ID of each baseline.

        gains (array_like):
            Complex gains, with the same ordering as `ants`. Expected shape is
            (Nants, Nfreqs, Ntimes).

        ants (array_like):
            Array of antenna IDs.

        antpairs (list of tuples):
            List of antenna pair tuples.

        realisation (bool):
            Whether to include the random realisation terms in the RHS
            (constrained realisation), or just the deterministic terms (Wiener
            filter).

    Returns:
        rhs (array_like):
            The RHS of the linear system.
    """
    # fft: data -> fourier
    # ifft: fourier -> data
    Nvis, Ntimes, Nfreqs = data.shape
    Nfrate = Ntimes
    Ntau = Nfreqs

    # Switch to turn random realisations on or off
    realisation_switch = 1.0 if realisation else 0.0

    # (Term 2): \omega_y
    b = (
        realisation_switch
        * (
            1.0 * np.random.randn(Nvis, Nfrate, Ntau)
            + 1.0j * np.random.randn(Nvis, Nfrate, Ntau)
        )
        / np.sqrt(2.0)
    )

    # (Terms 1+3): S^1/2 F^dagger A^\dagger [ N^{-1} r + N^{-1/2} \omega_r ]
    omega_r = (
        realisation_switch
        * (1.0 * np.random.randn(*data.shape) + 1.0j * np.random.randn(*data.shape))
        / np.sqrt(2.0)
    )

    # Loop over visibilities, weight terms 1 and 3 by N^-1 and N^-1/2, then
    # apply conjugate of gains, FFT, and apply sqrt power spectrum
    y = (data * inv_noise_var) + (omega_r * np.sqrt(inv_noise_var))
    for k, bl in enumerate(antpairs):

        # Get antenna indices
        ant1, ant2 = bl
        i1 = np.where(ants == ant1)[0][0]
        i2 = np.where(ants == ant2)[0][0]

        y[k,:,:] = fft.fft2(y[k,:,:] * gains[i1].conj() * gains[i2])

    # Apply sqrt(S) operator
    y = apply_sqrt_pspec(sqrt_pspec, y, group_id, ifft=False)

    # Add the transformed Terms 1+3 to b vector
    return b + y
