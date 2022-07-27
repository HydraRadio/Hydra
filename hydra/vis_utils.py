
import numpy as np

def apply_gains(v, gains, ants, antpairs, inline=False):
    """
    Apply gain factors to an input array of complex visibility values.

    Parameters:
        v (array_like):
            Input visibility array to which gains will be applied. This must
            have shape (Nbaselines, Nfreqs, Ntimes). The ordering of the 0th
            (baseline) dimension is assumed to be the same ordering as
            `antpairs`.
        gain (array_like):
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
