
from mpi4py import MPI
from mpi4py.util.dtlib import from_numpy_dtype
import numpy as np
import pyuvsim
from hydra.ptsrc_sampler import calc_proj_operator, precompute_op, construct_rhs
from hydra.utils import build_hex_array, get_flux_from_ptsrc_amp
import time

comm = MPI.COMM_WORLD
nworkers = comm.Get_size()
myid = comm.Get_rank()


def freqs_times_for_worker(myid, freqs, times, fchunks, tchunks=1):
    """
    Get the unique frequency and time chunk for a worker.
    """
    assert myid < fchunks * tchunks, "There are more workers than time and frequency chunks"

    # Get chunks for this worker
    allidxs = np.arange(fchunks * tchunks).reshape((fchunks, tchunks))
    fidx, tidx = np.where(allidxs == myid)
    fidx, tidx = int(fidx[0]), int(tidx[0])
    freq_idxs = np.arange(freqs.size)
    time_idxs = np.arange(times.size)
    freq_idx_chunk = np.array_split(freq_idxs, fchunks)[fidx]
    time_idx_chunk = np.array_split(time_idxs, tchunks)[tidx]
    return freq_idx_chunk, time_idx_chunk


def precompute_mpi(comm, ra, dec, fluxes, 
                   ant_pos, antpairs, 
                   freqs, times, beams, 
                   inv_noise_var,
                   resid, 
                   amp_prior_std, 
                   fchunks, 
                   tchunks=1, 
                   realisation=True):
    """
    Precompute the projection operator and matrix operator in parallel.
    """
    # Get frequency/time indices for this worker
    freq_idxs, time_idxs = freqs_times_for_worker(myid, freqs=freqs, times=times, 
                                                  fchunks=fchunks, tchunks=tchunks)
    freq_chunk = freqs[freq_idxs]
    time_chunk = times[time_idxs]

    # (1) Calculate projection operator for this worker
    proj = calc_proj_operator(
              ra=ra, 
              dec=dec, 
              fluxes=fluxes[:,freq_idxs], 
              ant_pos=ant_pos, antpairs=antpairs, 
              freqs=freq_chunk, 
              times=time_chunk, 
              beams=beams,
              multiprocess=False
    )

    # (2) Precompute linear system operator
    nsrcs = proj.shape[-1]
    my_linear_op = np.zeros((nsrcs, nsrcs), dtype=proj.real.dtype)

    # inv_noise_var has shape (Nbls, Nfreqs, Ntimes)
    inv_noise_var_chunk = inv_noise_var[:, freq_idxs, :][:, :, time_idxs]
    v = proj * np.sqrt(inv_noise_var_chunk[...,np.newaxis])

    # Treat real and imaginary separately, and get copies, to massively
    # speed-up the matrix multiplication!
    v_re = v.reshape((-1, nsrcs)).real.copy()
    v_im = v.reshape((-1, nsrcs)).imag.copy()
    my_linear_op[:,:] = v_re.T @ v_re + v_im.T @ v_im

    # Do Reduce (sum) operation to get total operator on root node
    linear_op = np.zeros((1,1), dtype=my_linear_op.dtype) # dummy data for non-root workers
    if myid == 0:
        linear_op = np.zeros_like(my_linear_op)
    
    comm.Reduce(my_linear_op,
                linear_op,
                op=MPI.SUM,
                root=0)
    
    # (3) Calculate linear system RHS
    proj = proj.reshape((-1, nsrcs))
    resid_chunk = resid[:, freq_idxs, :][:, :, time_idxs]

    # Switch to turn random realisations on or off
    realisation_switch = 1.0 if realisation else 0.0

    # (Terms 1+3): S^1/2 A^\dagger [ N^{-1} r + N^{-1/2} \omega_r ]
    omega_n = (
        realisation_switch
        * (1.0 * np.random.randn(*resid_chunk.shape) + 1.0j * np.random.randn(*resid_chunk.shape))
        / np.sqrt(2.0)
    )

    # Separate complex part of RHS into real and imaginary parts, and apply
    # the real and imaginary parts of the projection operator separately.
    # This is necessary to get a real RHS vector
    y = ((resid_chunk * inv_noise_var_chunk) + (omega_n * np.sqrt(inv_noise_var_chunk))).flatten()
    b = amp_prior_std * (proj.T.real @ y.real + proj.T.imag @ y.imag)

    # Reduce (sum) operation on b
    linear_rhs = np.zeros((1,), dtype=b.dtype) # dummy data for non-root workers
    if myid == 0:
        linear_rhs = np.zeros_like(b)
    comm.Reduce(b, linear_rhs, op=MPI.SUM, root=0)

    # (Term 2): \omega_a
    if myid == 0:
        linear_rhs += realisation_switch * np.random.randn(nsrcs) # real vector

    return linear_op, linear_rhs



#-------------------------------------------------------------------------------
# (1) Simulate some data
#-------------------------------------------------------------------------------

np.random.seed(5)
lst_min, lst_max, Ntimes = 0.1, 0.3, 28
freq_min, freq_max, Nfreqs = 100., 120., 24
Nptsrc = 100
ra_low, ra_high = 0., 0.2
dec_low, dec_high = np.deg2rad(-34.), np.deg2rad(-28.)
hex_array = (6, 7)

# Simulate some data
times = np.linspace(lst_min, lst_max, Ntimes)
freqs = np.linspace(freq_min, freq_max, Nfreqs)
ant_pos = build_hex_array(hex_spec=hex_array, d=14.6)
ants = np.array(list(ant_pos.keys()))
Nants = len(ants)

antpairs = []
for i in range(len(ants)):
    for j in range(i, len(ants)):
        if i != j:
            # Exclude autos
            antpairs.append((i,j))
ants1, ants2 = list(zip(*antpairs))

# Generate random point source locations
# RA goes from [0, 2 pi] and Dec from [-pi / 2, +pi / 2].
ra = np.random.uniform(low=ra_low, high=ra_high, size=Nptsrc)

# Inversion sample to get them uniform on the sphere, in case wide bounds are used
U = np.random.uniform(low=0, high=1, size=Nptsrc)
dsin = np.sin(dec_high) - np.sin(dec_low)
dec = np.arcsin(U * dsin + np.sin(dec_low)) # np.arcsin returns on [-pi / 2, +pi / 2]

# Generate fluxes
beta_ptsrc = -2.7
ptsrc_amps = 10.**np.random.uniform(low=-1., high=2., size=Nptsrc)
fluxes = get_flux_from_ptsrc_amp(ptsrc_amps, freqs, beta_ptsrc)

amp_prior_std = 0.1 * np.ones(ptsrc_amps.size)

# Beams
beams = [pyuvsim.analyticbeam.AnalyticBeam('gaussian', diameter=14.)
         for ant in ants]

# Noise variance
inv_noise_var = np.ones((len(antpairs), freqs.size, times.size))
resid = np.random.randn(*inv_noise_var.shape) / np.sqrt(inv_noise_var)


#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
# (2a) Normal way
#-------------------------------------------------------------------------------

if myid == 0:
    print("Starting serial way")
    t0 = time.time()

    # Calculate projection operator for this worker
    proj = calc_proj_operator(
              ra=ra, 
              dec=dec, 
              fluxes=fluxes, 
              ant_pos=ant_pos, antpairs=antpairs, 
              freqs=freqs, 
              times=times, 
              beams=beams,
              multiprocess=False
    )

    linear_op = precompute_op(proj, inv_noise_var)
    linear_rhs = construct_rhs(resid, inv_noise_var, amp_prior_std, proj, realisation=False)
    print("Serial took: %6.3f sec" % (time.time() - t0))
    print("Serial way:", linear_op.shape, np.sum(linear_op), np.sum(linear_rhs))
comm.barrier()

#-------------------------------------------------------------------------------
# (2b) MPI way
#-------------------------------------------------------------------------------
t0 = time.time()

linear_op_mpi, linear_rhs_mpi = precompute_mpi(
                                           comm, 
                                           ra, 
                                           dec, 
                                           fluxes, 
                                           ant_pos, 
                                           antpairs, 
                                           freqs, 
                                           times, 
                                           beams, 
                                           inv_noise_var, 
                                           resid, 
                                           amp_prior_std,
                                           fchunks=7, 
                                           tchunks=2,
                                           realisation=False)
if myid == 0:
    print("MPI took: %6.3f sec" % (time.time() - t0))
    print("MPI way:   ", linear_op_mpi.shape, np.sum(linear_op_mpi), np.sum(linear_rhs_mpi))

comm.barrier()