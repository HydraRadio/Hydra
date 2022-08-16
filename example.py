#!/usr/bin/env python

import numpy as np
import pylab as plt
import hydra

import numpy.fft as fft
from scipy.sparse.linalg import cg, LinearOperator
from scipy.signal import blackmanharris
import pyuvsim
import time, os
from hydra.vis_utils import flatten_vector, reconstruct_vector


np.random.seed(12)


SAMPLE_GAINS = True
SAMPLE_VIS = True
SAMPLE_PTSRC_AMPS = True
OUTPUT_DIAGNOSTICS = True

# Simulation settings
Nptsrc = 70
Ntimes = 30
Nfreqs = 20
Nants = 15
Niters = 250

sigma_noise = 0.5

hera_latitude = -30.7215 * np.pi / 180.0

# Check that output directory exists
if not os.path.exists("./output"):
    os.makedirs("./output")

#-------------------------------------------------------------------------------
# (1) Simulate some data
#-------------------------------------------------------------------------------

# Simulate some data
times = np.linspace(0.2, 0.5, Ntimes)
freqs = np.linspace(100., 120., Nfreqs)

ants = np.arange(Nants)
antpairs = []
for i in range(len(ants)):
    for j in range(i, len(ants)):
        if i != j:
            # Exclude autos
            antpairs.append((i,j))
ant_pos = {ant: [14.7*(ant % 5) + 0.5*14.7*(ant // 5),
                 14.7*(ant // 5),
                 0.] for ant in ants} # hexagon-like packing
ants1, ants2 = list(zip(*antpairs))

# Generate random point source locations
# RA goes from [0, 2 pi] and Dec from [-pi, +pi].
ra = np.random.uniform(low=0.0, high=1.8, size=Nptsrc)
dec = np.random.uniform(low=-0.6, high=-0.4, size=Nptsrc) # close to HERA stripe

# Generate fluxes
beta_ptsrc = -2.7
ptsrc_amps = 10.**np.random.uniform(low=-1., high=2., size=Nptsrc)
fluxes = ptsrc_amps[:,np.newaxis] * ((freqs / 100.)**beta_ptsrc)[np.newaxis,:]
print("pstrc amps (input):", ptsrc_amps[:5])

# Beams
beams = [pyuvsim.analyticbeam.AnalyticBeam('gaussian', diameter=14.)
         for ant in ants]

# Run a simulation
t0 = time.time()
_sim_vis = hydra.vis_simulator.simulate_vis(
        ants=ant_pos,
        fluxes=fluxes,
        ra=ra,
        dec=dec,
        freqs=freqs*1e6, # MHz -> Hz
        lsts=times,
        beams=beams,
        polarized=False,
        precision=2,
        latitude=hera_latitude,
        use_feed="x",
    )
print("Simulation took %3.2f sec" % (time.time() - t0))

# Allocate computed visibilities to only the requested baselines (saves memory)
model0 = hydra.extract_vis_from_sim(ants, antpairs, _sim_vis)
del _sim_vis # save some memory

# Define gains and gain perturbations
gains = (1. + 0.j) * np.ones((Nants, Nfreqs, Ntimes), dtype=model0.dtype)
#delta_g = np.array([0.5*np.sin(times[np.newaxis,:] \
#                                 * freqs[:,np.newaxis]/100.)
#                    for ant in ants])

# Generate gain fluctuations from FFT basis
frate = np.fft.fftfreq(times.size, d=times[1] - times[0])
tau = np.fft.fftfreq(freqs.size, d=freqs[1] - freqs[0])
_delta_g = 50. \
         * np.fft.ifft2(np.exp(-0.5 * (((frate[np.newaxis,:]-9.)/2.)**2.
                                     + ((tau[:,np.newaxis] - 0.05)/0.03)**2.)))
delta_g = np.array([_delta_g for ant in ants], dtype=model0.dtype)

# FIXME
# Apply a Blackman-Harris window to apodise the edges
window = blackmanharris(model0.shape[1], sym=True)[np.newaxis,:,np.newaxis] \
       * blackmanharris(model0.shape[2], sym=True)[np.newaxis,np.newaxis,:]
window = 1.

# Apply gains to model
data = model0.copy() * window # FIXME
hydra.apply_gains(data, gains * (1. + delta_g), ants, antpairs, inline=True)

# Plot input (simulated) gain perturbation for ant 0
vminmax = np.max(delta_g[0].real)
plt.matshow(delta_g[0].real, vmin=-vminmax, vmax=vminmax)
plt.colorbar()
plt.savefig("output/gain_000_xxxxx.png")

# Plot input (simulated) visibility model
vminmax_vis = np.max(model0[0,:,:].real)
plt.matshow(model0[0,:,:].real, vmin=-vminmax_vis, vmax=vminmax_vis)
plt.colorbar()
plt.savefig("output/data_model_000_xxxxx.png")

# Add noise
data += sigma_noise * np.sqrt(0.5) \
      * (  1.0 * np.random.randn(*data.shape) \
         + 1.j * np.random.randn(*data.shape))

# Plot data (including noise)
plt.matshow(data[0,:,:].real, vmin=-vminmax_vis, vmax=vminmax_vis)
plt.colorbar()
plt.savefig("output/data_000.png")

#-------------------------------------------------------------------------------
# (2) Set up Gibbs sampler
#-------------------------------------------------------------------------------

# Get initial visibility model guesses (use the actual baseline model for now)
# This SHOULD NOT include gain factors of any kind
current_data_model = 1.01*model0.copy() * window # FIXME

# Initial gain perturbation guesses
#current_delta_gain = np.zeros(gains.shape, dtype=model0.dtype)
# FIXME
current_delta_gain = np.zeros_like(delta_g)

# Initial point source amplitude factor
current_ptsrc_a = np.ones(ra.size)

# Precompute visibility projection operator (without gain factors) for ptsrc
# amplitude sampling step. NOTE: This has to be updated within the Gibbs loop
# if other components of the visibility model are being sampled
t0 = time.time()
vis_proj_operator0 = hydra.ptsrc_sampler.calc_proj_operator(
                                ra=ra,
                                dec=dec,
                                fluxes=fluxes,
                                ant_pos=ant_pos,
                                antpairs=antpairs,
                                freqs=freqs,
                                times=times,
                                beams=beams,
                                latitude=hera_latitude
)
print("Precomp. ptsrc proj. operator took %3.2f sec" % (time.time() - t0))

# Set priors and auxiliary information
# FIXME: amp_prior_std is a prior around amp=0 I think, so can skew things low!
noise_var = (sigma_noise)**2. * np.ones(data.shape)
inv_noise_var = window / noise_var

gain_pspec_sqrt = 0.1 * np.ones((gains.shape[1], gains.shape[2]))
gain_pspec_sqrt[0,0] = 1e-3 # FIXME: Try to fix the zero point?

amp_prior_std = 0.1 * np.ones(Nptsrc)
amp_prior_std[19] = 1e-4 # FIXME
vis_pspec_sqrt = 0.01 * np.ones((1, Nfreqs, Ntimes)) # currently same for all visibilities
vis_group_id = np.zeros(len(antpairs), dtype=int) # index 0 for all

A_real, A_imag = hydra.gain_sampler.proj_operator(ants, antpairs)

gain_shape = gains.shape
N_gain_params = 2 * gains.shape[0] * gains.shape[1] * gains.shape[2]
N_vis_params = 2 * data.shape[0] * data.shape[1] * data.shape[2]


# FIXME: Check that model0 == vis_proj_operator0 @ ones






"""

# FIXME: Testing projection operator for gains

delta_g_fourier = np.array([fft.fft2(_delta_g) for ant in ants], dtype=model0.dtype)

ggv = hydra.apply_gains(np.ones_like(model0),
                        gains,
                        ants,
                        antpairs,
                        inline=False)

v_obs_f = hydra.gain_sampler.apply_proj(delta_g_fourier,
                                      A_real,
                                      A_imag,
                                      ggv)
v_obs = np.array([fft.ifft2(v_obs_f[i]) for i in range(v_obs_f.shape[0])])
v_obs += ggv

dd = np.ones_like(model0, dtype=np.complex128)
hydra.apply_gains(dd, gains * (1. + delta_g), ants, antpairs, inline=True)

print("*"*60)
print(dd.dtype, gains.dtype, delta_g.dtype)
#print(dd[0])


plt.close('all')
plt.subplot(331)
plt.matshow(dd[0].real, fignum=False, vmin=-1.8, vmax=1.8, cmap='Spectral')
plt.colorbar()
plt.title("real gains real")

plt.subplot(332)
plt.matshow(dd[0].imag, fignum=False, vmin=-1.8, vmax=1.8, cmap='Spectral')
plt.colorbar()
plt.title("real gains imag")

plt.subplot(333)
plt.matshow(np.abs(dd[0]), fignum=False, vmin=-1.8, vmax=1.8, cmap='Spectral')
plt.colorbar()
plt.title("real gains abs")


plt.subplot(334)
plt.matshow(v_obs[0].real, fignum=False, vmin=-1.8, vmax=1.8, cmap='Spectral')
plt.colorbar()
plt.title("proj gains real")

plt.subplot(335)
plt.matshow(v_obs[0].imag, fignum=False, vmin=-1.8, vmax=1.8, cmap='Spectral')
plt.colorbar()
plt.title("proj gains imag")

plt.subplot(336)
plt.matshow(np.abs(v_obs[0]), fignum=False, vmin=-1.8, vmax=1.8, cmap='Spectral')
plt.colorbar()
plt.title("proj gains abs")


lin_gains = gains[0]*gains[1].conj()*(1. + delta_g[0] + delta_g[1])
plt.subplot(337)
plt.matshow(lin_gains.real, fignum=False, vmin=-1.8, vmax=1.8, cmap='Spectral')
plt.colorbar()
plt.title("linear gains real")

plt.subplot(338)
plt.matshow(lin_gains.imag, fignum=False, vmin=-1.8, vmax=1.8, cmap='Spectral')
plt.colorbar()
plt.title("linear gains imag")

plt.subplot(339)
plt.matshow(np.abs(lin_gains), fignum=False, vmin=-1.8, vmax=1.8, cmap='Spectral')
plt.colorbar()
plt.title("linear gains abs")


print("@"*60)
print((delta_g[0]*delta_g[1].conj())[0] / (delta_g[0])[0])

plt.gcf().set_size_inches((14., 10.))
plt.show()


exit()
"""








#-------------------------------------------------------------------------------
# (3) Iterate Gibbs sampler
#-------------------------------------------------------------------------------

# Iterate the Gibbs sampler
print("="*60)
print("Starting Gibbs sampler (%d iterations)" % Niters)
print("="*60)
for n in range(Niters):
    print("-"*60)
    print(">>> Iteration %4d / %4d" % (n+1, Niters))
    print("-"*60)

    #---------------------------------------------------------------------------
    # (A) Gain sampler
    #---------------------------------------------------------------------------
    if SAMPLE_GAINS:
        # Current_data_model DOES NOT include gbar_i gbar_j^* factor, so we need
        # to apply it here to calculate the residual
        ggv = hydra.apply_gains(current_data_model,
                                gains,
                                ants,
                                antpairs,
                                inline=False)
        resid = data - ggv

        b = hydra.gain_sampler.flatten_vector(
                hydra.gain_sampler.construct_rhs(resid,
                                                 inv_noise_var,
                                                 gain_pspec_sqrt,
                                                 A_real,
                                                 A_imag,
                                                 ggv,
                                                 realisation=True)
            )

        def gain_lhs_operator(x):
            return hydra.gain_sampler.flatten_vector(
                        hydra.gain_sampler.apply_operator(
                            hydra.gain_sampler.reconstruct_vector(x, gain_shape),
                                             inv_noise_var,
                                             gain_pspec_sqrt,
                                             A_real,
                                             A_imag,
                                             ggv)
                                 )

        # Build linear operator object
        gain_lhs_shape = (N_gain_params, N_gain_params)
        gain_linear_op = LinearOperator(matvec=gain_lhs_operator,
                                        shape=gain_lhs_shape)

        # Solve using Conjugate Gradients
        t0 = time.time()
        x_soln, convergence_info = cg(gain_linear_op, b)
        print("(A) Gain sampler took %3.2f sec" % (time.time() - t0))

        # Reshape solution into complex array and multiply by S^1/2 to get set of
        # Fourier coeffs of the actual solution for the frac. gain perturbation
        x_soln = hydra.vis_utils.reconstruct_vector(x_soln, gain_shape)
        x_soln = hydra.gain_sampler.apply_sqrt_pspec(gain_pspec_sqrt, x_soln)

        # x_soln is a set of Fourier coefficients, so transform to real space
        # (ifft gives Fourier -> data space)
        xgain = np.zeros_like(x_soln)
        for k in range(xgain.shape[0]):
            xgain[k, :, :] = fft.ifft2(x_soln[k, :, :])

        print("    Gain sample:", xgain[0,0,0], xgain.shape)
        np.save("output/delta_gain_%05d" % n, x_soln)
        plt.matshow(xgain[0].real, vmin=-vminmax, vmax=vminmax)
        plt.colorbar()
        plt.title("%05d" % n)
        plt.savefig("output/gain_000_%05d.png" % n)

        # Residual with true gains (abs)
        plt.matshow(np.abs(xgain[0]) - np.abs(delta_g[0]),
                    vmin=-np.max(np.abs(delta_g[0])),
                    vmax=np.max(np.abs(delta_g[0])))
        plt.colorbar()
        plt.title("%05d" % n)
        plt.savefig("output/gain_resid_amp_000_%05d.png" % n)

        # Residual with true gains (real)
        plt.matshow(np.real(xgain[0]) - np.real(delta_g[0]),
                    vmin=-np.max(np.real(delta_g[0])),
                    vmax=np.max(np.real(delta_g[0])))
        plt.colorbar()
        plt.title("%05d" % n)
        plt.savefig("output/gain_resid_real_000_%05d.png" % n)

        # Residual with true gains (imag)
        plt.matshow(np.imag(xgain[0]) - np.imag(delta_g[0]),
                    vmin=-np.max(np.imag(delta_g[0])),
                    vmax=np.max(np.imag(delta_g[0])))
        plt.colorbar()
        plt.title("%05d" % n)
        plt.savefig("output/gain_resid_imag_000_%05d.png" % n)

        # DEBUG
        # Compare imaginary parts of gains
        plt.subplot(121)
        plt.matshow(np.imag(xgain[0]),
                    vmin=-np.max(np.imag(delta_g[0])),
                    vmax=np.max(np.imag(delta_g[0])), fignum=False)
        plt.colorbar()

        plt.subplot(122)
        plt.matshow(np.imag(delta_g[0]),
                    vmin=-np.max(np.imag(delta_g[0])),
                    vmax=np.max(np.imag(delta_g[0])), fignum=False)
        plt.colorbar()
        plt.title("%05d" % n)
        plt.savefig("output/gain_resid_compare_imag_000_%05d.png" % n)

        # Update gain model with latest solution (in real space)
        current_delta_gain = xgain


    #---------------------------------------------------------------------------
    # (B) Visibility sampler
    #---------------------------------------------------------------------------
    if SAMPLE_VIS:
        # Current_data_model DOES NOT include gbar_i gbar_j^* factor, so we need
        # to apply it here to calculate the residual
        ggv = hydra.apply_gains(current_data_model,
                                gains*(1.+current_delta_gain),
                                ants,
                                antpairs,
                                inline=False)
        resid = data - ggv

        # Construct RHS of linear system
        bvis = flatten_vector(
                    hydra.vis_sampler.construct_rhs(
                                               data=resid,
                                               inv_noise_var=inv_noise_var,
                                               sqrt_pspec=vis_pspec_sqrt,
                                               group_id=vis_group_id,
                                               gains=gains*(1.+current_delta_gain),
                                               ants=ants,
                                               antpairs=antpairs,
                                               realisation=True)
                                               )

        vis_lhs_shape = (N_vis_params, N_vis_params)
        def vis_lhs_operator(x):
            # Re-pack flattened x vector into complex vector of the right shape
            y = hydra.vis_sampler.apply_operator(reconstruct_vector(x, data.shape),
                                                 inv_noise_var=inv_noise_var,
                                                 sqrt_pspec=vis_pspec_sqrt,
                                                 group_id=vis_group_id,
                                                 gains=gains*(1.+current_delta_gain),
                                                 ants=ants,
                                                 antpairs=antpairs)
            return flatten_vector(y)

        # Build linear operator object
        vis_linear_op = LinearOperator(matvec=vis_lhs_operator,
                                       shape=vis_lhs_shape)

        # Solve using Conjugate Gradients
        t0 = time.time()
        x_soln, convergence_info = cg(vis_linear_op, bvis)
        print("(B) Visibility sampler took %3.2f sec" % (time.time() - t0))

        # Reshape solution into complex array and multiply by S^1/2 to get set of
        # Fourier coeffs of the actual solution for the frac. gain perturbation
        x_soln = hydra.vis_utils.reconstruct_vector(x_soln, data.shape)
        x_soln = hydra.vis_sampler.apply_sqrt_pspec(vis_pspec_sqrt,
                                                    x_soln,
                                                    vis_group_id,
                                                    ifft=True)

        print("    Vis sample:", x_soln[0,0,0], x_soln.shape)
        np.save("output/vis_%05d" % n, x_soln)
        plt.matshow(current_data_model[0].real + x_soln[0].real,
                    vmin=-vminmax_vis,
                    vmax=vminmax_vis)
        plt.colorbar()
        plt.title("%05d" % n)
        plt.savefig("output/vis_000_%05d.png" % n)

        # Update current state
        current_data_model = current_data_model + x_soln

        # Residual with true data model
        plt.subplot(121)
        plt.matshow(current_data_model[0].real - model0[0].real,
                    vmin=-0.5,
                    vmax=0.5,
                    fignum=False, aspect='auto')
        plt.colorbar()
        plt.title("%05d" % n)

        plt.subplot(122)
        plt.matshow(current_data_model[0].imag - model0[0].imag,
                    vmin=-0.5,
                    vmax=0.5,
                    fignum=False, aspect='auto')
        plt.colorbar()
        plt.gcf().set_size_inches((10., 4.))
        plt.savefig("output/resid_datamodel_000_%05d.png" % n)


    #---------------------------------------------------------------------------
    # (C) Point source amplitude sampler
    #---------------------------------------------------------------------------

    if SAMPLE_PTSRC_AMPS:

        # Get the projection operator with most recent gains applied
        t0 = time.time()
        proj = vis_proj_operator0.copy()
        for k, bl in enumerate(antpairs):
            ant1, ant2 = bl
            i1 = np.where(ants == ant1)[0][0]
            i2 = np.where(ants == ant2)[0][0]
            proj[k,:,:,:] *= (gains * (1. + current_delta_gain))[i1,:,:,np.newaxis] \
                           * (gains * (1. + current_delta_gain))[i2,:,:,np.newaxis].conj()
        print("(C) Applying gains to ptsrc proj. operator took %3.2f sec" \
              % (time.time() - t0))

        # Precompute the point source matrix operator
        t0 = time.time()
        ptsrc_precomp_mat = hydra.ptsrc_sampler.precompute_op(proj, noise_var)
        print("(C) Precomp. ptsrc matrix operator took %3.2f sec" \
              % (time.time() - t0))

        # Construct current state of model (residual from amplitudes = 1)
        resid = data.copy() \
              - ( proj.reshape((-1, Nptsrc)) @ np.ones_like(amp_prior_std) ).reshape(current_data_model.shape)

        # Construct RHS of linear system
        bsrc = hydra.ptsrc_sampler.construct_rhs(resid.flatten(),
                                                 noise_var.flatten(),
                                                 amp_prior_std,
                                                 proj,
                                                 realisation=True)

        ptsrc_lhs_shape = (Nptsrc, Nptsrc)
        def ptsrc_lhs_operator(x):
            return hydra.ptsrc_sampler.apply_operator(x,
                                                      noise_var.flatten(),
                                                      amp_prior_std,
                                                      ptsrc_precomp_mat)

        # Build linear operator object
        ptsrc_linear_op = LinearOperator(matvec=ptsrc_lhs_operator,
                                         shape=ptsrc_lhs_shape)

        # Solve using Conjugate Gradients
        t0 = time.time()
        x_soln, convergence_info = cg(ptsrc_linear_op, bsrc)
        print("(C) Point source sampler took %3.2f sec" % (time.time() - t0))
        x_soln *= amp_prior_std # we solved for x = S^-1/2 s, so recover s
        print("    Example soln:", x_soln[:5]) # this is fractional deviation from assumed amplitude, so should be close to 0
        np.save("output/ptsrc_amp_%05d" % n, x_soln)

        # Plot point source amplitude perturbations
        plt.subplot(111)
        plt.plot(x_soln, 'r.')
        plt.axhline(0., ls='dashed', color='k')
        plt.axhline(-np.max(amp_prior_std), ls='dotted', color='gray')
        plt.axhline(np.max(amp_prior_std), ls='dotted', color='gray')
        plt.ylim((-1., 1.))
        plt.title("%05d" % n)
        plt.savefig("output/ptsrc_amp_%05d.png" % n)

        # Update visibility model with latest solution (does not include any gains)
        # Applies projection operator to ptsrc amplitude vector
        current_data_model = ( vis_proj_operator0.reshape((-1, Nptsrc)) @ (1. + x_soln) ).reshape(current_data_model.shape)

        # Plot visibility waterfalls for current model
        plt.matshow(current_data_model[0,:,:].real, vmin=-vminmax_vis, vmax=vminmax_vis)
        plt.colorbar()
        plt.title("%05d" % n)
        plt.savefig("output/data_model_000_%05d.png" % n)

    #---------------------------------------------------------------------------
    # (O) Output diagnostics
    #---------------------------------------------------------------------------
    if OUTPUT_DIAGNOSTICS:

        # Output residual
        ggv = hydra.apply_gains(current_data_model,
                                gains*(1.+current_delta_gain),
                                ants,
                                antpairs,
                                inline=False)
        resid = data - ggv
        plt.matshow(np.abs(resid[0]), vmin=-5., vmax=5.)
        plt.colorbar()
        plt.title("%05d" % n)
        plt.savefig("output/resid_abs_000_%05d.png" % n)

        # Output chi^2
        chisq = (resid[0].real**2. + resid[0].imag**2.) / noise_var[0]
        chisq_tot = np.sum( (resid.real**2. + resid.imag**2.) / noise_var )
        plt.matshow(chisq, vmin=0., vmax=40.)
        plt.title(r"$\chi^2_{\rm tot} = %5.3e$" % chisq_tot)
        plt.colorbar()
        plt.title("%05d" % n)
        plt.savefig("output/chisq_000_%05d.png" % n)


    # Close all figures made in this iteration
    plt.close('all')
