#!/usr/bin/env python

import time, os

import numpy as np
import hydra

from scipy.stats import norm, rayleigh
from scipy.linalg import cholesky
from hydra.utils import timing_info, build_hex_array, get_flux_from_ptsrc_amp, \
                         convert_to_tops
from pyuvdata.analytic_beam import GaussianBeam, AiryBeam
from pyuvdata import UVBeam, BeamInterface

import argparse
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines

import beam_example_utils

if __name__ == '__main__':

    description = "Example Gibbs sampling of the joint posterior of per-antenna beam "  \
                  "parameters from a simulated visibility data set " 
    parser = beam_example_utils.get_parser(description)
    args, output_dir = beam_example_utils.setup_args_dirs(parser)
    
    array_lat, ant_pos, Nants = beam_example_utils.get_array_params(args)
    times, freqs = beam_example_utils.get_obs_params(args)
    ra, dec, beta_ptsrc, ptsrc_amps, fluxes = beam_example_utils.get_src_params(args, output_dir)
    
    beams = []
    for ant_ind in range(Nants):
        ref_cond = args.perts_only and ant_ind == 0
        if args.beam_type == "pert_sim":
            pow_sb = beam_example_utils.perturbed_beam(
                args, 
                output_dir, 
                seed=args.beam_seed + ant_ind
            )        
            beams.append(pow_sb)
            if ref_cond:
                ref_beam = UVBeam.from_file(args.beam_file)
                ref_beam.peak_normalize()
        elif args.beam_type in ["gaussian", "airy"]:
            # Underilluminated HERA dishes
            beam_rng = np.random.default_rng(seed=args.beam_seed + ant_ind)
            beam, beam_class = beam_example_utils.get_analytic_beam(args, beam_rng)
            beams.append(beam)
            if ref_cond:
                ref_beam = beam_class(diameter=12.)
        else:
            raise ValueError("beam-type arg must be one of ('gaussian', 'airy', 'pert_sim')")
    
    chain_seed, ftime, unpert_sb = beam_example_utils.init_prebeam_simulation_items(
        args, 
        output_dir,
        freqs
    )


    flux_inference, unpert_vis, data, inv_noise_var = beam_example_utils.vis_sim_wrapper(
        args, 
        output_dir, 
        array_lat, 
        ant_pos, 
        Nants, 
        times, 
        freqs, 
        ra, 
        dec, 
        beta_ptsrc, 
        ptsrc_amps, 
        fluxes, 
        beams, 
        ref_beam, 
        ftime
    )


    bess_matr, trig_matr, nmodes, mmodes, per_source_Dmatr_out, za, az = beam_example_utils.prep_beam_Dmatr_items(args, output_dir, array_lat, times, ra, dec, unpert_sb)
    if not os.path.exists(per_source_Dmatr_out):
        if args.beam_type == "pert_sim": # Use PCA-based basis for sqrt(power beam) in a per antenna way
            # Need a mode for the mean of the prior. 
            # Use the FB coeffs for the frequency closest to the center frequency of the simulation.
            mid_freq = freqs[args.Nfreqs // 2]
            closest_chan = np.argmin(np.abs(mid_freq - pow_sb.freq_array))
            mean_mode = unpert_sb.bess_fits[:, :, 0, 0, 0, closest_chan]
            # Normalize it to have L^2 norm of 1
            mean_mode = mean_mode / np.sqrt(np.sum(np.abs(mean_mode)**2))
            # Load the PCA modes saved on disk ahead of time -- they are expressed in FB space
            pca_modes = np.load(args.pca_modes)[:, :args.Nbasis - 1].reshape(args.nmax, 2 * args.mmax + 1, args.Nbasis - 1) 
            # Make the full basis including the mean mode
            Pmatr = np.concatenate([mean_mode[:, :, None], pca_modes], axis=2)
            # Contract with FB design matrices so that the map goes from PCA -> source coordinates
            BPmatr = np.tensordot(bess_matr, Pmatr, axes=1).transpose(3, 0, 1, 2) # Nbasis, Ntimes, Nsrc, Naz
            # Matrix evaluating subset of FB modes at source coordinates
            Dmatr = np.sum(BPmatr * trig_matr, axis=3).transpose(1, 2, 0) # Ntimes, Nsrc, Nbasis
        else: # Using analytic beams, only use radial modes
            Dmatr = bess_matr[:, :, :args.Nbasis]
            Q, R = np.linalg.qr(unpert_sb.bess_matr[:, :args.Nbasis]) # orthoganalize radial modes...
            reshape = (args.Ntimes * args.Nptsrc, args.Nbasis)
            shape = (args.Ntimes, args.Nptsrc, args.Nbasis)
            # Get the orthogonalized basis evaluated at the source coordinates
            Dmatr = np.linalg.solve(R.T, Dmatr.reshape(reshape).T).T.reshape(shape)
        np.save(per_source_Dmatr_out, Dmatr)
    else:
        Dmatr = np.load(per_source_Dmatr_out)

    
    # Have everything we need to analytically evaluate single-array beam
    if args.per_ant: 
        Dmatr_outer = hydra.beam_sampler.get_bess_outer(Dmatr)
        np.save(os.path.join(output_dir, "dmo.npy"), Dmatr_outer)
    
        beam_coeffs = np.zeros([Nants, args.Nfreqs, args.Nbasis, 1, 1])
        if not args.perts_only:
            if args.decent_prior:
                # FIXME: Hardcode!, start at answer!
                beam_coeffs += np.load("data/14m_airy_bessel_soln.npy")[None, None, :, None, None] 
            else:
                beam_coeffs[:, :, 0] = 1
            # Want shape Nbasis, Nfreqs, Nants, Npol, Npol
        beam_coeffs = np.swapaxes(beam_coeffs, 0, 2).astype(complex)

        sig_freq = 0.1 * (freqs[-1] - freqs[0])
        # FIXME: Hardcode!
        cov_file = "data/ramped_variance_bessel_cov.npy" if args.decent_prior else None
        cov_tuple = hydra.beam_sampler.make_prior_cov(freqs, args.beam_prior_std,
                                                    sig_freq, args.Nbasis,
                                                    ridge=1e-6,
                                                    cov_file=cov_file)
        cho_tuple = hydra.beam_sampler.do_cov_cho(cov_tuple, check_op=False)

        bsc_outpath = os.path.join(output_dir, "bsc.npy")
        if os.path.exists(bsc_outpath):
            bess_sky_contraction = np.load(bsc_outpath)
        else:
            t0 = time.time()
            bess_sky_contraction = hydra.beam_sampler.get_bess_sky_contraction(Dmatr_outer, 
                                                                            ant_pos, 
                                                                            fluxes, 
                                                                            ra,
                                                                            dec, 
                                                                            freqs, 
                                                                            times,
                                                                            polarized=False, 
                                                                            latitude=array_lat,)
            np.save(bsc_outpath, bess_sky_contraction)
            tsc = time.time() - t0
            timing_info(ftime, 0, "(0) bess_sky_contraction", tsc)
            print(f"bess_sky_contraction took {tsc} seconds")

        if args.perts_only:
            ref_contraction_outpath = os.path.join(output_dir, "ref_constraction.npy")
            if os.path.exists(ref_contraction_outpath):
                ref_contraction = np.load(ref_contraction_outpath)
            else:
                ref_beam_response = BeamInterface(ref_beam, beam_type="power")
                # FIXME: Hardcode feed x
                ref_beam_response = ref_beam_response.with_feeds(["x"])
                ref_beam_response = ref_beam_response.compute_response(az_array=az,
                                                                    za_array=za,
                                                                    freq_array=freqs)
                ref_beam_response = ref_beam_response.reshape([args.Nfreqs,
                                                            args.Ntimes,
                                                            args.Nptsrc,])
                # Square root of power beam
                ref_beam_response = np.sqrt(ref_beam_response)
                ref_contraction = hydra.beam_sampler.get_bess_sky_contraction(Dmatr, 
                                                                              ant_pos, 
                                                                              fluxes, 
                                                                              ra,
                                                                              dec, 
                                                                              freqs, 
                                                                              times,
                                                                              polarized=False, 
                                                                              latitude=array_lat,
                                                                              outer=False,
                                                                              ref_beam_response=ref_beam_response)

                np.save(ref_contraction_outpath, ref_contraction)
            coeff_mean = np.zeros_like(beam_coeffs[:, :, 0])
        else:
            # Be lazy and just use the initial guess -- either the right answer or 1 in the first basis function.
            coeff_mean = np.copy(beam_coeffs[:, :, 0])
        # shuffle the initial position by pulling from the prior
        if chain_seed is not None: 
            chain_rng = np.random.default_rng(seed=chain_seed)
            beam_coeffs = chain_rng.normal(loc=coeff_mean[:, :, None].real,
                                        scale=args.beam_prior_std,
                                        size=beam_coeffs.shape).astype(complex)


        # Iterate the Gibbs sampler
        print("="*60)
        print("Starting Gibbs sampler (%d iterations)" % args.Niters)
        print("="*60)
        for n in range(args.Niters):
            print("-"*60)
            print(">>> Iteration %4d / %4d" % (n+1, args.Niters))
            print("-"*60)
            t0iter = time.time()

            if args.anneal:
                temp = max(2000. - 2 * n, 1.)
            else:
                temp = 1.

            for ant_samp_ind in range(Nants):
                bess_trans = hydra.beam_sampler.get_bess_to_vis_from_contraction(bess_sky_contraction,
                                                                                 beam_coeffs, 
                                                                                 Nants, 
                                                                                 ant_samp_ind)
                
                inv_noise_var_use = hydra.beam_sampler.select_subarr(inv_noise_var[None, None], # add pol axes of length 1
                                                                     ant_samp_ind, 
                                                                     Nants)
                if args.infnoise:
                    inv_noise_var_use[:] = 0
                else:
                    inv_noise_var_use /= temp
                data_use = hydra.beam_sampler.select_subarr(data[None, None], ant_samp_ind, Nants)
                if args.perts_only:
                    # Contract other antenna coefficients with object that has been pre-multiplied by reference beam
                    other_ants_with_ref = hydra.beam_sampler.get_bess_to_vis_from_contraction(ref_contraction,
                                                                                              beam_coeffs, 
                                                                                              Nants, 
                                                                                              ant_samp_ind,
                                                                                              ref_contraction=True)
                    data_use -= other_ants_with_ref
                    ant_inds = hydra.beam_sampler.get_ant_inds(ant_samp_ind, Nants)
                    # Add the term that contracts the reference beam with current antenna's perturbations
                    bess_trans += ref_contraction[:, :, :, :, ant_inds, ant_samp_ind]

                # Construct RHS vector
                rhs_unflatten = hydra.beam_sampler.construct_rhs(data_use,
                                                                inv_noise_var_use,
                                                                coeff_mean,
                                                                bess_trans,
                                                                cov_tuple,
                                                                cho_tuple)
                bbeam = rhs_unflatten.flatten()
                    

                shape = (args.Nfreqs, args.Nbasis,  1, 1, 2)
                cov_Qdag_Ninv_Q = hydra.beam_sampler.get_cov_Qdag_Ninv_Q(inv_noise_var_use,
                                                                        bess_trans,
                                                                        cov_tuple)
                
                axlen = np.prod(shape)

                # fPbpQBcCF->fbQcFBpPC
                matr = cov_Qdag_Ninv_Q.transpose((0,2,4,6,8,5,3,1,7)).reshape([axlen, axlen]) + np.eye(axlen)
                
                # FIXME: This is skipped!
                def beam_lhs_operator(x):
                    y = hydra.beam_sampler.apply_operator(np.reshape(x, shape),
                                                        cov_Qdag_Ninv_Q)
                    return(y.flatten())

                # What the shape would be if the matrix were represented densely
                beam_lhs_shape = (axlen, axlen)
                print("Solving")
                x_soln = np.linalg.solve(matr, bbeam)
                print("Done solving")


                if args.test_close:
                    btest = matr @ x_soln
                    allclose = np.allclose(btest, bbeam)
                    if not allclose:
                        abs_diff = np.abs(btest-bbeam)
                        wh_max_diff = np.argmax(abs_diff)
                        max_diff = abs_diff[wh_max_diff]
                        max_val = bbeam[wh_max_diff]
                        raise AssertionError(f"btest not close to bbeam, max_diff: {max_diff}, max_val: {max_val}")
                x_soln_res = np.reshape(x_soln, shape)

                # Has shape Nfreqs, Nbasis, Npol, Npol, ncomp
                # Want shape Nbasis, Nfreqs, Npol, Npol, ncomp
                x_soln_swap = np.swapaxes(x_soln_res, 0, 1)

                # Update the coeffs between rounds
                beam_coeffs[:, :, ant_samp_ind] = 1.0 * x_soln_swap[:, :, :, :, 0] \
                                                    + 1.j * x_soln_swap[:, :, :, :, 1]
                
            timing_info(ftime, n, "(D) Beam sampler", time.time() - t0iter)
            np.save(os.path.join(output_dir, "beam_%05d" % n), beam_coeffs)

            np1 = n + 1
            if not np1 % args.roundup:
                print(f"Rounding up files for iteration: {np1}")
                files_to_round_up = glob.glob(f"{output_dir}/beam*.npy")
                sorted_files = sorted(files_to_round_up)
                sample_arr = np.zeros((args.roundup,) + beam_coeffs.shape)
                for file_ind, file in enumerate(sorted_files):
                    if "roundup" in file:
                        continue
                    sample_arr[file_ind] = np.load(file)
                    os.remove(file)
                np.save(os.path.join(output_dir, f"beam_roundup_{np1}"), sample_arr)

        

        
