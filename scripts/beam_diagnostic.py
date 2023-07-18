import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import argparse
import glob

from hydra.beam_sampler import get_bess_matr

parser = argparse.ArgumentParser()
parser.add_argument("--chdir", required=True, type=str, nargs=1,
                    help="The directory where the beam chain is stored")
parser.add_argument("--burn-in", required=False, type=int, nargs=1, dest="burn_in",
                    default=5000, help="Number of samples to discard in chain")
parser.add_argument("--outdir", required=True, type=str, nargs=1,
                    help="Where to store the outputs")
parser.add_argument("--nmodes-path", required=True, type=str, nargs=1,
                    dest="nmodes_path", help="Path to array of nmodes")
parser.add_argument("--mmodes-path", required=True, type=str, nargs=1,
                    dest="mmodes_path", help="Path to array of mmodes")
parser.add_argument("--rho-const", required=False, type=float, nargs=1,
                    dest="rho_const", default=np.sqrt(1-np.cos(np.pi * 23 / 45)))
parser.add_argument("--ref-freq-ind", required=False, type=int, nargs=1,
                    dest="ref_freq_ind", default=0,
                    help="The reference frequency for the beam plot")
args = parser.parse_args()


def plot_cmatr(fig, ax, cmatr):
    shape = cmatr.shape[:(cmatr.ndim // 2)]
    siz = np.prod(shape)
    im = ax.matshow(cmatr.reshape(siz, siz))
    fig.colorbar(im, ax=ax)

    for arrax in range(-1, -len(shape), -1):
        cadence = np.prod(shape[-1:-(arrax + 1):-1])
        ax.axvline(x=[cadence * cind for cind in range(shape[arrax])],
                   color="white", linewidth=-2 * arrax)
        ax.axhline(y=[cadence * cind for cind in range(shape[arrax])],
                   color="white", linewidth=-2 * arrax)

    return

fl = glob.glob(f"{args.chdir}/beam_*.npy")

chain = []
for fn in fl[args.burn_in:]:
    chain.append(np.load(fn))
chain = np.array(chain)

# Plot some traces
for bas_ind in range(Npols):
    for feed_ind in range(Npols):
        chfig, chax = plt.subplots(figsize=(14, 8), nrows=2, ncols=Nants,
                                   subplot_kw={"projection": "polar"})
        for ant_idx in range(Nants):
            chain_use = chain[:, :, args.ref_freq_ind, ant_idx, bas_ind, feed_ind]
            chax[ant_idx].plot(chain_use.real)
            chax[ant_idx].plot(chain_use.imag)
        chfig.tight_layout()
        chfig.savefig(f"{args.outdir}/beam_chain_plot_bas_{bas_ind}_feed_{feed_ind}.png")
        plt.close(chfig)


chain_split = np.concatenate([chain.real[:, np.newaxis], chain.imag[:, np.newaxis]], axis=1)
Niters = chain.shape[0]
mean = np.mean(chain_split, axis=0)
mfig, meax = plt.subplots(figsize=(4, 4))
maex.plot(mean[0].reshape(mean.size // 2), label="Real Component")
meax.plot(mean[1].reshape(mean.size // 2), label="Imaginary Component")
meax.legend()
mfig.savefig(f"{args.outdir}/beam_post_mean.pdf")
plt.close(mfig)

cov = np.tensordot(chain_split - mean, (chain_split - mean), axes=((0,), (0,))) / (Niters - 1)
vars = np.diag(cov.reshape(mean.size, mean.size)).reshape(mean.shape)
corr_norm = np.sqrt(np.outer(corr_norm, corr_norm))

corr = cov / corr

cfig, cax = plt.subplots(figsize=(8, 4), ncols=2)
plot_cmatr(cfig, cax[0], cov)
plot_cmatr(cfig, cax[1], corr)
cfig.savefig(f"{args.outdir}/beam_post_cov.pdf")
plt.close(cfig)

mean_comp = mean[0] + 1.j * mean[1]

nmodes = np.load(args.nmodes_path)
mmodes = np.load(args.mmodes_path)
az = np.arange(0, 360) * np.pi / 180
za = np.arange(91) * np.pi / 180
rho = np.sqrt(1 - np.cos(za)) / args.rho_const
Az, Rho = np.meshgrid(az, rho)
_, Za = np.meshgrid(az, za)
bess_matr = get_bess_matr(nmodes, mmodes, Rho, Az)

# Shape ncoeff, Nfreqs, Nants, Npols, Npols -> Naz, Nrho, Nfreqs, Nants, Npols, Npols
mean_beam = np.tensordot(bess_matr, mean_comp, axes=1)
Nants = mean_comp.shape[2]
Npols = mean_comp.shape[-1]

# Plot beams
for bas_ind in range(Npols):
    for feed_ind in range(Npols):
        bcfig, bcax = plt.subplots(figsize=(14, 8), nrows=Nants, ncols=Nants,
                                 subplot_kw={"projection": "polar"})
        bfig, bax = plt.subplots(figsize=(14, 8), nrows=2, ncols=Nants,
                                 subplot_kw={"projection": "polar"})
        for ant_ind1 in range(Nants):
            for ant_ind2 in range(ant_ind1, Nants):
                beam_use1 = mean_beam[:, :, args.ref_freq_ind, ant_ind1, bas_ind, feed_ind]
                beam_use2 = mean_beam[:, :, args.ref_freq_ind, ant_ind2, bas_ind, feed_ind]

                bcax[ant_ind1, ant_ind2].pcolormesh(Az, Za, np.abs(beam_use1 * beam_use2.conj()),
                                                    norm=LogNorm())
                if ant_ind1 != ant_ind2:
                    bcax[ant_ind2, ant_ind1].pcolormesh(Az, ZA, np.angle(beam_use1 * beam_use2.conj()),
                                                        cmap="twilight")
                else:
                    bax[0, ant_ind1].pcolormesh(Az, Za, beam_use1.real,
                                                norm=SymLogNorm(linthresh=1e-3),
                                                cmap="coolwarm")
                    bax[1, ant_ind1].pcolormesh(Az, Za, beam_use1.imag,
                                                norm=SymLogNorm(linthresh=1e-3),
                                                cmap="coolwarm")

        bcfig.tight_layout()
        bfig.tight_layout()
        bcfig.savefig(f"{args.outdir}/beam_cross_by_ants_bas_{bas_ind}_feed_{feed_ind}.png")
        bfig.savefig(f"{args.outdir}/beam_by_ants_bas_{bas_ind}_feed_{feed_ind}.png")
        plt.close(bcfig)
        plt.close(bfig)
