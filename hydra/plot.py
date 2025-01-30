def plot_beam_cross(beam_coeffs, ant_ind, iter, output_dir, tag="", type="cross"):
    # Shape ncoeffs, Nfreqs, Nant -- just use a ref freq
    coeff_use = beam_coeffs[:, 0, :]
    Nants = coeff_use.shape[1]
    if type == "cross":
        fig, ax = plt.subplots(
            figsize=(16, 9),
            nrows=Nants,
            ncols=Nants,
            subplot_kw={"projection": "polar"},
        )
        for ant_ind1 in range(Nants):

            beam_use1 = bess_matr_fit @ coeff_use[:, ant_ind1, 0, 0]
            for ant_ind2 in range(Nants):
                beam_use2 = bess_matr_fit @ (coeff_use[:, ant_ind2, 0, 0])
                beam_cross = beam_use1 * beam_use2.conj()
                if ant_ind1 >= ant_ind2:
                    ax[ant_ind1, ant_ind2].pcolormesh(
                        PHI, RHO, np.abs(beam_cross), vmin=0, vmax=1
                    )
                else:
                    ax[ant_ind1, ant_ind2].pcolormesh(
                        PHI,
                        RHO,
                        np.angle(beam_cross),
                        vmin=-np.pi,
                        vmax=np.pi,
                        cmap="twilight",
                    )
    else:
        fig, ax = plt.subplots(ncols=2, subplot_kw={"projection": "polar"})
        beam_use = bess_matr_fit @ (coeff_use[:, ant_ind, 0, 0])
        ax[0].pcolormesh(PHI, RHO, np.abs(beam_use), vmin=0, vmax=1)
        ax[1].pcolormesh(
            PHI, RHO, np.angle(beam_use), vmin=-np.pi, vmax=np.pi, cmap="twilight"
        )

    fig.savefig(f"{output_dir}/beam_plot_ant_{ant_ind}_iter_{iter}_{type}_{tag}.png")
    plt.close(fig)
    return
