from pyuvdata import UVBeam
import numpy as np
from scipy.special import jn, jn_zeros
from scipy.linalg import solve, lstsq
from scipy.interpolate import interp1d
import hashlib


class sparse_beam(UVBeam):

    def __init__(
        self,
        filename,
        nmax,
        mmodes,
        za_range=(0, 90),
        save_fn="",
        load=False,
        bound="Dirichlet",
        Nfeeds=None,
        fit_beam=True,
        alpha=np.sqrt(1 - np.cos(46 * np.pi / 90)),
        num_modes_comp=64,
        nmodes_comp=None,
        mmodes_comp=None,
        sparse_fit_coeffs=None,
        perturb=False,
        za_ml=np.deg2rad(18.0),
        dza=np.deg2rad(3.0),
        Nsin_pert=8,
        sin_pert_coeffs=None,
        cSL=0.2,
        gam=None,
        sqrt=False,
        rot=0.0,
        stretch_x=1.0,
        stretch_y=1.0,
        trans_x=0.0,
        trans_y=0.0,
        **kwargs,
    ):
        """
        Construct the sparse_beam instance, which is a subclass of UVBeam

        Parameters:
            filename (str):
                The filename to for the UVBeam compatible file.
            nmax (int):
                Maximum number of radial (Bessel) modes.
            mmodes (array of int):
                Which azimuthal (Fourier) modes to include.
            za_range (tuple):
                Minimum and maximum zenith angle to read in.
            save_fn (str):
                filepath to save a numpy array of coefficients once fitting is
                complete.
            load (bool):
                Whether to load coefficients from save_fn rather than fitting
                anew.
            bound (str):
                Options are 'Dirichlet' or 'Neumann'. Refers to the boundary
                conditions for the Laplace equation in cylindrical coordinates
                i.e. it determines whether to use 0th order or 1st order
                Bessel functions.
            Nfeeds (int):
                Number of feeds to read in. This does not usually need to be set,
                but may need to be set depending on pyuvdata version.
            fit_beam (bool):
                Whether to do the FB fit after the beam is read in.
            alpha (float):
                A constant to adjust where the boundary condition is satisfied
                on the disk. Default is slightly underneath the horizon.
            num_modes_comp (int):
                Number of modes to use for the compressed representation of the 
                beam.
            nmodes_comp (array):
                List of radial mode numbers to use for the compressed beam, if
                not doing the fit from scratch (if not fit_beam).
            mmodes_comp (array):
                Azimuthal mode numbers for compressed beam corresponding to 
                radial mode numbers contained in nmodes.
            sparse_fit_coeffs (array):
                Sparse fit coefficients from previous run if fit_beam is False.
            perturb (bool):
                Whether to perturb the beam.
            za_ml (float):
                A parameter determining the size in zenith angle for 
                mainlobe-specific perturbations, not including coordinate
                transformations.
            dza (float):
                An angular scale in radians that determines how rapidly sidelobe 
                perturbations turn on after exiting the zone delineated by za_ml.
            Nsin_pert (int):
                Number of sine modes for the sidelobe perturbations.
            sin_pert_coeffs (array):
                The low order Fourier coefficients for the sidelobe perturbations.
            cSL (float):
                Parameter controlling the maximum strength of the sidelobe
                perturbations.
            gam (float):
                Parameter controlling additional mainlobe perturbations (Not)
                used in Wilensky+ 2025 but was used in Choudhuri+ 2021.
            sqrt (bool):
                Whether to take the square root of the beam before evaluating,
                interpolating, etc. This is useful for Eish beams (power beam
                based E-field beams). 
            rot (float):
                How many radians by which to rotate the coordinate system for
                perturbed beams.
            stretch_x (float):
                Factor by which to stretch the az=0 direction.
            stretch_y (float):
                Factor by which to stretch the az=pi/2 direction.
            trans_x (float):
                Radians by which to translate along the az=0 direction 
                (tilts the beam).
            trans_y (float):
                Radians by which to translate along the az=pi/2 direction
                (tilts the beam).
            kwargs:
                Additional kwargs to pass to UVBeam.read_beamfits
        """
        super().__init__()
        self.bound = bound
        self.read_beamfits(filename, za_range=za_range, **kwargs)
        self.peak_normalize()
        self.perturb = perturb
        if perturb:
            self.rot = rot
            self.stretch_x = stretch_x
            self.stretch_y = stretch_y
            self.trans_x = trans_x
            self.trans_y = trans_y
            self.za_ml = za_ml
            self.dza = dza
            self.Nsin_pert = Nsin_pert
            self.gam = gam
            self.sin_pert_coeffs = sin_pert_coeffs
            self.cSL = cSL

        if Nfeeds is not None:  # power beam may not have the Nfeeds set
            assert self.Nfeeds is None, "Nfeeds already set on the beam"
            self.Nfeeds = Nfeeds

        if sqrt:
            self.data_array = np.sqrt(self.data_array)

        self.alpha = alpha

        self.save_fn = save_fn
        self.nmax = nmax
        self.mmodes = mmodes

        self.az_array = self.axis1_array
        self.rad_array = self.get_rad_array()

        if fit_beam:
            self.az_grid, self.rad_grid = np.meshgrid(self.az_array, self.rad_array)
            self.ncoord = self.az_grid.size
            self.daz = self.axis1_array[1] - self.axis1_array[0]
            self.drad = self.rad_array[1] - self.rad_array[0]
            self.dA = self.rad_grid * self.drad * self.daz

            self.ncoeff_bess = self.nmax * len(self.mmodes)
            self.bess_matr, self.trig_matr = self.get_dmatr()
            self.bess_fits, self.bess_beam = self.get_fits(load=load)
            self.bess_ps = np.abs(self.bess_fits) ** 2

            self.num_modes_comp = num_modes_comp
            self.nmodes_comp, self.mmodes_comp = self.get_comp_inds()
            self.comp_fits, self.comp_beam = self.sparse_fit_loop()

        elif sparse_fit_coeffs is None:
            raise ValueError(
                "Must either set fit_beam=True or supply " "sparse_fit_coeffs"
            )
        else:
            self.comp_fits = sparse_fit_coeffs
            self.num_modes_comp = self.comp_fits.shape[-1]
            if (nmodes_comp is None) or (mmodes_comp is None):
                raise ValueError(
                    "Sparse fit coeffs supplied without "
                    "corresponding nmodes or mmodes. Check "
                    "sparse_nomdes and sparse_mmodes kwargs."
                )
            else:
                self.nmodes_comp = nmodes_comp
                self.mmodes_comp = mmodes_comp

        # Cache dicts for repeated interpolation
        self.az_array_dict = {}
        self.za_array_dict = {}
        self.trig_matr_interp_dict = {}
        self.bess_matr_interp_dict = {}
        self.bt_matr_interp_dict = {}

    def get_rad_array(self, za_array=None):
        """
        Get the radial coordinates corresponding to the zenith angles in
        za_array, calculated according to the formula in Hydra Beam Paper I.

        Parameters:
            za_array (array):
                The zenith angles in question.

        Returns:
            rad_array (array):
                The radial coordinates corresponding to the zenith angles.
        """
        if za_array is None:
            za_array = self.axis2_array
        rad_array = np.sqrt(1 - np.cos(za_array)) / self.alpha

        return rad_array

    def get_bzeros(self):
        """
        Get the zeros of the appropriate Bessel function based on the
        desired basis specified by the 'bound' attribute, along with the
        associated normalization.

        Returns:
            zeros (array):
                The zeros of the appropriate Bessel function
            norm (array):
                The normalization for the Bessel functions so that their L2 norm
                on the unit disc is 1.

        """
        if self.bound == "Dirichlet":
            zeros = jn_zeros(0, self.nmax)
            norm = jn(1, zeros)
        else:
            zeros = jn_zeros(1, self.nmax - 1)
            norm = jn(2, zeros)

            zeros = np.append(0, zeros)
            norm = np.append(1, norm)
        norm = norm / np.sqrt(2)

        return zeros, norm

    def get_dmatr(self):
        """
        Compute the factored design matrix that maps from Fourier-Bessel
        coefficients to pixel centers on the sky. Assumes az/za coordinates,
        AND uniform sampling in azimuth. Full design matrix is the tensor
        product of these two factors.

        Returns:
            bess_matr (array, complex):
                Has shape (Nza, Nn). Contains the radial information of the
                design matrix.
            trig_matr (array, complex):
                Has shape (Naz, Nm). Contains the azimuthal information of the
                design matrix.
        """
        zeros, norm = self.get_bzeros()

        Naz = len(self.az_array)

        bess_matr = jn(0, zeros[np.newaxis] * self.rad_array[:, np.newaxis]) / norm
        # Assume a regular az spacing and just make a unitary DFT matrix; better for fitting later
        trig_matr = np.exp(
            1.0j * np.array(self.mmodes)[np.newaxis] * self.az_array[:, np.newaxis]
        ) / np.sqrt(Naz)

        return bess_matr, trig_matr

    def get_dmatr_interp(self, az_array, za_array):
        """
        Get a design matrix specialized for interpolation rather than fitting.

        Parameters:
            az_array (array):
                Azimuth angles to evaluate bassis functions at. Does not have to
                be on a uniform grid, unlike the fitting design matrix. Should
                be 1-dimensional (i.e. flattened).
            za_array (array):
                Zenith angles to evaluate basis functions at. Should be
                1-dimensional (i.e. flattened)

        Returns:
            bess_matr (array):
                The Bessel part of the design matrix.
            trig_matr (array, complex):
                The Fourier part of the design matrix.
        """

        rad_array = self.get_rad_array(za_array)
        if self.perturb:
            if self.rot > 0:
                az_array = az_array - self.rot
            if (self.stretch_x != 1) or (self.stretch_y != 1):
                if self.stretch_x == self.stretch_y:
                    rad_array /= self.stretch_x
                else:
                    rad_array *= np.sqrt(
                        (np.cos(az_array) / self.stretch_x) ** 2
                        + (np.sin(az_array) / self.stretch_y) ** 2
                    )
            if self.trans_x != 0 or self.trans_y != 0:
                xtrans = rad_array * np.cos(az_array) - self.trans_x
                ytrans = rad_array * np.sin(az_array) - self.trans_y
                rad_array = np.sqrt(xtrans**2 + ytrans**2)
                az_array = np.arctan2(ytrans, xtrans)

        zeros, norm = self.get_bzeros()
        Naz = len(self.az_array)

        bess_matr = jn(0, zeros[np.newaxis] * rad_array[:, np.newaxis]) / norm
        if self.perturb:
            bess_matr *= self.SL_pert(rad_array=rad_array)[:, None]
        # Need to use the same normalization as in the dmatr used for fitting
        trig_matr = np.exp(
            1.0j * np.array(self.mmodes)[np.newaxis] * az_array[:, np.newaxis]
        ) / np.sqrt(Naz)

        return bess_matr, trig_matr

    def get_fits(self, load=False, data_array=None):
        """
        Compute Fourier-Bessel fits up to nmax and for all m-modes.

        Parameters:
            load (bool):
                Whether to load precomputed solutions
            data_array (array, complex):
                A data array to be fit. If None, just use the data_array attribute.
                Useful for fitting the result of a perturbed beam using the
                original basis function.

        Returns:
            fit_coeffs (array, complex):
                The coefficients for the Fourier-Bessel fit. Has shape
                (nmax, len(mmodes), Naxes_vec, 1, Nfeeds, Nfreqs)
            fit_beam (array, complex):
                The fit beam in sky coordinates. Has shape
                (Naxes_vec, 1, Nfeeds, Nfreqs, Nza, Naz)
        """

        if load:
            fit_coeffs = np.load(f"{self.save_fn}_bess_fit_coeffs.npy")
            fit_beam = np.load(f"{self.save_fn}_bess_fit_beam.npy")
        else:
            if data_array is None:
                data_array = self.data_array
            # az_modes are discretely orthonormal so just project onto the basis
            # Saves loads of memory and time
            az_fit = (
                data_array @ self.trig_matr.conj()
            )  # Naxes_vec, Nfeeds, Nfreq, Nza, Nm

            BtB = self.bess_matr.T @ self.bess_matr
            Baz = self.bess_matr.T @ az_fit  # Naxes_vec, Nfeeds, Nfreq, Nn, Nm
            Baz = Baz.transpose(3, 4, 0, 1, 2)  # Nn, Nm, Naxes_vec, Nfeeds, Nfreq

            fit_coeffs = solve(
                BtB, Baz, assume_a="sym"
            )  # Nn, Nm, Naxes_vec, Nfeeds, Nfreq

            # Apply design matrices to get fit beams
            fit_beam_az = np.tensordot(
                self.trig_matr, fit_coeffs, axes=((1,), (1,))
            )  # Naz, Nn, Naxes_vec, Nfeeds, Nfreq
            fit_beam = np.tensordot(
                self.bess_matr, fit_beam_az, axes=((1,), (1,))
            )  # Nza, Naz, Naxes_vec, Nfeeds, Nfreq
            fit_beam = fit_beam.transpose(2, 3, 4, 0, 1)

            np.save(f"{self.save_fn}_bess_fit_coeffs.npy", fit_coeffs)
            np.save(f"{self.save_fn}_bess_fit_beam.npy", fit_beam)

        return fit_coeffs, fit_beam

    def get_comp_inds(self, make_const_in_freq=True):
        """
        Get the indices for the self.num_modes most significant modes for each
        feed, polarization, and frequency.

        Parameters:
            make_const_in_freq (bool):
                Whether to use a frequency-dependent basis or not. Will use the
                best basis for the middle of the observing band, which could
                cause significant differences for wide bands.

        Returns:
            nmodes_comp (array, int):
                The radial mode numbers corresponding to the top num_modes
                Fourier-Bessl modes, in descending order of significance. Has
                shape (num_modes, Naxes_vec, 1, Nfeeds, Nfreqs).
            mmodes_comp (array, int):
                The azimuthal modes numbers corresponding to the top num_modes
                Fourier-Bessel modes, in descending order of significance. Has
                shape (num_modes, Naxes_vec, 1, Nfeeds, Nfreqs).
        """

        ps_sort_inds = np.argsort(
            self.bess_ps.reshape(
                (self.ncoeff_bess, self.Naxes_vec, self.Nfeeds, self.Nfreqs)
            ),
            axis=0,
        )
        # Highest modes start from the end
        sort_inds_flip = np.flip(ps_sort_inds, axis=0)[: self.num_modes_comp]
        nmodes_comp, mmodes_comp = np.unravel_index(
            sort_inds_flip, (self.nmax, len(self.mmodes))
        )
        if make_const_in_freq:
            mid_freq_ind = self.Nfreqs // 2
            nmodes_comp = np.repeat(
                nmodes_comp[:, :, :, mid_freq_ind : mid_freq_ind + 1],
                self.Nfreqs,
                axis=3,
            )
            mmodes_comp = np.repeat(
                mmodes_comp[:, :, :, mid_freq_ind : mid_freq_ind + 1],
                self.Nfreqs,
                axis=3,
            )

        return nmodes_comp, mmodes_comp

    def sparse_fit_loop(
        self,
        fit_beam=True,
        bess_matr=None,
        trig_matr=None,
        fit_coeffs=None,
        freq_array=None,
        data_array=None,
    ):
        """
        Do a loop over all the axes and fit _or_ evaluate fit in position space,
        using only the radial and azimuthal modes determined by the nmodes_comp
        and mmodes_comp attributes.

        Parameters:
            fit_beam (bool):
                Whether to do the fit (set to False if fit_coeffs supplied).
            bess_matr (array):
                Bessel part of design matrix.
            trig_matr (array, complex):
                Fourier part of design matrix.
            fit_coeffs (array, complex):
                Precomputed fit coefficients (if just evaluating).
            freq_array (array):
                The frequencies for the fit/evaluation.
            data_array (array):
                An array to be fit. If not provided, use the data_array attribute.

        Returns:
            fit_coeffs (array, complex; if fit_beam is True):
                The newly calculated fit coefficients in the sparse basis.
                The units are in the same units as the supplied beam since the 
                basis functions are dimensionless. 
            fit_beam (array, complex):
                The sparsely fit beam evaluated in position space.
        """
        # nmodes might vary from pol to pol, freq to freq. The fit is fast, just do a big for loop.
        interp_kwargs = [bess_matr, trig_matr, fit_coeffs]
        if fit_beam:
            fit_coeffs = np.zeros(
                [self.Naxes_vec, self.Nfeeds, self.Nfreqs, self.num_modes_comp],
                dtype=complex,
            )
            if data_array is None:
                data_array = self.data_array
            beam_shape = data_array.shape
            bess_matr = self.bess_matr
            trig_matr = self.trig_matr
        elif any([item is None for item in interp_kwargs[:2]]):
            raise ValueError(
                "Must supply bess_matr, and trig_matr " "if not doing fit."
            )
        else:
            Npos = bess_matr.shape[0]
            beam_shape = (self.Naxes_vec, self.Nfeeds, self.Nfreqs, Npos)
            if fit_coeffs is None:  # already have some fits
                fit_coeffs = self.comp_fits
        fit_beam = np.zeros(beam_shape, dtype=complex)

        Nfreqs = self.Nfreqs if freq_array is None else len(freq_array)

        for vec_ind in range(self.Naxes_vec):
            for feed_ind in range(self.Nfeeds):
                for freq_ind in range(Nfreqs):
                    if fit_beam:
                        dat_iter = data_array[vec_ind, feed_ind, freq_ind]
                    nmodes_iter = self.nmodes_comp[:, vec_ind, feed_ind, freq_ind]
                    mmodes_iter = self.mmodes_comp[:, vec_ind, feed_ind, freq_ind]
                    unique_mmodes_iter = np.unique(mmodes_iter)

                    for mmode in unique_mmodes_iter:
                        mmode_inds = mmodes_iter == mmode
                        # Get the nmodes that this mmode is used for
                        nmodes_mmode = nmodes_iter[mmode_inds]

                        bess_matr_mmode = bess_matr[:, nmodes_mmode]
                        trig_mode = trig_matr[:, mmode]

                        if fit_beam:
                            az_fit_mmode = dat_iter @ trig_mode.conj()  # Nza

                            fit_coeffs_mmode = lstsq(bess_matr_mmode, az_fit_mmode)[0]
                            fit_coeffs[vec_ind, feed_ind, freq_ind, mmode_inds] = (
                                fit_coeffs_mmode
                            )
                            fit_beam[vec_ind, feed_ind, freq_ind] += np.outer(
                                bess_matr_mmode @ fit_coeffs_mmode, trig_mode
                            )
                        else:
                            fit_coeffs_mmode = fit_coeffs[
                                vec_ind, feed_ind, freq_ind, mmode_inds
                            ]
                            fit_beam[vec_ind, feed_ind, freq_ind] += (
                                bess_matr_mmode @ fit_coeffs_mmode
                            ) * trig_mode

        if fit_beam:
            return fit_coeffs, fit_beam
        else:
            return fit_beam

    def interp(
        self,
        sparse_fit=False,
        fit_coeffs=None,
        az_array=None,
        za_array=None,
        reuse_spline=False,
        freq_array=None,
        freq_interp_kind="cubic",
        **kwargs,
    ):
        """
        A very paired down override of UVBeam.interp that more resembles
        pyuvsim.AnalyticBeam.interp. Any kwarg for UVBeam.interp that is not
        explicitly listed in this version of interp will do nothing.

        Parameters:
            sparse_fit (bool):
                Whether a sparse fit is being supplied. If False (default), just
                uses the full fit specified at instantiation.
            fit_coeffs (bool):
                The sparse fit coefficients being supplied if sparse_fit is
                True.
            az_array (array):
                Flattened azimuth angles to interpolate to.
            za_array (array):
                Flattened zenith angles to interpolate to.
            reuse_spline (bool):
                Whether to reuse the spatial design matrix for a particular
                az_array and za_array (named to keep consistency with UVBeam).
            freq_array (array):
                Frequencies to interpolate to. If None (default), just computes
                the beam at all frequencies in self.freq_array.
            freq_interp_kind (str or int):
                Type of frequency interpolation function to use. Default is a
                cubic spline. See scipy.interpolate.interp1d 'kind' keyword
                documentation for other options.

        Returns:
            beam_vals (array, complex):
                The values of the beam at the interpolated
                frequencies/spatial positions. Has shape
                (Naxes_vec, 1, Npols, Nfreqs, Npos).
        """
        if az_array is None and za_array is None and freq_array is not None:
            # vis_cpu wants to get a new object with frequency freq_array. No can do. Return the whole thing.
            # FIXME: Can make a new_sparse_beam_from_self method to accomplish this
            return self

        if az_array is None:
            raise ValueError("Must specify an azimuth array.")
        if za_array is None:
            raise ValueError("Must specify a zenith-angle array.")

        if reuse_spline:
            az_hash = hashlib.sha1(az_array).hexdigest()
            za_hash = hashlib.sha1(za_array).hexdigest()
            if (az_hash in self.az_array_dict) and (za_hash in self.za_array_dict):
                trig_matr = self.trig_matr_interp_dict[az_hash]
                bess_matr = self.bess_matr_interp_dict[za_hash]
                bt_matr = self.bt_matr_interp_dict[(az_hash, za_hash)]
            else:
                self.az_array_dict[az_hash] = az_array
                self.za_array_dict[za_hash] = za_array

                bess_matr, trig_matr = self.get_dmatr_interp(az_array, za_array)
                bt_matr = trig_matr[:, np.newaxis] * bess_matr[:, :, np.newaxis]

                self.trig_matr_interp_dict[az_hash] = trig_matr
                self.bess_matr_interp_dict[za_hash] = bess_matr
                self.bt_matr_interp_dict[(az_hash, za_hash)] = bt_matr
        else:
            bess_matr, trig_matr = self.get_dmatr_interp(az_array, za_array)
            bt_matr = trig_matr[:, np.newaxis] * bess_matr[:, :, np.newaxis]

        if sparse_fit:
            if freq_array is None:
                fit_coeffs = self.comp_fits
            else:
                for ind_ob in [self.nmodes_comp, self.mmodes_comp]:
                    if not np.all(ind_ob == ind_ob[:, :, :, :1]):
                        raise NotImplementedError(
                            "Basis is not constant in "
                            "frequency. Cannot do "
                            "frequency interpolation for "
                            "sparse_fit=True"
                        )
                freq_array, freq_array_knots = self.prep_freq_array_for_interp(
                    freq_array
                )

                fit_coeffs_interp = interp1d(freq_array_knots, self.comp_fits, axis=2)
                fit_coeffs = fit_coeffs_interp(freq_array)
            beam_vals = self.sparse_fit_loop(
                fit_beam=False,
                fit_coeffs=fit_coeffs,
                bess_matr=bess_matr,
                trig_matr=trig_matr,
                freq_array=freq_array,
            )
        else:
            if freq_array is None:
                bess_fits = self.bess_fits
            else:
                freq_array, freq_array_knots = self.prep_freq_array_for_interp(
                    freq_array
                )
                bess_fits_interp = interp1d(
                    freq_array_knots, self.bess_fits, axis=4, kind=freq_interp_kind
                )
                bess_fits = bess_fits_interp(freq_array)
            #snm, nmpxf -> pxfs (Naxes_vec, Nfreed, Nfreqs, Nsource_pos)
            beam_vals = np.tensordot(bt_matr, bess_fits, axes=2).transpose(
                1, 2, 3, 0
            )
        if self.beam_type == "power":
            # FIXME: This assumes you are reading in a power beam and is just to get rid of the imaginary component
            beam_vals = np.abs(beam_vals)

        return beam_vals, None

    def prep_freq_array_for_interp(self, freq_array):
        """
        A helper function that prepares the frequency array for 
        interpolation.

        Parameters:
            freq_array (array):
                The frequency array within which to interpolate.
        Returns:
            freq_array (array):
                The input freq_array, potentially with its array shape modified.
            freq_array_knots (array):
                The knows for the freq spline.
        """
        freq_array = np.atleast_1d(freq_array)
        assert freq_array.ndim == 1, "Freq array for interp must be exactly 1d"

        # FIXME: More explicit and complete future_array_shapes compatibility throughout code base desired
        if self.freq_array.ndim > 1:
            freq_array_knots = self.freq_array[0]
        else:
            freq_array_knots = self.freq_array
        return freq_array, freq_array_knots

    def clear_cache(self):
        """
        Clear the interpolation cache (it can take a lot of memory).
        """
        self.az_array_dict.clear()
        self.za_array_dict.clear()
        self.trig_matr_interp_dict.clear()
        self.bess_matr_interp_dict.clear()
        self.bt_matr_interp_dict.clear()

        return

    def efield_to_power(*args, **kwargs):
        raise NotImplementedError("efield_to_power is not implemented yet.")

    def efield_to_pstokes(*args, **kwargs):
        raise NotImplementedError("efield_to_pstokes is not implemented yet.")

    def sigmoid_mod(self, rad_array=None):
        """
        Make the tanh-based modulator for the perturbed beam that separates
        main lobe from sidelobe perturbations.

        Parameters:
            rad_array (array):
                An array containing the radial coordinates. Supply if perturbing
                the coordinate system). Otherwise just uses the rad_array attribute.

        Reutrns:
            sigmoid (array):
                The sigmoidal function evaluated at the radial coordinates.
        """
        if rad_array is None:
            rad_array = self.rad_array
        za_array = np.arccos(1 - (self.alpha * rad_array) ** 2)
        return 0.5 * (1 + np.tanh((za_array - self.za_ml) / self.dza))

    def sin_perts(self, rad_array=None):
        """
        Generate the Fourier based sidelobe perturbations.

        Parameters:
            rad_array (array):
                An array containing the radial coordinates. Supply if perturbing
                the coordinate system). Otherwise just uses the rad_array attribute.

        Reutrns:
            sin_perts (array):
                The Fourier series evaluated at the radial coordinates.
        """
        if rad_array is None:
            rad_array = self.rad_array
        L = self.rad_array[
            -1
        ]  # Always make this zero-out at the horizon in unstretched coordinates
        dmatr = np.array(
            [np.sin(2 * np.pi * m * rad_array / L) for m in range(1, self.Nsin_pert + 1)]
        ).T
        sin_pert_unnorm = dmatr @ self.sin_pert_coeffs
        sp_range = np.amax(sin_pert_unnorm) - np.amin(sin_pert_unnorm)
        return sin_pert_unnorm / sp_range

    def SL_pert(self, rad_array=None):
        """
        Combine all the sidelobe perturbations.

        Parameters:
            rad_array (array):
                An array containing the radial coordinates. Supply if perturbing
                the coordinate system). Otherwise just uses the rad_array attribute.
        Returns:
            SL_pert (array):
                The perturbations evaluated at the radial coordinate.
        """
        if self.sin_pert_coeffs is None:
            return np.ones_like(rad_array)
        else:
            return 1 + self.cSL * self.sin_perts(
                rad_array=rad_array
            ) * self.sigmoid_mod(rad_array=rad_array)

    def ML_gauss_term(self, gam):
        """
        A contributing term to a mainlobe perturbation that essentially changes 
        the width of the main lobe.

        Parameters:
            gam (float):
                A scale factor that adjusts the width of the perturbation.
        Returns:
            gauss_term (float):
                The gaussian term evaluated at the _unperturbed_ radial coordinates.
        """
        return np.exp(-0.5 * self.axis2_array**2 / (gam * self.za_ml) ** 2)

    def ML_pert(self):
        """
        Generate mainlobe perturbations based on the gaussian term from
        ML_gauss_term and the sigmoidal term.

        Returns:
            ML_pert (array):
                The mainlobe perturbation evaluated at the _unperturbed_ radial
                coordinates.
        """
        sig_factor = 1 - self.sigmoid_mod()
        gauss_diff = self.ML_gauss_term(gam=self.gam) - self.ML_gauss_term(gam=1)
        return sig_factor * gauss_diff
