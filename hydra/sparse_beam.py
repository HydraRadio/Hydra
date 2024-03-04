from pyuvdata import UVBeam
import numpy as np
from scipy.special import jn, jn_zeros
from scipy.linalg import solve, lstsq
from scipy.interpolate import interp1d
import hashlib


class sparse_beam(UVBeam):
    
    def __init__(self, filename, nmax, mmodes, za_range=(0, 90), save_fn='', 
                 load=False, bound="Dirichlet", 
                 alpha=np.sqrt(1 - np.cos(46 * np.pi / 90)), **kwargs):
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
            alpha (float):
                A constant to adjust where the boundary condition is satisfied
                on the disk. Default is slightly underneath the horizon.
        """
        super().__init__()
        self.bound = bound
        self.read_beamfits(filename, za_range=za_range, **kwargs)
        self.peak_normalize()
        
        self.az_array = self.axis1_array
        self.alpha = alpha
        self.rad_array = self.get_rad_array()
        
        self.az_grid, self.rad_grid = np.meshgrid(self.az_array, self.rad_array)
        self.ncoord = self.az_grid.size
        
        self.nmax = nmax
        self.mmodes = mmodes
        self.ncoeff_bess = self.nmax * len(self.mmodes)
        
        self.bess_matr, self.trig_matr = self.get_dmatr()

        
        self.daz = self.axis1_array[1] - self.axis1_array[0]
        self.drad = self.rad_array[1] - self.rad_array[0]
        self.dA = self.rad_grid * self.drad * self.daz
        
        self.save_fn = save_fn
        self.bess_fits, self.bess_beam = self.get_fits(load=load)
        self.bess_ps = np.abs(self.bess_fits)**2

        self.az_array_dict = {}
        self.za_array_dict = {}
        self.trig_matr_interp_dict = {}
        self.bess_matr_interp_dict = {}

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
        trig_matr = np.exp(1.0j * np.array(self.mmodes)[np.newaxis] * self.az_array[:, np.newaxis]) / np.sqrt(Naz)
        
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
        zeros, norm = self.get_bzeros()
        Naz = len(self.az_array)

        bess_matr = jn(0, zeros[np.newaxis] * rad_array[:, np.newaxis]) / norm
        # Need to use the same normalization as in the dmatr used for fitting
        trig_matr = np.exp(1.j *  np.array(self.mmodes)[np.newaxis] * az_array[:, np.newaxis]) / np.sqrt(Naz)

        return bess_matr, trig_matr
    
    
    def get_fits(self, load=False):
        """
        Compute Fourier-Bessel fits up to nmax and for all m-modes.

        Parameters:
            load (bool): 
                Whether to load precomputed solutions

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
            # az_modes are discretely orthonormal so just project onto the basis
            # Saves loads of memory and time
            az_fit = self.data_array @ self.trig_matr.conj() # Naxes_vec, 1, Nfeeds, Nfreq, Nza, Nm

            BtB = self.bess_matr.T @ self.bess_matr
            Baz = self.bess_matr.T @ az_fit # Naxes_vec, 1, Nfeeds, Nfreq, Nn, Nm
            Baz = Baz.transpose(4, 5, 0, 1, 2, 3) # Nn, Nm, Naxes_vec, 1, Nfeeds, Nfreq


            fit_coeffs = solve(BtB, Baz, assume_a="sym") # Nn, Nm, Naxes_vec, 1, Nfeeds, Nfreq

            # Apply design matrices to get fit beams
            fit_beam_az = np.tensordot(self.trig_matr, fit_coeffs, axes=((1,), (1,))) # Naz, Nn, Naxes_vec, 1, Nfeeds, Nfreq
            fit_beam = np.tensordot(self.bess_matr, fit_beam_az, axes=((1,), (1,))) # Nza, Naz, Naxes_vec, 1, Nfeeds, Nfreq
            fit_beam = fit_beam.transpose(2, 3, 4, 5, 0, 1)

            np.save(f"{self.save_fn}_bess_fit_coeffs.npy", fit_coeffs)
            np.save(f"{self.save_fn}_bess_fit_beam.npy", fit_beam)        

        return fit_coeffs, fit_beam
    
    def get_comp_inds(self, num_modes=64):
        """
        Get the indices for the num_modes most significant modes for each
        feed, polarization, and frequency.

        Parameters:
            num_modes (int): 
                The number of modes to use for the compressed fit.

        Returns:
            nmodes_comp (array, int):
                The radial mode numbers corresponding to the top num_modes 
                Fourier-Bessl modes, in descending order of significance. Has 
                shape (Naxes_vec, 1, Nfeeds, Nfreqs, num_modes).
            mmodes_comp (array, int):    
                The azimuthal modes numbers corresponding to the top num_modes 
                Fourier-Bessel modes, in descending order of significance. Has
                shape (Naxes_vec, 1, Nfeeds, Nfreqs, num_modes).
        """

        ps_sort_inds = np.argsort(self.bess_ps.reshape((self.ncoeff_bess, 
                                                        self.Naxes_vec, 1, 
                                                        self.Nfeeds, 
                                                        self.Nfreqs)),
                                  axis=0)
        # Highest modes start from the end
        sort_inds_flip = np.flip(ps_sort_inds, axis=0)[:num_modes]
        nmodes_comp, mmodes_comp = np.unravel_index(sort_inds_flip, 
                                                    (self.nmax, 
                                                     len(self.mmodes)))
        
        return nmodes_comp, mmodes_comp
    
    def get_comp_fits(self, num_modes=64):
        """
        Get the beam fit coefficients and fit beams in a compressed basis using
        num_modes modes for each polarization, feed, and frequency.

        Parameters:
            num_modes (int):
                The number of Fourier-Bessel modes to use for the compresed fit.
            
        Returns:
            fit_coeffs (array, complex):
                The coefficients for the Fourier-Bessel fit. Has shape
                (Naxes_vec, 1, Nfeeds, Nfreqs, num_modes)
            fit_beam (array, complex):
                The fit beam in sky coordinates. Has shape 
                (Naxes_vec, 1, Nfeeds, Nfreqs, Nza, Naz)
        """
        nmodes_comp, mmodes_comp = self.get_comp_inds(num_modes=num_modes)
        num_modes = nmodes_comp.shape[0]

        
        fit_coeffs, fit_beam = self.sparse_fit_loop(num_modes, nmodes_comp, mmodes_comp)
        
        return fit_coeffs, fit_beam

    def sparse_fit_loop(self, num_modes, nmodes_comp, mmodes_comp, 
                        fit_coeffs=None, do_fit=True, bess_matr=None,
                        trig_matr=None):
        """
        Do a loop over all the axes and fit/evaluate fit in position space.

        Parameters:
            num_modes (int): 
                Number of modes in the sparse fit.
            nmodes_comp (array of int):
                Which nmodes are being used in the sparse fit 
                (output of get_comp_inds method).
            mmodes_comp (array_of_int):
                Which mmodes are being used in the sparse fit 
                (output of get_comp_inds method).
            fit_coeffs (array, complex):
                Precomputed fit coefficients (if just evaluating).
            do_fit (bool):
                Whether to do the fit (set to False if fit_coeffs supplied).
            bess_matr (array):
                Bessel part of design matrix.
            trig_matr (array, complex):
                Fourier part of design matrix.
        
        Returns:
            fit_coeffs (array, complex; if do_fit is True):
                The newly calculated fit coefficients in the sparse basis.
            fit_beam (array, complex):
                The sparsely fit beam evaluated in position space.
        """
        # nmodes might vary from pol to pol, freq to freq. The fit is fast, just do a big for loop.
        interp_kwargs = [bess_matr, trig_matr, fit_coeffs]
        if do_fit:
            fit_coeffs = np.zeros([self.Naxes_vec, 1, self.Nfeeds, self.Nfreqs, 
                                  num_modes], dtype=complex)
            beam_shape = self.data_array.shape
            bess_matr = self.bess_matr
            trig_matr = self.trig_matr
        elif any([item is None for item in interp_kwargs]):
            raise ValueError("Must supply fit_coeffs, bess_matr, and trig_matr "
                             "if not doing fit.")
        else:
            Npos = bess_matr.shape[0]
            beam_shape = (self.Naxes_vec, 1, self.Nfeeds, self.Nfreqs, Npos)
        fit_beam = np.zeros(beam_shape, dtype=complex)
        
        for vec_ind in range(self.Naxes_vec):
            for feed_ind in range(self.Nfeeds):
                for freq_ind in range(self.Nfreqs):
                    if do_fit:
                        dat_iter = self.data_array[vec_ind, 0, feed_ind, freq_ind]
                    nmodes_iter = nmodes_comp[:, vec_ind, 0, feed_ind, freq_ind]
                    mmodes_iter = mmodes_comp[:, vec_ind, 0, feed_ind, freq_ind]
                    unique_mmodes_iter = np.unique(mmodes_iter)
       
                    for mmode in unique_mmodes_iter:
                        mmode_inds = mmodes_iter == mmode
                        # Get the nmodes that this mmode is used for
                        nmodes_mmode = nmodes_iter[mmode_inds] 

                        bess_matr_mmode = bess_matr[:, nmodes_mmode]
                        trig_mode = trig_matr[:, mmode]

                        if do_fit:
                            az_fit_mmode = dat_iter @ trig_mode.conj() # Nza

                            fit_coeffs_mmode = lstsq(bess_matr_mmode, az_fit_mmode)[0]
                            fit_coeffs[vec_ind, 0, feed_ind, freq_ind, mmode_inds] = fit_coeffs_mmode
                            fit_beam[vec_ind, 0, feed_ind, freq_ind] += np.outer(bess_matr_mmode @ fit_coeffs_mmode, trig_mode)
                        else:
                            fit_coeffs_mmode = fit_coeffs[vec_ind, 0, feed_ind, freq_ind, mmode_inds]
                            fit_beam[vec_ind, 0, feed_ind, freq_ind] += (bess_matr_mmode @ fit_coeffs_mmode) * trig_mode
                        
        if do_fit:
            return fit_coeffs,fit_beam
        else:
            return fit_beam
    
    def interp(self, sparse_fit=False, fit_coeffs=None, az_array=None, 
               za_array=None, reuse_spline=False, **kwargs):
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
            reuse_spline (array):
                Whether to reuse the design matrix for a particular az_array and
                za_array (named to keep consistency with UVBeam).

        Returns:
            beam_vals (array, complex):
                The values of the beam at the interpolated 
                frequencies/spatial positions. Has shape 
                (Naxes_vec, 1, Npols, Nfreqs, Npos).
        """
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
            else:
                self.az_array_dict[az_hash] = az_array
                self.za_array_dict[za_hash] = za_array
                bess_matr, trig_matr = self.get_dmatr_interp(az_array, za_array)
                self.trig_matr_interp_dict[az_hash] = trig_matr
                self.bess_matr_interp_dict[za_hash] = bess_matr
        
        if sparse_fit:
            num_modes = fit_coeffs.shape[-1]
            nmodes_comp, mmodes_comp = self.get_comp_inds(num_modes)
            print("Getting beam_vals")
            beam_vals = self.sparse_fit_loop(num_modes, nmodes_comp, 
                                             mmodes_comp, fit_coeffs=fit_coeffs,
                                             do_fit=False, bess_matr=bess_matr,
                                             trig_matr=trig_matr)
        else:
            beam_vals = np.tensordot(trig_matr[:, np.newaxis] * bess_matr[:, :, np.newaxis], self.bess_fits, axes=2).transpose(1, 2, 3, 4, 0)
            
        return beam_vals
