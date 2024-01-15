from pyuvdata import UVBeam
import numpy as np
from scipy.special import jn, jn_zeros
from scipy.linalg import solve, lstsq


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
        self.rad_array = np.sqrt(1 - np.cos(self.axis2_array)) / alpha
        #self.rad_array = np.sqrt(np.cos(self.axis2_array))
        
        self.az_grid, self.rad_grid = np.meshgrid(self.az_array, self.rad_array)
        self.ncoord = self.az_grid.size
        
        self.nmax = nmax
        self.mmodes = mmodes
        self.ncoeff_bess = self.nmax * len(self.mmodes)
        
        self.bess_matr = self.get_bess_matr()

        
        self.daz = self.axis1_array[1] - self.axis1_array[0]
        self.drad = self.rad_array[1] - self.rad_array[0]
        self.dA = self.rad_grid * self.drad * self.daz
        
        self.save_fn = save_fn
        self.bess_fits, self.bess_beam = self.get_fits('bess', load=load)
        self.bess_ps = np.abs(self.bess_fits)**2 
        
        
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
        
        if self.bound == "Dirichlet":
            zeros = jn_zeros(0, self.nmax)
            orth = jn(1, zeros)
        else:
            zeros = jn_zeros(1, self.nmax - 1)
            orth = jn(2, zeros)
            
            zeros = np.append(0, zeros)
            orth = np.append(1, orth)
        orth = orth / np.sqrt(2)
        Naz = len(self.az_array)
                
        bess_matr = jn(0, zeros[np.newaxis] * self.rad_array[:, np.newaxis]) / orth
        # Assume a regular az spacing and just make a unitary DFT matrix; better for fitting later
        trig_matr = np.exp(1.0j * np.array(self.mmodes)[np.newaxis] * self.az_array[:, np.newaxis]) / np.sqrt(Naz)
        
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
            bess_matr, trig_matr = self.get_dmatr()
            # az_modes are discretely orthonormal so just project onto the basis
            # Saves loads of memory and time
            az_fit = self.data_array @ trig_matr.conj() # Naxes_vec, 1, Nfeeds, Nfreq, Nza, Nm

            BtB = bess_matr.T @ bess_matr
            Baz = bess_matr.T @ az_fit # Naxes_vec, 1, Nfeeds, Nfreq, Nn, Nm
            Baz = Baz.transpose(4, 5, 0, 1, 2, 3) # Nn, Nm, Naxes_vec, 1, Nfeeds, Nfreq


            fit_coeffs = solve(BtB, Baz, assume_a="sym")[0] # Nn, Nm, Naxes_vec, 1, Nfeeds, Nfreq

            # Apply design matrices to get fit beams
            fit_beam_az = np.tensordot(trig_matr, fit_beam_rad, axes=((1,), (1,))) # Naz, Nn, Naxes_vec, 1, Nfeeds, Nfreq
            fit_beam = np.tensordot(bess_matr, fit_beam_az, axes=((1,), (1,))) # Nza, Naz, Naxes_vec, 1, Nfeeds, Nfreq
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

        ps_sort_inds = np.argsort(self.bess_ps.reshape((self.Naxes_vec, 1, self.Nfeed, self.Nfreqs, 
                                                        self.ncoeff_bess)),
                                  axis=4)
        # Highest modes start from the end
        sort_inds_flip = np.flip(ps_sort_inds, axis=4)[:, :, :, :, :num_modes]
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
        bess_matr, trig_matr = self.get_dmatr()
        num_modes = nmodes_comp.shape[-1]

        # nmodes might vary from pol to pol, freq to freq. The fit is fast, just do a big for loop.
        fit_coeffs = np.zeros(self.Naxes_vec, 1, self.Nfeeds, self.Nfreqs, 
                              num_modes, dtype=complex)
        fit_beam = np.zeros_like(self.data_array)
        for vec_ind in range(self.Naxes_vec):
            for feed_ind in range(self.Nfeeds):
                for freq_ind in range(self.Nfreqs):
                    dat_iter = self.data_array[vec_ind, 0, feed_ind, freq_ind]
                    nmodes_iter = nmodes_comp[vec_ind, 0, feed_ind, freq_ind]
                    mmodes_iter = mmodes_comp[vec_ind, 0, feed_ind, freq_ind]
                    unique_mmodes_iter = np.unique(mmodes_iter)
       
                    for mmode in unique_mmodes_iter:
                        mmode_inds = mmodes_iter == mmode
                        # Get the nmodes that this mmode is used for
                        nmodes_mmode = nmodes_iter[mmode_inds] 

                        bess_matr_mmode = bess_matr[:, nmodes_mmode]
                        trig_mode = trig_matr[:, mmode]

                        az_fit_mmode = dat_iter @ trig_mode.conj() # Nza

                        fit_coeffs_mmode = lstsq(bess_matr_mmode, az_fit_mmode)[0]

                        fit_coeffs[vec_ind, 0, feed_ind, freq_ind, mmode_inds] = fit_coeffs_mmode
                        fit_beam[vec_ind, 0, feed_ind, freq_ind] += np.outer(bess_matr_mmode @ fit_coeffs_mmode, trig_mode)
        
        return fit_coeffs, fit_beam