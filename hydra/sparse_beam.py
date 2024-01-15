from pyuvdata import UVBeam
import numpy as np
from scipy.special import jn, jn_zeros
from scipy.linalg import solve


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
                The coefficients for the Fourier-Bessel basis. Has shape
                (nmax, len(mmodes), 2, 1, 2, Nfreqs)
            fit_beam (array, complex):
                The fit beam in sky coordinates. Has shape 
                (2, 1, 2, Nfreqs, Nza, Naz)
        """
        
        if load:
            fit_coeffs = np.load(f"{self.save_fn}_bess_fit_coeffs.npy")
            fit_beam = np.load(f"{self.save_fn}_bess_fit_beam.npy")
        else:
            bess_matr, trig_matr = self.get_dmatr()
            # az_modes are discretely orthonormal so just project onto the basis
            # Saves loads of memory and time
            az_fit = self.data_array @ trig_matr.conj() # 2, 1, 2, Nfreq, Nza, Nm

            BtB = bess_matr.T @ bess_matr
            Baz = bess_matr.T @ az_fit # 2, 1, 2, Nfreq, Nn, Nm
            Baz = Baz.transpose(4, 5, 0, 1, 2, 3) # Nn, Nm, 2, 1, 2, Nfreq


            fit_coeffs = solve(BtB, Baz, assume_a="sym")[0] # Nn, Nm, 2, 1, 2, Nfreq

            # Apply design matrices to get fit beams
            fit_beam_az = np.tensordot(trig_matr, fit_beam_rad, axes=((1,), (1,))) # Naz, Nn, 2, 1, 2, Nfreq
            fit_beam = np.tensordot(bess_matr, fit_beam_az, axes=((1,), (1,))) # Nza, Naz, 2, 1, 2, Nfreq
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
                Fourier-Bessl modes, in descending order of significance.
            mmodes_comp (array, int):    
                The azimuthal modes numbers corresponding to the top num_modes 
                Fourier-Bessel modes, in descending order of significance.
        """

        ps_sort_inds = np.argsort(self.bess_ps.reshape((2, 1, 2, self.Nfreqs, 
                                                        self.ncoeff_bess)),
                                  axis=4)
        # Highest modes start from the end
        sort_inds_flip = np.flip(ps_sort_inds, axis=4)[:, :, :, :, :num_modes]
        nmodes_comp, mmodes_comp = np.unravel_index(sort_inds_flip, 
                                                    (self.nmax, 
                                                     len(self.mmodes)))
        
        return nmodes_comp, mmodes_comp
    
    def get_comp_fits(self, basis_matr, comp_inds, lowmem=True, freq_dep=True):
        
        num_modes = comp_inds[0].shape[-1]
        if lowmem:
            fit_coeffs = np.zeros((2, 1, 2, self.Nfreqs, num_modes), dtype=complex)
            fit_beams = np.zeros_like(self.data_array)
            for freq_ind in range(self.Nfreqs):
                B = getattr(self, basis_matr)[:, :, comp_inds[0][:, :, :, freq_ind], 
                                                 comp_inds[1][:, :, :, freq_ind]]
                Bres = B.reshape(self.ncoord, 2, 1, 2, num_modes).transpose(1, 2, 3, 0, 4)

                BTdA = (Bres.conj().transpose(0, 1, 2, 4, 3)) 

                lhs_op = BTdA @ Bres
                lhs_op = block_diag(lhs_op.reshape(4, num_modes, num_modes))


                rhs_vec = (BTdA * self.data_array[:, :, :, freq_ind].reshape(2, 1, 2, 1, self.ncoord)).sum(axis=-1)
                rhs_vec = rhs_vec.flatten()
                soln = spsolve(lhs_op, rhs_vec)
                soln_res = soln.reshape(2, 1, 2, num_modes)
                fit_coeffs[:, :, :, freq_ind] = soln_res
                fit_beams[:, :, :, freq_ind, :, :] = (B * soln_res).sum(axis=-1).transpose(2, 3, 4, 0, 1)
        else:
            if freq_dep:
                raise NotImplemented("High mem option not implemented for freq_dep bases yet.")
            
            print("Indexing B")
            Bres = getattr(self, basis_matr)[:, :, comp_inds[0][0,0,0,0], comp_inds[1][0,0,0,0]]
            print("Reshaping B")
            Bres = Bres.reshape(self.ncoord, num_modes)

            BTdA = Bres.T.conj()
            print("computing lhs_op")
            lhs_op = BTdA @ Bres
            
            print("computing rhs_vec")
            rhs_vec = np.tensordot(BTdA, self.data_array.reshape(2, 1, 2, self.Nfreqs, self.ncoord),
                                   axes=((-1,), (-1,)))
            print(f"rhs_vec shape: {rhs_vec.shape}")
            print("Solving")
            # Nmode, 2, 1, 2, Nfreq
            fit_coeffs = solve(lhs_op, rhs_vec)
            
            # ncoord, 2, 1, 2, Nfreq -> 2, 1, 2, Nfreq, ncoord
            fit_beams = np.tensordot(Bres, fit_coeffs, axes=1).transpose(1, 2, 3, 4, 0)
            
            # Nmode, 2, 1, 2, Nfreq -> 2, 1, 2, Nfreq, Nmode
            fit_coeffs = fit_coeffs.transpose(1, 2, 3, 4, 0)
            # 2, 1, 2, Nfreq, ncoord -> 2, 1, 2, Nfreq, Nza, Naz
            fit_beams = fit_beams.reshape(2, 1, 2, self.Nfreqs, len(self.rad_array), len(self.az_array))
        
        return fit_coeffs, fit_beams