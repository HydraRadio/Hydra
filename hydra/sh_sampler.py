from mpi4py.MPI import SUM as MPI_SUM

import numpy as np
import scipy as sp
import healpy as hp

from .vis_simulator import simulate_vis_per_alm

# Wigner D matrices
import spherical, quaternionic

# Simulation
import pyuvsim

# Linear solver 
from scipy.sparse.linalg import cg, LinearOperator

from astropy import units
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.coordinates.builtin_frames import AltAz, ICRS
from astropy.time import Time
import time


def vis_proj_operator_no_rot(freqs, 
                             lsts, 
                             beams, 
                             ant_pos, 
                             lmax, 
                             nside, 
                             latitude=-0.5361913261514378, 
                             include_autos=False, 
                             autos_only=False,
                             ref_freq=100.,
                             spectral_idx=0.):
    """
    Precompute the real and imaginary blocks of the visibility response 
    operator. This should only be done once and then "apply_vis_response()"
    is used to get the actual visibilities.
    
    Parameters:
        freqs (array_like):
            Frequencies, in MHz.
        lsts (array_like):
            LSTs (times) for the simulation. In radians.
        beams (list of pyuvbeam):
            List of pyuveam objects, one for each antenna
        ant_pos (dict):
            Dictionary of antenna positions, [x, y, z], in m. The keys should
            be the numerical antenna IDs.    
        lmax (int):
            Maximum ell value. Determines the number of modes used.
        nside (int):
            Healpix nside to use for the calculation (longer baselines should 
            use higher nside).
        latitude (float):
            Latitude in decimal format of the simulated array/visibilities. 
        include_autos (bool):
            If `True`, the auto baselines are included.
        ref_freq (float):
            Reference frequency for the spectral dependence, in MHz.
        spectral_idx (float):
            Spectral index, `beta`, for the spectral dependence, 
            `~(freqs / ref_freq)^beta`.
    
    Returns:
        vis_response_2D (array_like):
            Visibility operator (δV_ij) for each (l,m) mode, frequency, 
            baseline and lst. Shape (Nvis, Nalms) where Nvis is Nbl x Ntimes x Nfreqs.
        ell (array of int):
            Array of ell-values for the visiblity simulation
        m  (array of int):
            Array of ell-values for the visiblity simulation
    """
    ell, m, vis_alm = simulate_vis_per_alm(lmax=lmax, 
                                           nside=nside, 
                                           ants=ant_pos, 
                                           freqs=freqs*1e6, # MHz -> Hz 
                                           lsts=lsts, 
                                           beams=beams,
                                           latitude=latitude)
    
    # Removing visibility responses corresponding to the m=0 imaginary parts 
    vis_alm = np.concatenate((vis_alm[:,:,:,:,:len(ell)],
                              vis_alm[:,:,:,:,len(ell)+(lmax+1):]), 
                             axis=4)
    
    ants = list(ant_pos.keys())
    antpairs = []
    if autos_only == False and include_autos == False:
        auto_ants = []
    for i in ants:
        for j in ants:
            # Toggle via keyword argument if you want to keep the auto baselines/only have autos
            if include_autos == True:
                if j >= i:
                    antpairs.append((ants[i], ants[j]))
            elif autos_only == True:
                if j == i:
                    antpairs.append((ants[i], ants[j]))
            else:
                if j == i:
                    auto_ants.append((ants[i], ants[j]))
                if j > i:
                    antpairs.append((ants[i], ants[j]))
                
    vis_response = np.zeros((len(antpairs), len(freqs), len(lsts), 2*len(ell)-(lmax+1)), 
                            dtype=np.complex128)
    
    ## Collapse the two antenna dimensions into one baseline dimension
    # Nfreqs, Ntimes, Nant1, Nant2, Nalms --> Nbl, Nfreqs, Ntimes, Nalms 
    for i, bl in enumerate(antpairs):
        idx1 = ants.index(bl[0])
        idx2 = ants.index(bl[1])
        vis_response[i, :] = vis_alm[:, :, idx1, idx2, :]  
    
    # Multiply by spectral dependence model (a powerlaw)
    # Shape: Nbl, Nfreqs, Ntimes, Nalms 
    vis_response *= ((freqs / ref_freq)**spectral_idx)[np.newaxis,:,np.newaxis,np.newaxis]

    # Reshape to 2D
    # TODO: Make this into a "pack" and "unpack" function
    # Nbl, Nfreqs, Ntimes, Nalms --> Nvis, Nalms
    Nvis = len(antpairs) * len(freqs) * len(lsts)
    vis_response_2D = vis_response.reshape(Nvis, 2*len(ell)-(lmax+1))
    
    if autos_only == False and include_autos == False:
        autos = np.zeros((len(auto_ants),len(freqs),len(lsts),2*len(ell)-(lmax+1)), dtype=np.complex128)
        ## Collapse the two antenna dimensions into one baseline dimension
        # Nfreqs, Ntimes, Nant1, Nant2, Nalms --> Nbl, Nfreqs, Ntimes, Nalms 
        for i, bl in enumerate(auto_ants):
            idx1 = ants.index(bl[0])
            idx2 = ants.index(bl[1])
            autos[i, :] = vis_alm[:, :, idx1, idx2, :]   

        ## Reshape to 2D
        ## TODO: Make this into a "pack" and "unpack" function
        # Nbl, Nfreqs, Ntimes, Nalms --> Nvis, Nalms
        Nautos = len(auto_ants) * len(freqs) * len(lsts)
        autos_2D = autos.reshape(Nautos, 2*len(ell)-(lmax+1))

    if autos_only == False and include_autos == False:
        return vis_response_2D, autos_2D, ell, m
    else:
        return vis_response_2D, ell, m


def alms2healpy(alms, lmax):
    """
    Takes a real array split as [real, imag] (without the m=0 modes 
    imag-part) and turns it into a complex array of alms (positive 
    modes only) ordered as in HEALpy.
      
    Parameters:
        alms (array_like):
            Array of zeros except for the specified mode. 
            The array represents all positive (+m) modes including zero 
            and has double length, as real and imaginary values are split. 
            The first half is the real values.
    
    Returns:
        healpy_modes (array_like):
            Array of zeros except for the specified mode. 
            The array represents all positive (+m) modes including zeroth modes.
    """
    
    real_imag_split_index = int((np.size(alms)+(lmax+1))/2)
    real = alms[:real_imag_split_index]
    
    add_imag_m0_modes = np.zeros(lmax+1)
    imag = np.concatenate((add_imag_m0_modes, alms[real_imag_split_index:]))
    
    healpy_modes = real + 1.j*imag
    
    return healpy_modes
    
    
def healpy2alms(healpy_modes):
    """
    Takes a complex array of alms (positive modes only) and turns into
    a real array split as [real, imag] making sure to remove the 
    m=0 modes from the imag-part.
      
    Parameters:
        healpy_modes (array_like, complex):
            Array of zeros except for the specified mode. 
            The array represents all positive (+m) modes including zeroth modes.
    
    Returns:
        alms (array_like):
            Array of zeros except for the specified mode. 
            The array represents all positive (+m) modes including zero 
            and is split into a real (first) and imag (second) part. The
            Imag part is smaller as the m=0 modes shouldn't contain and 
            imaginary part. 
    """
    lmax = hp.sphtfunc.Alm.getlmax(healpy_modes.size) # to remove the m=0 imag modes
    alms = np.concatenate((healpy_modes.real,healpy_modes.imag[(lmax+1):]))
        
    return alms   


def get_healpy_from_gsm(freq, lmax, nside=64, resolution="low", output_model=False, 
                        output_map=False):
    """
    Generate an array of alms (HEALpy ordered) from gsm 2016 
    (https://github.com/telegraphic/pygdsm)
    
    Parameters:
        freqs (array_like):
            Frequency (in MHz) for which to return GSM model.
        lmax (int):
            Maximum ell value for alms
        nside (int):
            The nside to upgrade/downgrade the map to. Default is nside=64.
        resolution (str):
            if "low/lo/l":  The GSM nside = 64  (default)
            if "hi/high/h": The GSM nside = 1024 
        output_model (bool):
            If output_model=True: Outputs model generated from the GSM data. 
            If output_model=False (default): no model output.
        output_map (bool):
            If output_map=True: Outputs map generated from the GSM data. 
            If output_map=False (default): no map output.

    Returns:
        healpy_modes (array_like):
            Complex array of alms with same size and ordering as in healpy (m,l)
        gsm_2016 (PyGDSM 2016 model):
            If output_model=True: Outputs model generated from the GSM data. 
            If output_model=False (default): no model output.
        gsm_map (healpy map):
            If output_map=True: Outputs map generated from the GSM data. 
            If output_map=False (default): no map output.
    
    """
    # Instantiate GSM model and extract alms
    gsm_2016 = GlobalSkyModel2016(freq_unit='MHz', resolution=resolution) 
    gsm_map = gsm_2016.generate(freqs=freq)
    gsm_upgrade = hp.ud_grade(gsm_map, nside)
    healpy_modes_gal = hp.map2alm(maps=gsm_upgrade,lmax=lmax)

    # By default it is in gal-coordinates, convert to equatorial
    rot_gal2eq = hp.Rotator(coord="GC")
    healpy_modes_eq = rot_gal2eq.rotate_alm(healpy_modes_gal)

    if output_model == False and output_map == False: # default
        return healpy_modes_eq
    elif output_model == False and output_map == True:
        return healpy_modes_eq, gsm_map 
    elif output_model == True and output_map == False:
        return healpy_modes_eq, gsm_2016 
    else:
        return healpy_modes_eq, gsm_2016, gsm_map


def get_alms_from_gsm(freq, lmax, nside=64, resolution='low', output_model=False, 
                      output_map=False):
    """
    Generate a real array split as [real, imag] (without the m=0 modes 
    imag-part) from gsm 2016 (https://github.com/telegraphic/pygdsm)
    
    Parameters:
    freqs (float or array_like):
        Frequency (in MHz) for which to return GSM model
    lmax (int):
        Maximum ell value for alms
    nside (int):
        The nside to upgrade/downgrade the map to. Default is nside=64.
    resolution (str):
        if "low/lo/l":  nside = 64  (default)
        if "hi/high/h": nside = 1024 
    output_model (bool):
        If output_model=True: Outputs model generated from the GSM data. 
        If output_model=False (default): no model output.
    output_map (bool):
        If output_map=True: Outputs map generated from the GSM data. 
        If output_map=False (default): no map output.

    Returns:
        alms (array_like):
            Array of zeros except for the specified mode. 
            The array represents all positive (+m) modes including zero 
            and has double length, as real and imaginary values are split. 
            The first half is the real values.
        gsm_2016 (PyGDSM 2016 model):
            If output_model=True: Outputs model generated from the GSM data. 
            If output_model=False (default): no model output.
        gsm_map (healpy map):
            If output_map=True: Outputs map generated from the GSM data. 
            If output_map=False (default): no map output.
    """
    return healpy2alms(get_healpy_from_gsm(freq, lmax, nside, resolution, output_model, output_map))


def construct_rhs_no_rot(data, inv_noise_var, inv_prior_var, omega_0, omega_1, a_0, vis_response):
    """
    Construct RHS of linear system.
    """
    real_data_term = vis_response.real.T @ (inv_noise_var*data.real 
                                            + np.sqrt(inv_noise_var)*omega_1.real)
    imag_data_term = vis_response.imag.T @ (inv_noise_var*data.imag 
                                            + np.sqrt(inv_noise_var)*omega_1.imag)
    prior_term = inv_prior_var*a_0 + np.sqrt(inv_prior_var)*omega_0

    right_hand_side = real_data_term + imag_data_term + prior_term 
    
    return right_hand_side


def apply_lhs_no_rot(a_cr, inv_noise_var, inv_prior_var, vis_response):
    """
    Apply LHS operator of linear system to an input vector.
    """
    real_noise_term = vis_response.real.T \
                    @ ( inv_noise_var[:,np.newaxis] * vis_response.real ) \
                    @ a_cr
    imag_noise_term = vis_response.imag.T \
                    @ ( inv_noise_var[:,np.newaxis]* vis_response.imag ) \
                    @ a_cr
    signal_term = inv_prior_var * a_cr
    
    left_hand_side = (real_noise_term + imag_noise_term + signal_term) 
    return left_hand_side


def construct_rhs_no_rot_mpi(comm, data, inv_noise_var, inv_prior_var, 
                             omega_a, omega_n, a_0, vis_response):
    """
    Construct RHS of linear system from data split across multiple MPI workers.
    """
    myid = comm.Get_rank()

    # Synchronise omega_a across all workers
    if myid != 0:
        omega_a *= 0.
    comm.Bcast(omega_a, root=0)

    # Calculate data terms
    my_data_term = vis_response.real.T @ ((inv_noise_var * data.real).flatten()
                                          + np.sqrt(inv_noise_var).flatten()
                                            * omega_n.real.flatten()) \
                 + vis_response.imag.T @ ((inv_noise_var * data.imag).flatten()
                                          + np.sqrt(inv_noise_var).flatten() 
                                            * omega_n.imag.flatten())
    
    # Do Reduce (sum) operation to get total operator on root node
    data_term = np.zeros((1,), dtype=my_data_term.dtype) # dummy data for non-root workers
    if myid == 0:
        data_term = np.zeros_like(my_data_term)
    
    comm.Reduce(my_data_term, data_term, op=MPI_SUM, root=0)
    comm.barrier()

    # Return result (only root worker has correct result)
    if myid == 0:
        return data_term \
             + inv_prior_var * a_0 \
             + np.sqrt(inv_prior_var) * omega_a
    else:
        return np.zeros_like(a_0)


def apply_lhs_no_rot_mpi(comm, a_cr, inv_noise_var, inv_prior_var, vis_response):
    """
    Apply LHS operator of linear system to an input vector that has been 
    split into chunks between MPI workers.
    """
    myid = comm.Get_rank()

    # Synchronise a_cr across all workers
    if myid != 0:
        a_cr *= 0.
    comm.Bcast(a_cr, root=0)

    # Calculate noise terms for this rank
    my_tot_noise_term = vis_response.real.T \
                        @ ( inv_noise_var.flatten()[:,np.newaxis] * vis_response.real ) \
                        @ a_cr \
                      + vis_response.imag.T \
                        @ ( inv_noise_var.flatten()[:,np.newaxis] * vis_response.imag ) \
                        @ a_cr

    # Do Reduce (sum) operation to get total operator on root node
    tot_noise_term = np.zeros((1,), dtype=my_tot_noise_term.dtype) # dummy data for non-root workers
    if myid == 0:
        tot_noise_term = np.zeros_like(my_tot_noise_term)
    
    comm.Reduce(my_tot_noise_term,
                tot_noise_term,
                op=MPI_SUM,
                root=0)

    # Return result (only root worker has correct result)
    if myid == 0:
        signal_term = inv_prior_var * a_cr
        return tot_noise_term + signal_term
    else:
        return np.zeros_like(a_cr)


def radiometer_eq(auto_visibilities, ants, delta_time, delta_freq, Nnights = 1, include_autos=False):
    nbls = len(ants)
    indx = auto_visibilities.shape[0]//nbls
    
    sigma_full = np.empty((0))#, autos.shape[-1]))

    for i in ants:
        vis_ii = auto_visibilities[i*indx:(i+1)*indx]#,:]

        for j in ants:
            if include_autos == True:
                if j >= i:
                    vis_jj = auto_visibilities[j*indx:(j+1)*indx]#,:]
                    sigma_ij = ( vis_ii*vis_jj ) / ( Nnights*delta_time*delta_freq )
                    sigma_full = np.concatenate((sigma_full,sigma_ij))
            else:
                if j > i:  # only keep this line if you don't want the auto baseline sigmas
                    vis_jj = auto_visibilities[j*indx:(j+1)*indx]#,:]
                    sigma_ij = ( vis_ii*vis_jj ) / ( Nnights*delta_time*delta_freq )
                    sigma_full = np.concatenate((sigma_full,sigma_ij))
                    
    return sigma_full



# MAIN    
if __name__ == "__main__":
    start_time = time.time()
    
    # Creating directory for output
    if ARGS['directory']: 
        directory = str(ARGS['directory'])
    else:
        directory = "output"

    path = f'/cosma8/data/dp270/dc-bull2/{directory}/'
    try: 
        os.makedirs(path)
    except FileExistsError:
        print('folder already exists')
    
    # Defining the data_seed for the precomputation random seed
    if ARGS['data_seed']:
        data_seed = int(ARGS['data_seed'])
    else:
        # if none is passed go back to 10 as before
        data_seed = 10

    # Defining the jobid to distinguish multiple runs in one go
    if ARGS['jobid']: 
        jobid = int(ARGS['jobid'])
    else:
        # if none is passed then don't change the keys
        jobid = 0

    ant_pos = build_hex_array(hex_spec=(3,4), d=14.6)  #builds array with (3,4,3) ants = 10 total
    ants = list(ant_pos.keys())
    lmax = 20
    nside = 128
    beam_diameter = 14.
    beams = [pyuvsim.AnalyticBeam('gaussian', diameter=beam_diameter) for ant in ants]
    freqs = np.linspace(100e6, 102e6, 2)
    lsts_hours = np.linspace(0.,8.,10)      # in hours for easy setting
    lsts = np.deg2rad((lsts_hours/24)*360) # in radian, used by HYDRA (and this code)
    delta_time = 60 # s
    delta_freq = 1e+06 # (M)Hz
    latitude = 31.7215 * np.pi / 180  # HERA loc in decimal numbers ## There's some sign error in the code, so this missing sign is a quick fix
    solver = cg

    vis_response, autos, ell, m = vis_proj_operator_no_rot(freqs=freqs, 
                                                        lsts=lsts, 
                                                        beams=beams, 
                                                        ant_pos=ant_pos, 
                                                        lmax=lmax, 
                                                        nside=nside,
                                                        latitude=latitude)

    np.random.seed(data_seed)
    x_true = get_alms_from_gsm(freq=100,lmax=lmax, nside=nside)
    model_true = vis_response @ x_true

    # Inverse noise covariance and noise on data
    noise_cov = radiometer_eq(autos@x_true, ants, delta_time, delta_freq)
    inv_noise_var = 1/noise_cov
    data_noise = np.random.randn(noise_cov.size)*np.sqrt(noise_cov) 
    data_vec = model_true + data_noise

    # Inverse signal covariance
    zero_value = 0.001
    prior_cov = (x_true*0.1)**2     # if 0.1 = 10% prior
    prior_cov[prior_cov == 0] = zero_value
    inv_prior_var = 1./prior_cov
    a_0 = np.random.randn(x_true.size)*np.sqrt(prior_cov) + x_true # gaussian centered on alms with S variance 

    # Define left hand side operator 
    def lhs_operator(x):
        y = apply_lhs_no_rot(x, inv_noise_var, inv_prior_var, vis_response)

        return y

    # Wiener filter solution to provide initial guess:
    omega_0_wf = np.zeros_like(a_0)
    omega_1_wf = np.zeros_like(model_true, dtype=np.complex128)
    rhs_wf = construct_rhs_no_rot(data_vec,
                                  inv_noise_var, 
                                  inv_prior_var, 
                                  omega_0_wf, 
                                  omega_1_wf, 
                                  a_0, 
                                  vis_response)
    
    # Build linear operator object 
    lhs_shape = (rhs_wf.size, rhs_wf.size)
    lhs_linear_op = LinearOperator(matvec = lhs_operator,
                                       shape = lhs_shape)

    # Get the Wiener Filter solution for initial guess
    wf_soln, wf_convergence_info = solver(A = lhs_linear_op,
                                          b = rhs_wf,
                                          # tol = 1e-07,
                                          maxiter = 15000)
    
    def samples(key):
        t_iter = time.time()

        # Set a random seed defined by the key
        random_seed = 100*jobid + key
        np.random.seed(random_seed)
        #random_seed = np.random.get_state()[1][0] #for test/output purposes

        # Generate random maps for the realisations
        omega_0 = np.random.randn(a_0.size)
        omega_1 = (np.random.randn(model_true.size) + 1.j*np.random.randn(model_true.size))/np.sqrt(2)

        # Construct the right hand side
        rhs = construct_rhs_no_rot(data_vec,
                                   inv_noise_var, 
                                   inv_prior_var,
                                   omega_0,
                                   omega_1,
                                   a_0,
                                   vis_response)

        # Run and time solver
        time_start_solver = time.time()
        x_soln, convergence_info = solver(A = lhs_linear_op,
                                          b = rhs,
                                          # tol = 1e-07,
                                          maxiter = 15000,
                                          x0 = wf_soln) #initial guess
        solver_time = time.time() - time_start_solver
        iteration_time = time.time()-t_iter
        
        # Save output
        np.savez(path+'results_'+f'{data_seed}_'+f'{random_seed}',
                 omega_0=omega_0,
                 omega_1=omega_1,
                 key=key,
                 x_soln=x_soln,
                 rhs=rhs,
                 convergence_info=convergence_info,
                 solver_time=solver_time,
                 iteration_time=iteration_time
        )
        
        return key, iteration_time
    
    # Time for all precomputations
    precomp_time = time.time()-start_time
    print(f'\nprecomputation took:\n{precomp_time}\n')
    
    avg_iter_time = 0

    # Multiprocessing, getting the samples    
    number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    print(f'\nSLURM_CPUS_PER_TASK = {number_of_cores}')

    with Pool(number_of_cores) as pool:
        # issue tasks and process results
        for result in pool.map(samples, range(100)):
            key, iteration_time = result
            avg_iter_time += iteration_time
            #print(f'Iteration {key} completed in {iteration_time:.2f} seconds')

    avg_iter_time /= (key+1)
    print(f'average_iter_time:\n{avg_iter_time}\n')

    total_time = time.time()-start_time
    print(f'total_time:\n{total_time}\n')
    print(f'All output saved in folder {path}\n')
    print(f'Note, ant_pos (dict) is saved in own file in {path}\n')
    
    # Saving all globally calculated data
    np.savez(path+'precomputed_data_'+f'{data_seed}_'+f'{jobid}',
             vis_response=vis_response,
             x_true=x_true,
             inv_noise_var=inv_noise_var,
             zero_value=zero_value,
             inv_prior_var=inv_prior_var,
             wf_soln=wf_soln,
             nside=nside,
             lmax=lmax,
             ants=ants,
             beam_diameter=beam_diameter,
             freqs=freqs,
             lsts_hours=lsts_hours,
             precomp_time=precomp_time,
             total_time=total_time
            )
    # creating a dictionary with string-keys as required by .npz files
    ant_dict = dict((str(ant), ant_pos[ant]) for ant in ant_pos)
    np.savez(path+'ant_pos',**ant_dict)
    
    
