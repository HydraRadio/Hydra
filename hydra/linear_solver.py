
from mpi4py.MPI import SUM as MPI_SUM 
from mpi4py.MPI import LAND as MPI_LAND
import numpy as np


def matvec_mpi(comm_row, mat_block, vec_block):
    """
    Do matrix-vector product for a row of a block matrix.
    
    Each block in the matrix row is multiplied by the corresponding row block 
    of the vector. The result on each worker is then summed together to give 
    the result for the corresponding row of the result vector.
    
    All workers in the row will posses the result for the same row of the 
    result vector.
    
    For example, for the first row of this (block) linear system:
    ( A B C )     ( x )     ( r0 )
    ( D E F )  .  ( y )  =  ( r1 )
    ( G H I )     ( z )     ( r2 )
    
    workers 0, 1, and 2 will compute Ax, By, and Cz respectively. They will 
    then collectively sum over their results to obtain `r0 = Ax + By + Cz`. The 
    three workers will all possess copies of `r0`.
    
    Parameters:
        comm_row (MPI.Intracomm):
            MPI group communicator for a row of the block matrix.
        mat_block (array_like):
            Block of the matrix belonging to this worker.
        vec_block (array_like):
            Block of the vector belonging to this worker.
    
    Returns:
        res_block (array_like):
            Block of the result vector corresponding to this row.
    """
    # Do matrix-vector product for the available blocks
    y = mat_block @ vec_block
    
    # Do reduce to all members of this column group
    ytot = np.zeros_like(y)
    comm_row.Allreduce(y, ytot, op=MPI_SUM)
    return ytot


def setup_mpi_blocks(comm, matrix_shape, split=1):
    """
    Set up a scheme for dividing the linear system into blocks. This function 
    determines the number and size of the blocks, creates a map between MPI 
    workers and blocks, and sets up some MPI communicator groups that are 
    needed by the CG solver to communicate intermediate results.
    
    The linear system matrix operator is assumed to be square, and the blocks 
    must also be square. The blocks will be zero-padded at the edges if the 
    operator cannot be evenly divided into the blocks.
    
    Parameters:
        comm (MPI.Communicator):
            MPI communicator object for all active workers.
        matrix_shape (tuple of int):
            The shape of the linear operator matrix that is to be divided into 
            blocks.
        split (int):
            How many rows and columns to split the matrix into. For instance, 
            `split = 2` will split the matrix into 2 rows and 2 columns, for a 
            total of 4 blocks.
    
    Returns:
        comm_groups (tuple of MPI.Intracomm):
            Group communicators for the blocks (active, row, col, diag). 
            
            These correspond to the MPI workers that are active, and the ones 
            for each row, each column, and along the diagonal of the block 
            structure, respectively.
            
            Each worker will return its own set of communicators (e.g. for the 
            row or column it belongs to). Where it is not a member of a 
            relevant group, `None` will be returned instead.
        block_map (dict):
            Dictionary of tuples, one for each worker in the `active` 
            communicator group, with the row and column ID of the block that it 
            is managing.
        block_shape (tuple of int):
            Shape of the square blocks that the full matrix operator (and RHS 
            vector) should be split up into. These will be square.
    """
    assert len(matrix_shape) == 2, \
        "'matrix_shape' must be a tuple with 2 entries"
    assert matrix_shape[0] == matrix_shape[1], \
        "Only square matrices are currently supported"
    myid = comm.Get_rank()
    nworkers = comm.Get_size()
    
    # Check that enough workers are available
    nblocks = split * split
    assert nworkers >= nblocks, "Specified more blocks than workers"
    workers = np.arange(nblocks).reshape((split, split))
    
    # Handle workers that do not have an assigned block
    if myid >= nblocks:
        return None, {}, {}
    
    # Construct map of block row/column IDs vs worker IDs
    block_map = {}
    for w in workers.flatten():
        _row, _col = np.where(workers == w)
        block_map[w] = (_row[0], _col[0])
    myrow, mycol = block_map[myid]
    
    # Setup communicator groups for each row, columns, and the diagonals
    grp_active = workers.flatten()
    grp_row = workers[myrow,:]
    grp_col = workers[:,mycol]
    grp_diag = np.diag(workers)
    comm_active = comm.Create( comm.group.Incl(grp_active) )
    comm_row = comm.Create( comm.group.Incl(grp_row) )
    comm_col = comm.Create( comm.group.Incl(grp_col) )
    comm_diag = comm.Create( comm.group.Incl(grp_diag) )
    
    # Calculate block size (all blocks must have the same shape, so some zero-
    # padding will be done if the matrix operator shape is not exactly 
    # divisible by 'split')
    block_rows = int(np.ceil(matrix_shape[0] / split)) # rows per block
    block_cols = int(np.ceil(matrix_shape[1] / split)) # cols per block
    block_shape = (block_rows, block_cols)
    assert block_rows == block_cols, \
        "Current implementation assumes that blocks are square"
    
    comms = (comm_active, comm_row, comm_col, comm_diag)
    return comms, block_map, block_shape


def collect_linear_sys_blocks(comm, block_map, block_shape, Amat=None, 
                              bvec=None, verbose=False):
    """
    Send LHS operator matrix and RHS vector blocks to assigned workers.
    
    Parameters:
        comm (MPI.Communicator):
            MPI communicator object for all active workers.
        block_map (dict):
            Dictionary of tuples, one for each worker in the `active` 
            communicator group, with the row and column ID of the block that it 
            is managing.
        block_shape (tuple of int):
            Shape of the square blocks that the full matrix operator (and RHS 
            vector) should be split up into. These must be square.
        Amat (array_like):
            The full LHS matrix operator, which will be split into blocks.
        bvec (array_like):
            The full right-hand side vector, which will be split into blocks.
        verbose (bool):
            If `True`, print status messages when MPI communication is complete.
    
    Returns:
        my_Amat (array_like):
            The single block of the matrix operator belonging to this worker. 
            It will have shape `block_shape`. If the matrix operator cannot be 
            exactly divided into same-sized blocks, the blocks at the far edges 
            will be zero-padded. Returns `None` if worker is not active.
        my_bvec (array_like):
            The single block of the RHS vector belonging to this worker. Note 
            that workers in the same column have the same block. Returns `None` 
            if worker is not active.
    """
    myid = comm.Get_rank()
    dtype = bvec.dtype
    
    # Determine whether this worker is participating
    workers_used = np.array(list(block_map.keys()))
    workers_used.sort()
    if myid not in workers_used:
        return None, None
    
    # Initialise blocks of A matrix and b vector
    block_rows, block_cols = block_shape
    my_Amat = np.zeros((block_rows, block_cols), dtype=dtype)
    my_bvec = np.zeros((block_rows,), dtype=dtype)
    
    # Send blocks from root worker
    if myid == 0:
        reqs = []
        for w in workers_used:
            
            # Get row and column indices for this worker
            wrow, wcol = block_map[w]
            
            # Start and end indices of block (handles edges)
            ii = wrow*block_rows
            jj = wcol*block_cols
            iip = ii + block_rows
            jjp = jj + block_cols
            if iip  > Amat.shape[0]: iip = Amat.shape[0]
            if jjp  > Amat.shape[1]: jjp = Amat.shape[1]
            
            if w == 0:
                # Block belongs to root worker
                my_Amat[:iip-ii,:jjp-jj] = Amat[ii:iip, jj:jjp]
                my_bvec[:jjp-jj] = bvec[jj:jjp]
                
            else:
                # Send blocks to other worker
                # Handles zero-padding of blocks at edge of matrix
                # FIXME: Do we have to copy here, to get contiguous memory?
                Amat_buf = np.zeros_like(my_Amat)
                bvec_buf = np.zeros_like(my_bvec)
                
                Amat_buf[:iip-ii,:jjp-jj] = Amat[ii:iip, jj:jjp]
                bvec_buf[:jjp-jj] = bvec[jj:jjp]
                
                # The flattened Amat_buf array is reshaped into 2D when received
                comm.Send(Amat_buf.flatten().copy(), dest=w)
                comm.Send(bvec_buf, dest=w)
                
        if verbose:
            print("All send operations completed.")
    else:
        # Receive this worker's assigned blocks of Amat and bvec
        comm.Recv(my_Amat, source=0)
        comm.Recv(my_bvec, source=0)
        
        if verbose:
            print("Worker %d finished receive" % myid)
    
    return my_Amat, my_bvec
    

def cg_mpi(comm_groups, Amat_block, bvec_block, vec_size, block_map, 
           maxiters=1000, abs_tol=1e-8):
    """
    A distributed CG solver for linear systems with square matrix operators. 
    The linear operator matrix is split into square blocks, each of which is 
    handled by a single worker.
    
    Parameters:
        comm_groups (tuple of MPI.Intracomm):
            Group communicators for the blocks (active, row, col, diag). 
            
            These are set up by `setup_mpi_blocks`, and correspond to the MPI 
            workers that are active, the ones for each row, each column, and 
            along the diagonal of the block structure, respectively.
            
            If `None`, this is assumed to be an inactive worker and nothing is 
            done.
        Amat_block (array_like):
            The block of the matrix operator belonging to this worker.
        bvec_block (array_like):
            The block of the right-hand side vector corresponding to this 
            worker's matrix operator block.
        vec_size (int):
            The size of the total result vector, across all blocks.
        block_map (dict):
            Dictionary of tuples, one for each worker in the `active` 
            communicator group, with the row and column ID of the block that it 
            is managing.
        maxiters (int):
            Maximum number of iterations of the solver to perform before 
            returning.
        abs_tol (float):
            Absolute tolerance on each element of the residual. Once this 
            tolerance has been reached for all entries of the residual vector, 
            the solution is considered to have converged.
    
    Returns:
        x (array_like):
            Solution vector for the full system. Only workers on the diagonal 
            have the correct solution vector; other workers will return `None`.
    """
    if comm_groups is None:
        # FIXME: Need to fix this so non-active workers are ignored without hanging
        return None
        
    comm_active, comm_row, comm_col, comm_diag = comm_groups
    #grp_active, grp_row, grp_col, grp_diag = groups
    myid = comm_active.Get_rank()
    myrow, mycol = block_map[myid]
    
    # Initialise solution vector
    x_block = np.zeros_like(bvec_block)
    
    # Calculate initial residual (if x0 is not 0, need to rewrite this, i.e.
    # be careful with x_block ordering)
    r_block = bvec_block[:] #- matvec_mpi(comm_row, Amat_block, x_block)
    pvec_block = r_block[:]
    
    # Iterate
    niter = 0
    finished = False
    while niter < maxiters and not finished:
        
        # Check convergence criterion from all workers
        converged = np.all(np.abs(r_block) < abs_tol)
        
        # Check if convergence is reached
        # (reduce with logical-AND operation)
        converged = comm_active.allreduce(converged, op=MPI_LAND)
        if converged:
            finished = True
            break
        
        # Distribute pvec_block to all workers (identical in each column)
        # NOTE: Assumes that the rank IDs in comm_col are in the same order as 
        # for the original comm_world communicator, in which case the rank with 
        # ID = mycol will be the one on the diagonal that we want to broadcast 
        # the up-to-date value of pvec_block from
        if myrow != mycol:
            pvec_block *= 0.
        comm_col.Bcast(pvec_block, root=mycol)
        
        # Calculate matrix operator product with p-vector (returns result for 
        # this row)
        A_dot_p = matvec_mpi(comm_row, Amat_block, pvec_block)
        
        # Only workers with mycol == myrow will give correct updates, so only 
        # calculate using those
        if mycol == myrow:
            # Calculate residual norm, summed across all (diagonal) workers
            r_dot_r = comm_diag.allreduce(np.dot(r_block.T, r_block), op=MPI_SUM)
            
            # Calculate quadratic, summed across all (diagonal) workers
            pAp = comm_diag.allreduce(np.dot(pvec_block.T, A_dot_p), op=MPI_SUM)
            
            # Calculate alpha (valid on all diagonal workers)
            alpha = r_dot_r / pAp
            
            # Update solution vector and residual for this worker
            x_block = x_block + alpha * pvec_block
            r_block = r_block - alpha * A_dot_p
            
            # Calculate updated residual norm
            rnew_dot_rnew = comm_diag.allreduce(np.dot(r_block.T, r_block), 
                                                op=MPI_SUM)
            
            # Calculate beta (valid on all diagonal workers)
            beta = rnew_dot_rnew / r_dot_r
            
            # Update pvec_block (valid on all diagonal workers)
            pvec_block = r_block + beta * pvec_block
        
        comm_active.barrier()
        
        # Increment iteration
        niter += 1
    
    # Gather all the blocks into a single array (on diagonal workers only)
    if myrow == mycol:
        x_all = np.zeros((vec_size), dtype=x_block.dtype)
        x_all_blocks = np.zeros((x_block.size * comm_diag.Get_size()), 
                                dtype=x_block.dtype) # needed for zero-padding
        comm_diag.Allgather(x_block, x_all_blocks)
        
        # Remove zero padding if necessary
        x_all[:] = x_all_blocks[:vec_size]
    else:
        x_all = None
    
    comm_active.barrier()
    return x_all


def cg(Amat, bvec, maxiters=1000, abs_tol=1e-8, use_norm_tol=False, linear_op=None):
    """
    Simple Conjugate Gradient solver that operates in serial. This uses the 
    same algorithm as `cg_mpi()` and so can be used for testing/comparison of 
    results.
    
    Note that this function will still permit threading used within numpy.
    
    Parameters:
        Amat (array_like):
            Linear operator matrix.
        bvec (array_like):
            Right-hand side vector.
        maxiters (int):
            Maximum number of iterations of the solver to perform before 
            returning.
        abs_tol (float):
            Absolute tolerance on each element of the residual. Once this 
            tolerance has been reached for all entries of the residual vector, 
            the solution is considered to have converged.    
        use_norm_tol (bool):
            Whether to use the tolerance on each element (as above), or an 
            overall tolerance on the norm of the residual.
        linear_op (func):
            If specified, this function will be used to operate on vectors, 
            instead of the Amat matrix. Must have call signature `func(x)`.
    
    Returns:
        x (array_like):
            Solution vector for the full system.
    """
    # Use Amat as the linear operator if function not specified
    if linear_op is None:
        linear_op = lambda v: Amat @ v
    
    # Initialise solution vector
    x = np.zeros_like(bvec)
    
    # Calculate initial residual
    r = bvec - linear_op(x)
    pvec = r[:]
    
    # Blocks indexed by i,j: y = A . x = Sum_j A_ij b_j
    niter = 0
    finished = False
    while niter < maxiters and not finished:
        
        try:
            # Check convergence criterion
            if use_norm_tol:
                # Check tolerance on norm of r
                if np.linalg.norm(r) < abs_tol:
                    finished = True
                    break
            else:
                # Check tolerance per array element
                if np.all(np.abs(r) < abs_tol):
                    finished = True
                    break

            # Do CG iteration
            r_dot_r = np.dot(r.T, r)
            A_dot_p = linear_op(pvec)
            
            pAp = pvec.T @ A_dot_p
            alpha = r_dot_r / pAp

            x = x + alpha * pvec
            r = r - alpha * A_dot_p

            # Update pvec
            beta = np.dot(r.T, r) / r_dot_r
            pvec = r + beta * pvec
            
            # Increment iteration
            niter += 1
        except:
            raise
        
    return x
