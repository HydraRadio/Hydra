
from mpi4py.MPI import SUM as MPI_SUM
import numpy as np


def matvec_mpi(comm_row, mat_block, vec_block):
    """
    Do matrix-vector product and broadcast result to each worker in the row.
    """
    # Do matrix-vector product for the available blocks
    y = mat_block @ vec_block
    
    # Do reduce to all members of this column group
    ytot = np.zeros_like(y)
    comm_row.Allreduce(y, ytot, op=MPI_SUM)
    return ytot


def setup_mpi_blocks(comm, matrix_shape, split=1):
    """
    Assign blocks of the linear operator matrix to each available worker.
    
    Enforces square split, since cg_mpi assumes square matrices and blocks.
    """
    assert len(matrix_shape) == 2, "'matrix_shape' must be a tuple with 2 entries"
    myid = comm.Get_rank()
    nworkers = comm.Get_size()
    comm_row = None # comm group for each row
    
    # Check that enough workers are available
    nblocks = split * split
    assert nworkers >= nblocks, "Specified more blocks than workers"
    workers = np.arange(nblocks).reshape((split, split))
    
    # Handle workers that do not have an assigned block
    if myid >= nblocks:
        myrow, mycol = None, None
        return None, None, None, None, {}
    
    # Setup communicator group for each row    
    myrow, mycol = np.where(workers == myid)
    myrow = myrow[0]
    mycol = mycol[0]
    
    grp_active = workers.flatten()
    grp_row = workers[myrow,:]
    grp_col = workers[:,mycol]
    grp_diag = np.diag(workers)
    comm_active = comm.Create( comm.group.Incl(grp_active) )
    comm_row = comm.Create( comm.group.Incl(grp_row) )
    comm_col = comm.Create( comm.group.Incl(grp_col) )
    comm_diag = comm.Create( comm.group.Incl(grp_diag) )
    
    # Share out row and column indices in blocks
    # FIXME: This could be done more efficiently than using array_split
    rowidxs = np.arange(matrix_shape[0])
    colidxs = np.arange(matrix_shape[1])
    rowidx_blocks = np.array_split(rowidxs, split)
    colidx_blocks = np.array_split(colidxs, split)
    
    # Dictionary of operator matrix blocks belonging to each worker
    matrix_blocks = {}
    for w in workers.flatten():
        _row, _col = np.where(workers == w)
        matrix_blocks[w] = ((rowidx_blocks[_row[0]].min(), 
                             rowidx_blocks[_row[0]].max()), 
                            (colidx_blocks[_col[0]].min(),
                             colidx_blocks[_col[0]].max()))
    
    comms = (comm_active, comm_row, comm_col, comm_diag)
    groups = (grp_active, grp_row, grp_col, grp_diag)
    return comms, groups, myrow, mycol, matrix_blocks


def collect_linear_sys_blocks(comm, matrix_blocks, Amat=None, bvec=None, 
                              dtype=np.float64, verbose=False):
    """
    Send LHS operator matrix and RHS vector blocks to assigned workers.
    """
    myid = comm.Get_rank()
    
    # Determine whether this worker is participating
    workers_used = list(matrix_blocks.keys())
    if myid not in workers_used:
        return None, None
    
    # Initialise blocks of A matrix and b vector
    row_range, col_range = matrix_blocks[myid]
    my_Amat = np.zeros((row_range[1] - row_range[0] + 1, 
                        col_range[1] - col_range[0] + 1), dtype=dtype)
    my_bvec = np.zeros((row_range[1] - row_range[0] + 1,), dtype=dtype)
    
    # Send blocks from root worker
    if myid == 0:
        reqs = []
        for w in workers_used:
            if w == 0:
                # Block belongs to root worker
                my_Amat[:,:] = Amat[row_range[0]:row_range[1]+1,
                                    col_range[0]:col_range[1]+1]
                my_bvec[:] = bvec[col_range[0]:col_range[1]+1]
                
            else:
                # Send blocks to other worker
                rowr, colr = matrix_blocks[w]
                # FIXME: Do we have to copy here, to get contiguous memory?
                Amat_block = Amat[rowr[0]:rowr[1]+1, 
                                  colr[0]:colr[1]+1].flatten().copy()
                comm.Send(Amat_block, 
                          dest=w)
                comm.Send(bvec[colr[0]:colr[1]+1], dest=w)
        if verbose:
            print("All send operations completed.")
    else:
        comm.Recv(my_Amat, source=0)
        comm.Recv(my_bvec, source=0)
        
        if verbose:
            print("Worker %d finished receive" % myid)
    
    return my_Amat, my_bvec
    

def cg_mpi(comm, comm_groups, groups, Amat_block, bvec_block, vec_size,
           myrow, mycol, maxiters=1000, abs_tol=1e-8):
    """
    Distributed CG solver.
    """
    if comm_groups is None:
        # FIXME: Need to fix this so non-active workers are ignored without hanging
        return None
        
    comm_active, comm_row, comm_col, comm_diag = comm_groups
    grp_active, grp_row, grp_col, grp_diag = groups
    myid = comm_active.Get_rank()
    
    # Initialise solution vector
    x_block = np.zeros_like(bvec_block)
    
    # Calculate initial residual (if x is not 0, need to rewrite this, i.e.
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
        converged = comm_active.allreduce(converged, op=MPI.LAND)
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
    x_all = np.zeros((vec_size), dtype=x_block.dtype)
    if myrow == mycol:
        comm_diag.Allgather(x_block, x_all)
    comm_active.barrier()
    
    return x_all


def cg(Amat, bvec, maxiters=1000, abs_tol=1e-8):
    """
    Distributed CG solver.
    """
    # Initialise solution vector
    x = np.zeros_like(bvec)
    
    # Calculate initial residual
    r = bvec - Amat @ x
    pvec = r[:]
    
    # Blocks indexed by i,j: y = A . x = Sum_j A_ij b_j
    niter = 0
    finished = False
    while niter < maxiters and not finished:
        
        # Check convergence criterion
        if np.all(np.abs(r) < abs_tol):
            finished = True
            break

        # Do CG iteration
        r_dot_r = np.dot(r.T, r)
        A_dot_p = Amat @ pvec
        
        pAp = pvec.T @ A_dot_p
        alpha = r_dot_r / pAp

        x = x + alpha * pvec
        r = r - alpha * (Amat @ pvec)

        # Update pvec
        beta = np.dot(r.T, r) / r_dot_r
        pvec = r + beta * pvec
        
        # Increment iteration
        niter += 1
        
    return x
