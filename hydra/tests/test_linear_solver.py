
import unittest

import numpy as np
from hydra import linear_solver

class TestLinearSolver(unittest.TestCase):

    def test_cg(self):

        # Set up example linear system with random solution
        np.random.seed(1)
        Ndim = 401 # numerical errors will build up with higher dimensions
        Amat1 = np.eye(Ndim)
        xtrue = np.random.randn(Amat1.shape[0])
        bvec1 = Amat1 @ xtrue

        # Trivial diagonal matrix operator
        xsoln1 = linear_solver.cg(Amat1, 
                                  bvec1, 
                                  maxiters=1000, 
                                  abs_tol=1e-8, 
                                  use_norm_tol=False, 
                                  x0=None, 
                                  linear_op=None, 
                                  comm=None)
        self.assertTrue(np.allclose(xtrue, xsoln1))

        # Slightly more difficult operator
        Amat2 = np.eye(xtrue.size) + 0.1 * np.random.randn(*Amat1.shape)
        Amat2 = 0.5 * (Amat2 + Amat2.T) # symmetrise
        bvec2 = Amat2 @ xtrue
        xsoln2 = linear_solver.cg(Amat2, 
                                  bvec2, 
                                  maxiters=1000, 
                                  abs_tol=1e-8, 
                                  use_norm_tol=False, 
                                  x0=None, 
                                  linear_op=None, 
                                  comm=None)
        self.assertTrue(np.allclose(xtrue, xsoln2))

        # Test with linear_op function instead of explicit matrix operator
        linear_op = lambda x: Amat2 @ x
        xsoln2a = linear_solver.cg(None, 
                                   bvec2, 
                                   maxiters=1000, 
                                   abs_tol=1e-8, 
                                   use_norm_tol=False, 
                                   x0=None, 
                                   linear_op=linear_op, 
                                   comm=None)
        self.assertTrue(np.allclose(xtrue, xsoln2a))
        self.assertTrue(np.allclose(xsoln2, xsoln2a))


    """
    matvec_mpi(comm_row, mat_block, vec_block)
    setup_mpi_blocks(comm, matrix_shape, split=1)
    collect_linear_sys_blocks(comm, block_map, block_shape, Amat=None, bvec=None, verbose=False)
    cg_mpi(comm_groups, Amat_block, bvec_block, vec_size, block_map, maxiters=1000, abs_tol=1e-8)
    """