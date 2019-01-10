from numba import cuda, jit, float64, int32
import numpy as np
import math
BLOCK_SIZE=16
T_BLOCK_SIZE=BLOCK_SIZE*BLOCK_SIZE

@cuda.jit('void(float64[:,:], float64[:,:], float64[:,:], float64[:,:])')
def revert_kernel(density0, density1, energy0, energy1):
    col, row= cuda.grid(2)
    if col>1 and col<density0.shape[0]-2 and row>1 and row< density0.shape[1]:
        density1[col,row] = density0[col,row]
        energy1[col,row]= energy0[col,row]
