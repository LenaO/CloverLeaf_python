from numba import cuda, jit, float64, int32
import numpy as np
import math
@cuda.jit('void(float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:])')
def reset_field_kernel(density0, density1, energy0, energy1, xvel0, xvel1, yvel0, yvel1):
    col, row = cuda.grid(2)
    if row >1 and row < energy0.shape[1]-1 and col > 1 and col < energy0.shape[0]-1:
        xvel0[col, row] = xvel1[col,row]
        yvel0[col,row] = yvel1[col,row]

        if row <= energy0.shape[1]-3 and col <= energy0.shape[0]-3:
            density0[col,row] = density1[col,row]
            energy0[col,row]  = energy1[col,row]
