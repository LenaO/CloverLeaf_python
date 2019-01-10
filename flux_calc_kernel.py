from numba import cuda, jit, float64, int32
import numpy as np
import math
BLOCK_SIZE=16
T_BLOCK_SIZE=BLOCK_SIZE*BLOCK_SIZE

@cuda.jit('void(int32, int32, float64, float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:])')
def flux_calc_kernel(x_max, y_max, dt, xarea, yarea, xvel0, yvel0, xvel1, yvel1, vol_flux_x, vol_flux_y):

    col, row = cuda.grid(2)

    if row > 1  and row <= (y_max + 1) and col > 1 and col <= x_max + 2:
        vol_flux_x[col,row] = 0.25 * dt * xarea[col,row] \
            * (xvel0[col,row] + xvel0[col,row+1] \
            + xvel1[col,row] + xvel1[col, row+1])


    if row > 1 and row <= y_max + 2 and col > 1 and col <= (x_max + 1):
        vol_flux_y[col,row] = 0.25 * dt * yarea[col,row]\
            * (yvel0[col,row] + yvel0[col+1,row]\
            + yvel1[col,row] + yvel1[col+1,row])
