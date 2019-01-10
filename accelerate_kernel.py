from numba import cuda, jit, float64, int32
import numpy as np
import math
BLOCK_SIZE=16
T_BLOCK_SIZE=BLOCK_SIZE*BLOCK_SIZE

@cuda.jit('void(float64, float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:],\
        float64[:,:], float64[:,:],  float64[:,:])')
def accelerate_kernel(dbyt, xarea, yarea, volume, density0, pressure, viscosity, xvel0, yvel0, xvel1, yvel1):

    col, row= cuda.grid(2)

    # prevent writing to *vel1, then read from it, then write to it again

    if row > 1 and row <  volume.shape[1] - 1 and  col > 1  and col < volume.shape[0] - 1:
        nodal_mass = \
            (density0[col-1,row-1] * volume[col-1,row-1] \
            + density0[col, row-1] * volume[col,row-1]\
            + density0[col,row] * volume[col,row]\
            + density0[col-1,row] * volume[col-1, row])\
            * 0.25

        step_by_mass = 0.5 * dbyt / nodal_mass

        # x velocities
        xvel_temp = xvel0[col,row] - step_by_mass \
            * (xarea[col,row] * (pressure[col,row]  - pressure[col-1,row])\
            + xarea[col,row-1] * (pressure[col,row-1] - pressure[col-1,row-1]))

        xvel1[col,row]  = xvel_temp - step_by_mass \
            * (xarea[col,row]  * (viscosity[col,row]  - viscosity[col-1,row])\
            + xarea[col,row-1] * (viscosity[col,row-1] - viscosity[col-1,row-1]))

        # y velocities
        yvel_temp = yvel0[col,row]  - step_by_mass \
            * (yarea[col,row]  * (pressure[col,row]  - pressure[col,row-1])
            + yarea[col-1, row] * (pressure[col-1,row] - pressure[col-1,row-1]))

        yvel1[col,row]  = yvel_temp - step_by_mass \
            * (yarea[col,row]  * (viscosity[col,row]  - viscosity[col,row-1]) \
            + yarea[col-1,row] * (viscosity[col-1,row] - viscosity[col-1,row-1]))
