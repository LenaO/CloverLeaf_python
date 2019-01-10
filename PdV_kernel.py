from numba import cuda, jit, float64, int32
import numpy as np
import math
BLOCK_SIZE=16
T_BLOCK_SIZE=BLOCK_SIZE*BLOCK_SIZE

@cuda.jit('void(float64,int32[:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:],\
        float64[:,:], float64[:,:],  float64[:,:],  float64[:,:],  float64[:,:],  float64[:,:])')
def PdV_kernel_predict( dt, error_condition,  xarea,  yarea,  volume,  density0,  density1,  energy0,  energy1,\
 pressure, viscosity, xvel0, yvel0, xvel1, yvel1):

    col, row= cuda.grid(2)

    err_condition_shared = cuda.shared.array(shape=T_BLOCK_SIZE, dtype=int32)


    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    lid= tx+ty*BLOCK_SIZE

    err_condition_shared[lid] = 0

    if row > 1 and row <  volume.shape[1] - 2 and  col > 1  and col < volume.shape[0] - 2:

        left_flux   = (xarea[col,row] \
            * (xvel0[col,row] + xvel0[col,row+1] \
            + xvel0[col,row] + xvel0[col, row+1])) \
            * 0.25 * dt * 0.5
        right_flux  = (xarea[col+1,row] \
            * (xvel0[col+1,row] + xvel0[col+1,row+1]\
            + xvel0[col+1,row] + xvel0[col+1,row+1])) \
            * 0.25 * dt * 0.5

        bottom_flux = (yarea[col,row] \
            * (yvel0[col,row] + yvel0[col+1, row] \
            + yvel0[col,row] + yvel0[col+1,row]))\
            * 0.25 * dt * 0.5
        top_flux    = (yarea[col,row+1]\
            * (yvel0[col, row+1] + yvel0[col+1,row+1] \
            + yvel0[col,row+1] + yvel0[col+1,row+1])) \
            * 0.25 * dt * 0.5

        total_flux = right_flux - left_flux + top_flux - bottom_flux

        volume_change = volume[col, row] \
            / (volume[col, row] + total_flux)

        #minimum of total, horizontal, and vertical flux
        min_cell_volume = \
            min(volume[col,row] + total_flux, \
            min(volume[col,row] + right_flux - left_flux, \
                volume[col,row] + top_flux - bottom_flux))

        if volume_change <= 0.0:

            err_condition_shared[lid] = 1

        if min_cell_volume <= 0.0:

            err_condition_shared[lid] = 2


        recip_volume = 1.0/volume[col,row]

        energy_change = ((pressure[col,row] / density0[ col,row]) \
            + (viscosity[col,row] / density0[col,row])) \
            * total_flux * recip_volume

        energy1[col,row] = energy0[col,row] - energy_change
        density1[col,row] = density0[col,row] * volume_change
        offset =int(T_BLOCK_SIZE/2)
    cuda.syncthreads()
    while offset > 0:
        if lid < offset:
            err_condition_shared[lid]=max(err_condition_shared[lid],err_condition_shared[lid+offset])
        cuda.syncthreads()
        offset = int(offset/2)

    block_id = cuda.blockIdx.x + cuda.blockIdx.y*cuda.gridDim.x
    if(lid==0):
        error_condition[block_id] = err_condition_shared[0]



@cuda.jit('void(float64,int32[:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:],\
        float64[:,:], float64[:,:],  float64[:,:],  float64[:,:],  float64[:,:],  float64[:,:])')
def PdV_kernel_not_predict( dt, error_condition,  xarea,  yarea,  volume,  density0,  density1,  energy0,  energy1,\
 pressure, viscosity, xvel0, yvel0, xvel1, yvel1):

    col, row= cuda.grid(2)

    err_condition_shared = cuda.shared.array(shape=T_BLOCK_SIZE, dtype=int32)


    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    lid= tx+ty*BLOCK_SIZE
    err_condition_shared[lid]=0

    if row > 1 and row <  volume.shape[1] - 2 and  col > 1  and col < volume.shape[0] - 2:
        left_flux   = (xarea[col, row] \
            * (xvel0[col, row] + xvel0[col, row+1] \
            + xvel1[col, row] + xvel1[col, row+1]))\
            * 0.25 * dt
        right_flux  = (xarea[col+1, row]
            * (xvel0 [col+1, row] + xvel0[col+1,row+1] \
            + xvel1[col+1, row] + xvel1[col+1,row+1])) \
            * 0.25 * dt

        bottom_flux = (yarea[col, row] \
            * (yvel0[col, row] + yvel0[col+1, row] \
            + yvel1[col, row] + yvel1[col+1, row])) \
            * 0.25 * dt
        top_flux    = (yarea[col, row+1]\
            * (yvel0[col, row+1] + yvel0[col+1,row+1] \
            + yvel1[col, row+1] + yvel1[col+1,row+1]))\
            * 0.25 * dt

        total_flux = right_flux - left_flux + top_flux - bottom_flux

        volume_change = volume[col, row] \
            / (volume[col, row] + total_flux)


        min_cell_volume = \
            min(volume[col, row] + total_flux, \
            min(volume[col, row] + right_flux - left_flux,\
                volume[col, row] + top_flux - bottom_flux))

        if (volume_change <= 0.0):
            err_condition_shared[lid] = 1

        if (min_cell_volume <= 0.0):
            err_condition_shared[lid] = 2


        recip_volume = 1.0/volume[col, row]

        energy_change = ((pressure[col, row]/ density0[col, row]) \
            + (viscosity[col, row] / density0[col, row])) \
            * total_flux * recip_volume

        energy1[col, row] = energy0[col, row] - energy_change
        density1[col, row] = density0[col, row] * volume_change

        offset = int(T_BLOCK_SIZE/2)
    cuda.syncthreads()
    while offset > 0:
        if lid < offset:
            err_condition_shared[lid]=max(err_condition_shared[lid],err_condition_shared[lid+offset])
        cuda.syncthreads()
        offset = int(offset/2)

    block_id = cuda.blockIdx.x + cuda.blockIdx.y*cuda.gridDim.x
    if(lid==0):
        error_condition[block_id] = err_condition_shared[0]
