from numba import cuda, jit, float64, int32
import numpy as np
import math

BLOCK_SIZE=16
T_BLOCK_SIZE=BLOCK_SIZE*BLOCK_SIZE

@cuda.jit('void(float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], \
        float64[:], float64[:], float64[:], float64[:], float64[:])')
def field_summary_kernel(volume, density0, energy0, pressure, xvel0, yvel0, vol, mass, ie, ke, press):
    vol_shared =  cuda.shared.array (shape = T_BLOCK_SIZE, dtype=float64)
    mass_shared = cuda.shared.array (shape = T_BLOCK_SIZE, dtype=float64)
    ie_shared = cuda.shared.array  (shape = T_BLOCK_SIZE, dtype=float64)
    ke_shared = cuda.shared.array  (shape = T_BLOCK_SIZE, dtype=float64)
    press_shared = cuda.shared.array (shape = T_BLOCK_SIZE, dtype=float64)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    lid= tx+ty*BLOCK_SIZE
    vol_shared[lid] = 0.0
    mass_shared[lid] = 0.0
    ie_shared[lid] = 0.0
    ke_shared[lid] = 0.0
    press_shared[lid] = 0.0

    col,row =  cuda.grid(2)

    if row > 1 and row <  volume.shape[1] - 2 and  col > 1  and col < volume.shape[0] - 2:
        vsqrd = 0.0

        #unrolled do loop
        vsqrd += 0.25 * (xvel0[col,row] * xvel0[col,row]
                        +yvel0[col,row] * yvel0[col,row])

        vsqrd += 0.25 * (xvel0[col+1,row] * xvel0[col+1,row]
                        +yvel0[col+1, row] * yvel0[col+1, row])

        vsqrd += 0.25 * (xvel0[col, row+1] * xvel0[col, row+1]
                        +yvel0[col, row+1] * yvel0[col, row+1])

        vsqrd += 0.25 * (xvel0[col+1, row+1] * xvel0[col+1, row+1]
                        +yvel0[col+1, row+1] * yvel0[col+1, row+1])

        cell_vol = volume[col, row]
        cell_mass = cell_vol * density0[col,row];

        vol_shared[lid] = cell_vol
        mass_shared[lid] = cell_mass
        ie_shared[lid] = cell_mass * energy0[col,row]
        ke_shared[lid] = cell_mass * 0.5 * vsqrd
        press_shared[lid] = cell_vol * pressure[col,row]


    cuda.syncthreads()
    offset=int(T_BLOCK_SIZE/2)
    while offset > 0:
        if lid < offset:
            vol_shared[lid] += vol_shared[lid + offset]
            mass_shared[lid] += mass_shared[lid + offset]
            ie_shared[lid] += ie_shared[lid + offset]
            ke_shared[lid] += ke_shared[lid + offset]
            press_shared[lid] += press_shared[lid + offset]
        cuda.syncthreads()
        offset = int(offset/2)

    #if(lid==0):
     #   print(vol_shared[0])
    block_id = cuda.blockIdx.x + cuda.blockIdx.y*cuda.gridDim.x
    if(lid==0):
        vol[block_id] = vol_shared[0]
        mass[block_id] = mass_shared[0]
        ie[block_id] = ie_shared[0]
        ke[block_id] = ke_shared[0]
        press[block_id] = press_shared[0]



@cuda.reduce
def sum_reduce(a, b):
    return a + b
