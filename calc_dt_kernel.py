from numba import cuda, jit, float64, int32
import numpy as np
import math
BLOCK_SIZE=16
T_BLOCK_SIZE=BLOCK_SIZE*BLOCK_SIZE

@cuda.jit('void(float64,float64, float64,float64, float64, float64, float64,\
        float64[:,:], float64[:,:], float64[:], float64[:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:],\
        float64[:], float64[:])')
def calc_dt_kernel(\
g_small, g_big, dtmin, dtc_safe, dtu_safe, dtv_safe, dtdiv_safe, \
xarea,  yarea, celldx, celldy, volume, density0, viscosity, soundspeed, xvel0, yvel0, jk_ctrl_out, dt_min_out):

    col,row = cuda.grid(2)
    dt_min_shared =  cuda.shared.array (shape = T_BLOCK_SIZE, dtype=float64)
    jk_ctrl_shared =  cuda.shared.array (shape = T_BLOCK_SIZE, dtype=float64)
    jk_shared_tmp = cuda.shared.array (shape = T_BLOCK_SIZE, dtype=int32)
    dt_min_val = g_big
    jk_control = 0.0


    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    lid= tx+ty*BLOCK_SIZE

    dt_min_shared[lid] = dt_min_val
    jk_ctrl_shared[lid] = jk_control
    jk_shared_tmp[lid] = col+ (volume.shape[0]+1)*row
    if row > 1 and row < volume.shape[1]-2 and col > 1 and col < volume.shape[0]-2:
        dsx = celldx[col]
        dsy = celldy[row]

        cc = soundspeed[col,row] * soundspeed[col,row]
        cc += 2.0 * viscosity[col,row] / density0[col,row]
        cc = max(math.sqrt(cc), g_small)

        dtct = dtc_safe * min(dsx, dsy)/cc

        div = 0.0

        # x
        dv1 = (xvel0[col,row] + xvel0[col,row+1])\
            * xarea[col,row]
        dv2 = (xvel0[col+1, row] + xvel0[col+1,row+1])\
            * xarea[col+1,row]

        div += dv2 - dv1

        dtut = dtu_safe * 2.0 * volume[col, row]\
            / max(g_small*volume[col, row],\
            max(math.fabs(dv1), math.fabs(dv2)))

        # y
        dv1 = (yvel0[col,row] + yvel0[col+1,row]) \
            * yarea[col,row]
        dv2 = (yvel0[col,row+1] + yvel0[col+1,row+1]) \
            * yarea[col, row+1]

        div += dv2 - dv1

        dtvt = dtv_safe * 2.0 * volume[col, row] \
            / max(g_small*volume[col,row], \
            max(math.fabs(dv1), math.fabs(dv2)))

        #
        div /= (2.0 * volume[col,row])
        if div < (-g_small): dtdivt= dtdiv_safe * (-1.0/div)
        else: dtdivt = g_big

        dt_min_shared[lid] = min(dtdivt, min(dtvt, min(dtct, dtut)))

        jk_ctrl_shared[lid] = (col + ((volume.shape[0]-4) * (row - 1))) + 0.4

    cuda.syncthreads()
    offset=int(T_BLOCK_SIZE/2)
    while offset > 0:
        if lid < offset:
            if(dt_min_shared[lid]>dt_min_shared[lid+offset]):
                jk_shared_tmp[lid]=jk_shared_tmp[lid+offset]
            dt_min_shared[lid]= min(dt_min_shared[lid],dt_min_shared[lid + offset])
            jk_ctrl_shared[lid] =max(jk_ctrl_shared[lid],jk_ctrl_shared[lid + offset])

        cuda.syncthreads()
        offset = int(offset/2)

    #if(lid==0):
     #   print(vol_shared[0])
    block_id = cuda.blockIdx.x + cuda.blockIdx.y*cuda.gridDim.x
    if(lid==0):
        jk_ctrl_out[block_id]=jk_ctrl_shared[0]
        dt_min_out[block_id]= dt_min_shared[0]

        #if(dt_min_shared[0] <= 0.0):
        #    print ("error2")

@cuda.reduce
def max_reduce(a, b):
    return max(a,b)

@cuda.reduce
def min_reduce(a, b):
    return min(a,b)
