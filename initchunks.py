from numba import cuda, jit, float64, int32
import numpy as np

@cuda.jit('void(float64,float64, float64, float64, float64[:],float64[:], float64[:], float64[:])')
def initialise_chunk_vertex(d_xmin, d_ymin, d_dx, d_dy, vertexx, vertexdx, vertexy, vertexdy):
    column,row  =  cuda.grid(2)

    #row = int(glob_id /(vertexx.shape[0]+4))
    #column = int(glob_id % (vertexx.shape[0]+4))
    #fill out x arrays
    if (row == 0  and column<vertexx.shape[0]):
        vertexx[column] = d_xmin + d_dx * float64(column-2)
        vertexdx[column] = d_dx

    #fill out y arrays
    if (column == 0 and row<vertexy.shape[0]):
        vertexy[row] = d_ymin + d_dy * float64(row - 2)
        vertexdy[row] = d_dy


@cuda.jit('void(float64, float64, float64[:], float64[:],float64[:],float64[:], float64[:], float64[:], float64[:,:], float64[:,:], float64[:,:])' )
def initialise_chunk(d_dx, d_dy, vertexx, vertexy, cellx, celldx, celly, celldy, volume, xarea, yarea):

    glob_x, glob_y  =  cuda.grid(2)

    if (glob_y==0 and glob_x<cellx.shape[0]):

        cellx[glob_x] = 0.5 * (vertexx[glob_x] + vertexx[glob_x + 1])
        celldx[glob_x] = d_dx
#
#
#    #fill y arrays
    if (glob_x == 0 and glob_y<celly.shape[0] ):
#
        celly[glob_y] = 0.5 * (vertexy[glob_y] + vertexy[glob_y + 1])
        celldy[glob_y] = d_dy
#
#

    if glob_y < volume.shape[1] and  glob_x< volume.shape[0]:
        volume[glob_x][glob_y] = d_dx * d_dy
        xarea[glob_x][glob_y] = d_dy
        yarea[glob_x][glob_y] = d_dx
#
