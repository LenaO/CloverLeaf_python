from numba import cuda, jit, float64, int32
import numpy as np
import math
@cuda.jit('void(float64[:], float64[:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:])')
def viscosity_kernel(\
        celldx, celldy, density0, pressure, viscosity, xvel0, yvel0):

    col, row= cuda.grid(2)
    x_max= density0.shape[0]-4

    if row > 1 and row < density0.shape[1]-2 and col > 1  and col < density0.shape[0]-2:
        ugrad = (xvel0[col+1, row] + xvel0[col+1, row+1])\
              - (xvel0[col, row] + xvel0[col, row+1])

        vgrad = (yvel0[col, row+1] + yvel0[col+1, row+1])\
              - (yvel0[col,row] + yvel0[col+1, row])

        div = (celldx[col] * ugrad) + (celldy[row] * vgrad)

        strain2 = 0.5 * (xvel0[col, row+1] + xvel0[col+1, row+1]\
                - xvel0[col,row] - xvel0[col+1, row])/celldy[row]\
                + 0.5 * (yvel0[col+1, row] + yvel0[col+1, row+1]\
                - yvel0[col, row] - yvel0[col,row+1])/celldx[col]

        pgradx = (pressure[col+1, row] - pressure[col-1, row])\
               / (celldx[col] + celldx[col + 1])
        pgrady = (pressure[col, row+1] - pressure[col, row-1])\
               / (celldy[row] + celldy[row + 1])

        pgradx2 = pgradx*pgradx
        pgrady2 = pgrady*pgrady

        limiter = ((0.5 * ugrad / celldx[col]) * pgradx2\
                + ((0.5 * vgrad / celldy[row]) * pgrady2)\
                + (strain2 * pgradx * pgrady))\
                / max(pgradx2 + pgrady2, 1.0e-16)

        if (limiter > 0 or div >= 0.0):

            viscosity[col,row] = 0.0

        else:

          pgradx = math.copysign(max(1.0e-16, math.fabs(pgradx)), pgradx)
          pgrady = math.copysign(max(1.0e-16, math.fabs(pgrady)), pgrady)
          pgrad = math.sqrt((pgradx * pgradx) + (pgrady * pgrady))

          xgrad = math.fabs(celldx[col] * pgrad / pgradx)
          ygrad = math.fabs(celldy[row] * pgrad / pgrady)

          grad = min(xgrad, ygrad)
          grad2 = grad * grad

          viscosity[col,row] = 2.0 * density0[col,row] * grad2 * (limiter * limiter)
