from numba import cuda, jit, float64, int32
import numpy as np
import math
@cuda.jit('void(float64[:], float64[:], float64[:], float64[:],\
        float64[:,:], float64[:,:], float64[:,:], float64[:,:], \
        float64, float64, float64, float64, float64, float64, float64, float64, float64,\
        int32, int32, int32, int32)')
def generate_chunk_kernel( vertexx, vertexy, cellx, celly, density0, energy0, xvel0, yvel0, \
state_density, state_energy, state_xvel, state_yvel, state_xmin, state_xmax, state_ymin, state_ymax, state_radius,\
state_geometry,  g_rect, g_circ, g_point):

    column,row =  cuda.grid(2)
    if row == 0 and column ==0:
        print(state_geometry, g_rect, state_xmin, state_xmax, state_ymin, state_ymax, state_energy)
    if row < density0.shape[1]   and column < density0.shape[0]:

        x_cent = state_xmin
        y_cent = state_ymin

        if g_rect == state_geometry:

            if vertexx[1 + column]>=state_xmin and vertexx[column]<state_xmax and vertexy[1 + row]>=state_ymin and vertexy[row]< state_ymax:

                energy0[column, row] = state_energy
                density0[column, row] = state_density

                #unrolled do loop
                xvel0[column, row] = state_xvel
                yvel0[column, row] = state_yvel

                xvel0[column+1, row] = state_xvel
                yvel0[column+1, row] = state_yvel

                xvel0[column, row+1] = state_xvel
                yvel0[column, row+1] = state_yvel

                xvel0[column+1, row+1] = state_xvel
                yvel0[column+1, row+1] = state_yvel

        elif (state_geometry == g_circ):

            x_pos = cellx[column]-x_cent
            y_pos = celly[row]-y_cent
            radius = math.sqrt(x_pos*x_pos + y_pos*y_pos)

            if (radius <= state_radius):

                energy0[column, row] = state_energy
                density0[column, row] = state_density

            #unrolled do loop
                xvel0[column,row] = state_xvel
                yvel0[column,row] = state_yvel

                xvel0[column+1,row] = state_xvel
                yvel0[column+1,row] = state_yvel

                xvel0[column,row+1] = state_xvel
                yvel0[column,row+1] = state_yvel

                xvel0[column+1,row+1] = state_xvel
                yvel0[column+1,row+1] = state_yvel


        elif state_geometry == g_point:
            if (vertexx[column] == x_cent and vertexy[row] == y_cent):

                energy0[column, row] = state_energy
                density0[column, row] = state_density

                #unrolled do loop
                xvel0[column, row] = state_xvel
                yvel0[column, row] = state_yvel

                xvel0[column+1, row] = state_xvel
                yvel0[column+1, row] = state_yvel

                xvel0[column,row+1]  = state_xvel
                yvel0[column,row+1]  = state_yvel

                xvel0[column+1,row+1] = state_xvel
                yvel0[column+1,row+1] = state_yvel


@cuda.jit('void(float64[:,:], float64[:,:], float64[:,:], float64[:,:],\
        float64, float64, float64, float64)')
def generate_chunk_kernel_init(density0,energy0, xvel0, yvel0,\
  state_density, state_energy, state_xvel, state_yvel):

    column , row =  cuda.grid(2)
    if row < density0.shape[1] and column < density0.shape[0]:
        energy0[column,row] = state_energy
        density0[column, row] = state_density
        xvel0[column, row] = state_xvel
        yvel0[column, row] = state_yvel
