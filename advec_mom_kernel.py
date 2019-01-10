from numba import cuda, jit, float64, int32
import numpy as np
import math

@cuda.jit('void(int32, float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:])')
def advec_mom_vol_kernel(\
mom_sweep, post_vol, pre_vol, volume,vol_flux_x, vol_flux_y):

    col, row = cuda.grid(2)

    if row < volume.shape[1] and col < volume.shape[0]:
        if (mom_sweep == 1):

            post_vol[col,row] = volume[col,row] \
                + vol_flux_y[col,row+1] - vol_flux_y[col,row]
            pre_vol[col,row] = post_vol[col,row] \
                + vol_flux_x[col+1,row] - vol_flux_x[col,row]
        elif (mom_sweep == 2):
            post_vol[col,row] = volume[col,row]\
                    + vol_flux_x[col+1,row] - vol_flux_x[col,row]
            pre_vol[col,row] = post_vol[col,row]\
                + vol_flux_y[col,row+1] - vol_flux_y[col,row]
        elif (mom_sweep == 3):
            post_vol[col,row] = volume[col,row]
            pre_vol[col,row] = post_vol[col,row] \
                + vol_flux_y[col,row+1] - vol_flux_y[col,row]

        elif (mom_sweep == 4):
            post_vol[col,row] = volume[col,row]
            pre_vol[col,row] = post_vol[col,row]\
                + vol_flux_x[col+1,row] - vol_flux_x[col,row]


#///////////////////////////////////////////////////////////
#//x kernels

@cuda.jit('void(float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:])')
def advec_mom_node_flux_post_x_kernel( node_flux, node_mass_post,  mass_flux_x, post_vol, density1):

    col, row = cuda.grid(2)
    if row > 1 and  row  < density1.shape[1]-1 and  col < density1.shape[0]:

        node_flux[col,row] = 0.25\
            * (mass_flux_x[col,row-1] + mass_flux_x[col,row]\
            + mass_flux_x[col+1,row-1] + mass_flux_x[col+1,row])



    if row > 1 and row < density1.shape[1]-1 and col > 0  and col < density1.shape[0]:

            node_mass_post[col,row] = 0.25 \
                *(density1[col,row-1]  * post_vol[col,row-1] \
                + density1[col,row]   * post_vol[col,row] \
                + density1[col-1,row-1] * post_vol[col-1,row-1] \
                + density1[col-1,row]  * post_vol[col-1,row])


@cuda.jit('void(int32, int32, float64[:,:], float64[:,:], float64[:,:])')
def advec_mom_node_pre_x_kernel( x_max, y_max, node_flux, node_mass_post, node_mass_pre):

    col, row = cuda.grid(2)
    if row > 1 and row <= y_max + 2 and col >= 1  and col <= x_max + 3:

        node_mass_pre[col,row] = node_mass_post[col,row] \
            - node_flux[col-1,row] + node_flux[col,row]



@cuda.jit('void(int32, int32, float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:], float64[:,:])')
def advec_mom_flux_x_kernel(x_max, y_max, node_flux, node_mass_post, node_mass_pre, xvel1, celldx, mom_flux):

    col, row = cuda.grid(2)

    if row > 1 and row <= y_max + 2 and col >= 1 and col <= x_max + 2:
        if (node_flux[col,row] < 0.0):
            upwind = 2
            donor = 1
            downwind = 0
            dif = donor
        else:
            upwind = -1
            donor = 0
            downwind = 1
            dif = upwind

        sigma = math.fabs(node_flux[col,row]) / node_mass_pre[col+donor, row]
        vdiffuw = xvel1[col+donor,row] - xvel1[col+upwind, row]
        vdiffdw = xvel1[col+downwind,row] - xvel1[col+donor,row]
        limiter = 0.0

        if (vdiffdw * vdiffuw > 0.0):
            auw = math.fabs(vdiffuw)
            adw = math.fabs(vdiffdw)
            wind = 1.0
            if (vdiffdw <= 0.0): wind = -1.0
            width = celldx[col]
            limiter = wind * min(width * ((2.0 - sigma) * adw / width\
                + (1.0 + sigma) * auw / celldx[col + dif]) / 6.0, \
                min(auw, adw))

        advec_vel = xvel1[col+donor,row] + (1.0 - sigma) * limiter
        mom_flux[col,row] = advec_vel * node_flux[col,row]

@cuda.jit('void(uint32, uint32, float64[:,:], float64[:,:], float64[:,:], float64[:,:])')
def advec_mom_xvel_kernel(x_max, y_max, node_mass_post, node_mass_pre, mom_flux, xvel1):

    col, row = cuda.grid(2)



    if row > 1 and row <= y_max + 2 and  col > 1 and col <= x_max + 2:

        xvel1[col,row] = (xvel1[col,row]\
            * node_mass_pre[col,row] + mom_flux[col-1,row]\
            - mom_flux[col,row]) / node_mass_post[col,row]

#////////////////////////////////////////////////////////////
#//y kernels


@cuda.jit('void(float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:])')
def advec_mom_node_flux_post_y_kernel(node_flux, node_mass_post, mass_flux_y, post_vol,density1):

    col, row = cuda.grid(2)
    x_max = density1.shape[0]-4
    y_max = density1.shape[1]-4

    if row <= y_max + 3 and col > 1 and col <= x_max  + 2:

        node_flux[col,row] = 0.25\
            * (mass_flux_y[col-1,row] + mass_flux_y[col,row]\
            + mass_flux_y[col-1,row+1] + mass_flux_y[col,row+1])

    if row >= 1  and row <= y_max + 3 and col > 1  and col <= x_max + 2:
        node_mass_post[col,row] = 0.25 \
            * (density1[col,row-1] * post_vol[col,row-1] \
            + density1[col,row]   * post_vol[col,row] \
            + density1[col-1,row-1] * post_vol[col-1,row-1] \
            + density1[(col-1,row)]  * post_vol[(col-1,row)])


@cuda.jit('void(int32, int32, float64[:,:], float64[:,:], float64[:,:])')
def advec_mom_node_pre_y_kernel( x_max, y_max, node_flux, node_mass_post, node_mass_pre):

    col, row = cuda.grid(2)

    if row >0  and  row <= y_max+3 and col > 1 and  col <= x_max + 2:

        node_mass_pre[col,row] = node_mass_post[col,row]\
            - node_flux[col,row-1] + node_flux[col,row]

@cuda.jit('void(int32, int32, float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:], float64[:,:])')
def advec_mom_flux_y_kernel(x_max, y_max,  node_flux,  node_mass_post,  node_mass_pre, yvel1, celldy, mom_flux):

    col, row = cuda.grid(2)

    if row > 0 and row <= y_max + 2 and col > 1 and col <= x_max + 2:
        if (node_flux[col,row] < 0.0):
            upwind = 2
            donor = 1
            downwind = 0
            dif = donor

        else:
            upwind = -1
            donor = 0
            downwind = 1
            dif = upwind

        sigma = math.fabs(node_flux[col,row]) / node_mass_pre[col,row+donor]
        vdiffuw = yvel1[col,row+donor] - yvel1[col,row+upwind]
        vdiffdw = yvel1[col,row+downwind] - yvel1[col,row+donor]
        limiter = 0.0

        if (vdiffdw * vdiffuw > 0.0):
            auw = math.fabs(vdiffuw)
            adw = math.fabs(vdiffdw)
            if (vdiffdw <= 0.0): wind = -1.0
            else:  wind = 1.0

            width = celldy[row]
            limiter = wind * min(width * ((2.0 - sigma) * adw / width \
                + (1.0 + sigma) * auw / celldy[row + dif]) / 6.0,
                min(auw, adw))

        advec_vel = yvel1[col,row+donor] + (1.0 - sigma) * limiter
        mom_flux[col,row] = advec_vel * node_flux[col,row]


@cuda.jit('void(int32, int32, float64[:,:], float64[:,:], float64[:,:], float64[:,:])')
def advec_mom_yvel_kernel(x_max, y_max, node_mass_post, node_mass_pre,mom_flux, yvel1):

    col, row = cuda.grid(2)

    if (row > 1) and  row <= y_max + 2 and  col > 1 and col <= x_max + 2:

        yvel1[col,row] = (yvel1[col,row]\
            * node_mass_pre[col,row] + mom_flux[col,row-1]\
            - mom_flux[col,row]) / node_mass_post[col,row]
