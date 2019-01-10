from numba import cuda, jit, float64, int32
import numpy as np
import math
BLOCK_SIZE=16
T_BLOCK_SIZE=BLOCK_SIZE*BLOCK_SIZE

@cuda.jit('void(int32, float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:])')
def pre_vol_kernel_x (swp_nmbr, pre_vol,  post_vol,  volume, vol_flux_x,  vol_flux_y):

    col, row= cuda.grid(2)

    if  row <  volume.shape[1] and col < volume.shape[0]:

        if (swp_nmbr == 1):

            pre_vol[col,row] = volume[col,row] \
                +(vol_flux_x[col+1,row] - vol_flux_x[col,row]\
                + vol_flux_y[col,row+1] - vol_flux_y[col,row])
            post_vol[col,row] = pre_vol[col,row] \
                - (vol_flux_x[col+1,row] - vol_flux_x[col,row])

        else:

            pre_vol[col,row] = volume[col,row]\
                + vol_flux_x[col+1,row] - vol_flux_x[col,row]
            post_vol[col,row] = volume[col,row]

@cuda.jit('void(int32, float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:], float64[:,:])')
def ener_flux_kernel_x(swp_nmbr,volume, vol_flux_x, vol_flux_y, pre_vol, density1, energy1,\
        ener_flux, vertexdx, mass_flux_x):


    one_by_six = 1.0/6.0

    #
    #/ if cell is within x area:
    # +++++++++++++++++++++
    # +++++++++++++++++++++
    # ++xxxxxxxxxxxxxxxxxxx
    # +++++++++++++++++++++
    # +++++++++++++++++++++
    col, row= cuda.grid(2)

    if row > 1 and row <  volume.shape[1] - 2 and  col > 1  and col < volume.shape[0]:
          # if flowing right
        if (vol_flux_x[col,row] > 0.0):

            upwind = -2
            donor = -1
            downwind = 0
            dif = donor

        else:

            #  tries to get from below, unless it would be reading from a cell
            #  which would be off the right, in which case read from cur cell
            upwind = 1
            if (col == volume.shape[0]-1):
                upwind= 0
            donor = 0
            downwind = -1
            dif = upwind

        sigmat = math.fabs(vol_flux_x[col,row]) / pre_vol[col+donor, row]
        sigma3 = (1.0 + sigmat) * (vertexdx[col] / vertexdx[col + dif])
        sigma4 = 2.0 - sigmat

        diffuw = density1[col+donor,row ] - density1[col+upwind, row]
        diffdw = density1[col+downwind, row] - density1[col+donor, row]

        if (diffuw * diffdw > 0.0):

            limiter = (1.0 - sigmat) * math.copysign(1.0, diffdw) \
                * min(math.fabs(diffuw), min(math.fabs(diffdw), one_by_six\
                * (sigma3 * math.fabs(diffuw) + sigma4 * math.fabs(diffdw))))

        else:

            limiter = 0.0


        mass_flux_x[col,row] = vol_flux_x[col,row]\
            * (density1[col+donor, row] + limiter)

        sigmam = math.fabs(mass_flux_x[col,row]) \
            / (density1[col+donor, row] * pre_vol[col+donor, row])
        diffuw = energy1[col+donor, row] - energy1[col+upwind, row]
        diffdw = energy1[col+downwind, row] - energy1[col+donor, row]

        if (diffuw * diffdw > 0.0):

            limiter = (1.0 - sigmam) * math.copysign(1.0, diffdw) \
                * min(math.fabs(diffuw), min(math.fabs(diffdw), one_by_six \
                * (sigma3 * math.fabs(diffuw) + sigma4 * math.fabs(diffdw))))

        else:

            limiter = 0.0


        ener_flux[col,row] = mass_flux_x[col,row]\
            * (energy1[col+donor,row] + limiter)


@cuda.jit('void(int32, float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:])')
def advec_cell_kernel_x(swp_nmbr,volume, vol_flux_x, vol_flux_y, pre_vol, density1, energy1, ener_flux, mass_flux_x):

    col,row=cuda.grid(2)
   #
   #   if cell is within x area:
   #   +++++++++++++++++++++
   #   +++++++++++++++++++++
   #   ++xxxxxxxxxxxxxxxxx++
   #   +++++++++++++++++++++
   #   +++++++++++++++++++++
   #

    if row > 1 and row <  volume.shape[1] - 2 and  col > 1  and col < volume.shape[0] - 2:

        pre_mass = density1[col,row] * pre_vol[col,row]\

        post_mass = pre_mass + mass_flux_x[col,row]\
            - mass_flux_x[col+1,row]

        post_ener = (energy1[col,row] * pre_mass \
            + ener_flux[col,row] - ener_flux[col+1,row]) \
            / post_mass

        advec_vol = pre_vol[col,row] + vol_flux_x[col,row] \
            - vol_flux_x[col+1,row]

        density1[col,row] = post_mass / advec_vol
        energy1[col,row] = post_ener



#//////////////////////////////////////////////////////////////////////////
#//y kernels

@cuda.jit('void(int32, float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:])')
def pre_vol_kernel_y( swp_nmbr, pre_vol, post_vol, volume, vol_flux_x, vol_flux_y):

    col, row = cuda.grid(2)

    if row<volume.shape[1] and col <volume.shape[0]:
        if (swp_nmbr == 1):

            pre_vol[col,row] = volume[col,row]\
                +(vol_flux_y[col,row+1] - vol_flux_y[col,row]\
                + vol_flux_x[col+1,row] - vol_flux_x[col,row])
            post_vol[col,row] = pre_vol[col,row]\
                - (vol_flux_y[col,row+1] - vol_flux_y[col,row])

        else:

            pre_vol[col,row] = volume[col,row]\
                + vol_flux_y[col,row+1] - vol_flux_y[col,row]
            post_vol[col,row] = volume[col,row]


@cuda.jit('void(int32, float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:],float64[:,:], float64[:], float64[:,:])')
def ener_flux_kernel_y(swp_nmbr,volume, vol_flux_x, vol_flux_y, pre_vol, density1, energy1, ener_flux,\
        vertexdy,  mass_flux_y):

    col, row= cuda.grid(2)

    one_by_six = 1.0/6.0

   #
   #   if cell is within x area:
   #   +++++++++++++++++++++
   #   +++++++++++++++++++++
   #   ++xxxxxxxxxxxxxxxxx++
   #   ++xxxxxxxxxxxxxxxxx++
   #

    if row > 1 and row <  volume.shape[1]  and  col > 1  and col < volume.shape[0] - 2:
        #if flowing right
        if (vol_flux_y[col,row] > 0.0):

            upwind = -2
            donor = -1
            downwind = 0
            dif = donor

        else:
            #
            # tries to get from below, unless it would be reading from a cell
            # which would be off the bottom, in which case read from cur cell
            #
            upwind = 1
            if row == volume.shape[1]-1:
                upwind =  0
            donor = 0
            downwind = -1
            dif = downwind


        sigmat = math.fabs(vol_flux_y[col,row]) / pre_vol[col, row+donor]
        sigma3 = (1.0 + sigmat) * (vertexdy[row] / vertexdy[row + dif])
        sigma4 = 2.0 - sigmat

        diffuw = density1[col, row+donor] - density1[col, row+upwind]
        diffdw = density1[col,row+downwind] - density1[col,row+donor]

        if (diffuw * diffdw > 0.0):
            limiter = (1.0 - sigmat) * math.copysign(1.0, diffdw)\
                * min(math.fabs(diffuw), min(math.fabs(diffdw), one_by_six\
                * (sigma3 * math.fabs(diffuw) + sigma4 * math.fabs(diffdw))))

        else:
            limiter = 0.0


        mass_flux_y[col,row] = vol_flux_y[col,row]\
            * (density1[col, row+donor] + limiter)

        sigmam = math.fabs(mass_flux_y[col,row])\
            / (density1[col, row+donor] * pre_vol[col,row+ donor])
        diffuw = energy1[col,row+donor] - energy1[col, row+upwind]
        diffdw = energy1[col, row+downwind] - energy1[col, row+donor]

        if (diffuw * diffdw > 0.0):

            limiter = (1.0 - sigmam) * math.copysign(1.0, diffdw)\
                * min(math.fabs(diffuw), min(math.fabs(diffdw), one_by_six \
                * (sigma3 * math.fabs(diffuw) + sigma4 * math.fabs(diffdw))))

        else:

            limiter = 0.0

        ener_flux[col,row] = mass_flux_y[col,row]\
            * (energy1[col,row+donor] + limiter)


@cuda.jit('void(int32, float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:])')
def advec_cell_kernel_y(swp_nmbr,volume, vol_flux_x, vol_flux_y, pre_vol, density1, energy1, ener_flux,\
    mass_flux_y):

    col,row=cuda.grid(2)

   #
   #   if cell is within x area:
   #   +++++++++++++++++++++
   #   +++++++++++++++++++++
   #   ++xxxxxxxxxxxxxxxxx++
   #   +++++++++++++++++++++
   #   +++++++++++++++++++++
   #

    if row > 1 and row <  volume.shape[1] - 2 and  col > 1  and col < volume.shape[0] - 2:
        pre_mass = density1[col,row] * pre_vol[col,row]

        post_mass = pre_mass + mass_flux_y[col,row]\
            - mass_flux_y[col,row+1]

        post_ener = (energy1[col,row] * pre_mass\
            + ener_flux[col,row] - ener_flux[col,row+1])\
             /post_mass

        advec_vol = pre_vol[col,row] + vol_flux_y[col,row]\
            - vol_flux_y[col,row+1]

        density1[col,row] = post_mass / advec_vol
        energy1[col,row] = post_ener
