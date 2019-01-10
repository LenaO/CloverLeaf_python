from numba import cuda, jit, float64, int32
import numpy as np
import math
@cuda.jit('void(float64[:,:], float64[:,:], float64[:,:], float64[:,:])')
def  ideal_gas_kernel(density, energy, pressure, sound_speed):

    column, row  =  cuda.grid(2)

    v=0.0
    pressurebyenergy= 0.0
    pressurebyvolume=0.0
    sound_speed_squared = 0.0


    if  row > 1 and row < energy.shape[1]-2 and column > 1 and column< energy.shape[0]-2:

        v = 1.0 / density[column][row]

        pressure[column][row] = (1.4 - 1.0) * density[column][row] * energy[column][row]

        pressurebyenergy = (1.4 - 1.0) * density[column][row]

        pressurebyvolume = - density[column][row] * pressure[column][row]

        sound_speed_squared = v * v * (pressure[column][row] * pressurebyenergy - pressurebyvolume)

        sound_speed[column][row] = math.sqrt(sound_speed_squared)
