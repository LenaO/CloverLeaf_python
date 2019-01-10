from definitions import chunks
import numpy as np
from CloverLeafCuda import cuda_chunk
def build_field(chunk,x_cells,y_cells):

    
    
    chunks[chunk].field.x_min=1
    chunks[chunk].field.y_min=1

    chunks[chunk].field.x_max=x_cells
    chunks[chunk].field.y_max=y_cells

    cuda_chunk.init( chunks[chunk].field.x_min,  chunks[chunk].field.x_max,  chunks[chunk].field.y_min,  chunks[chunk].field.y_max, chunk)
    return 
    print((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-2)),abs(int(chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-2))))
    chunks[chunk].field.density0 =   np.zeros((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-2)),abs(int(chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-2))))
    chunks[chunk].field.density1 =   np.zeros((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-2)),abs(int(chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-2))))
    chunks[chunk].field.energy0  =   np.zeros((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-2)),abs(int(chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-2))))
    chunks[chunk].field.energy1  =   np.zeros((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-2)),abs(int(chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-2))))
    chunks[chunk].field.pressure =   np.zeros((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-2)),abs(int(chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-2))))
    chunks[chunk].field.viscosity=   np.zeros((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-2)),abs(int(chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-2))))
    chunks[chunk].field.soundspeed = np.zeros((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-2)),abs(int(chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-2))))
    chunks[chunk].field.xvel0 = np.zeros((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-3)), abs(int(chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-3))))
    chunks[chunk].field.xvel1 = np.zeros((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-3)), abs(int(chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-3))))
    chunks[chunk].field.yvel0 = np.zeros((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-3)), abs(int(chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-3))))
    chunks[chunk].field.yvel1 = np.zeros((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-3)), abs(int(chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-3))))

    chunks[chunk].field.vol_flux_x = np.zeros((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-3)), abs(int(chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-2))))
    chunks[chunk].field.mass_flux_x= np.zeros((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-3)), abs(int(chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-2))))
    chunks[chunk].field.vol_flux_y = np.zeros((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-2)), abs(int(chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-3))))
    chunks[chunk].field.mass_flux_y= np.zeros((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-2)), abs(int(chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-3))))

    chunks[chunk].field.work_array1= np.zeros((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-3)),abs(int( chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-3))))
    chunks[chunk].field.work_array2= np.zeros((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-3)),abs(int( chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-3))))
    chunks[chunk].field.work_array3= np.zeros((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-3)),abs(int( chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-3))))
    chunks[chunk].field.work_array4= np.zeros((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-3)),abs(int( chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-3))))
    chunks[chunk].field.work_array5= np.zeros((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-3)),abs(int( chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-3))))
    chunks[chunk].field.work_array6= np.zeros((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-3)),abs(int( chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-3))))
    chunks[chunk].field.work_array7= np.zeros((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-3)),abs(int( chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-3))))

    chunks[chunk].field.cellx   = np.zeros(abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-2)))
    chunks[chunk].field.celly   = np.zeros(abs(int(chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-2)))
    chunks[chunk].field.vertexx = np.zeros(abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-3)))
    chunks[chunk].field.vertexy = np.zeros(abs(int(chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-3)))
    chunks[chunk].field.celldx  = np.zeros(abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-2)))
    chunks[chunk].field.celldy  = np.zeros(abs(int(chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-2)))
    chunks[chunk].field.vertexdx= np.zeros(abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-3)))
    chunks[chunk].field.vertexdy= np.zeros(abs(int(chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-3)))
    chunks[chunk].field.volume  = np.zeros((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-2)),abs(int(chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-2))))
    chunks[chunk].field.xarea   = np.zeros((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-3)),abs(int(chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-2))))
    chunks[chunk].field.yarea   = np.zeros((abs(int(chunks[chunk].field.x_min-2-chunks[chunk].field.x_max-2)),abs(int(chunks[chunk].field.y_min-2-chunks[chunk].field.y_max-3))))

 
        #
