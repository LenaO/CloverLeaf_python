import data
import numpy as np

class state_type:
    def __init__(self):
        self.defined = False
        self.density = np.zeros(1)
        self.density  = np.zeros(1) 
        self. energy = np.zeros(1)
        self. xvel  = np.zeros(1)
        self. yvel  = np.zeros(1)
        self.geometry = np.array(0, np.int32)

        self.xmin   = np.empty(1,np.float64)    
        self.xmax   = np.empty(1,np.float64)    
        self.ymin   = np.empty(1,np.float64)    
        self.ymax   = np.empty(1,np.float64)     
        self.radius = np.empty(1,np.float64)    


num_of_states = 0
states = []

class grid_type:
    def __init__(self):
        self.xmin   = np.empty(1,np.float64)    
        self.xmax   = np.empty(1,np.float64)    
        self.ymin   = np.empty(1,np.float64)    
        self.ymax   = np.empty(1,np.float64)     
        self.x_cells =  np.empty(1,np.int32)
        self.y_cells =  np.empty(1,np.int32)

step = 0

advect_x = False

error_condition=0

test_problem=0
complete=False

use_fortran_kernels = False
use_C_kernels = False
use_OA_kernels = False

profiler_on = False #Internal code profiler to make comparisons across systems easier

class profiler_type():
    def __init__(self):
        self.timestep     = np.empty(1)  
        self.acceleration = np.empty(1)  
        self.PdV          = np.empty(1)  
        self.cell_advectio= np.empty(1) 
        self.mom_advection= np.empty(1)  
        self.viscosity    = np.empty(1)  
        self.ideal_gas    = np.empty(1)  
        self.visit        = np.empty(1)  
        self.summary      = np.empty(1)  
        self.reset        = np.empty(1)  
        self.revert       = np.empty(1)  
        self.flux         = np.empty(1)  
        self.halo_exchange= np.empty(1)
                     
profiler = profiler_type()

end_time =0.0

end_step = 0

dtold       = np.empty(1)    
dt          = np.empty(1)  
time        = np.empty(1)  
dtinit      = np.empty(1)  
dtmin       = np.empty(1)  
dtmax       = np.empty(1)  
dtrise      = np.empty(1)  
dtu_safe    = np.empty(1)  
dtv_safe    = np.empty(1)  
dtc_safe    = np.empty(1)  
dtdiv_safe  = np.empty(1)  
dtc         = np.empty(1)  
dtu         = np.empty(1)  
dtv         = np.empty(1)  
dtdiv       = np.empty(1)

visit_frequency   = 0
summary_frequency = 0
jdt =0 
kdt = 0

class field_type:
    def __init__(self):
        self.density0 = np.array([[]])
        self.density1 = np.array([[]])
        self.energy0 = np.array([[]])
        self.energy1 = np.array([[]])
        self.Bpressure = np.array([[]]) 
        self.viscosity = np.array([[]])
        self.soundspeed = np.array([[]])
        self.xvel0 = np.array([[]])
        self.xvel1 = np.array([[]])
        self.yvel0 = np.array([[]])
        self.yvel1 = np.array([[]])
        self.vol_flux_x = np.array([[]])
        self.mass_flux_x = np.array([[]])
        self.vol_flux_y = np.array([[]])
        self.mass_flux_y = np.array([[]])
        self.work_array1  = np.array([[]])#node_flux, stepbymass, volume_change, pre_vol
        self.work_array2  = np.array([[]])#node_mass_post, post_vol
        self.work_array3  = np.array([[]])#node_mass_pre,pre_mass
        self.work_array4  = np.array([[]])#advec_vel, post_mass
        self.work_array5  = np.array([[]])#mom_flux, advec_vol
        self.work_array6  = np.array([[]])#pre_vol, post_ener
        self.work_array7  = np.array([[]])#post_vol, ener_flux

        self.left            = np.zeros(1, np.int32)
        self.right           = np.zeros(1, np.int32)
        self.bottom          = np.zeros(1, np.int32)
        self.top             = np.zeros(1, np.int32)
        self.left_boundary   = np.zeros(1, np.int32)
        self.right_boundary  = np.zeros(1, np.int32)
        self.bottom_boundary = np.zeros(1, np.int32)
        self.top_boundary = np.zeros(1, np.int32)

        self.x_min  = np.zeros(1, np.int32)
        self.y_min  = np.zeros(1, np.int32)
        self.x_max  = np.zeros(1, np.int32)
        self.y_max = np.zeros(1, np.int32)

        self.cellx = np.array([[]])
        self.celly  = np.array([[]])    
        self.vertexx  = np.array([[]])
        self.vertexy  = np.array([[]])
        self.celldx   = np.array([[]])
        self.celldy   = np.array([[]])
        self.vertexdx = np.array([[]])
        self.vertexdy = np.array([[]])
        self.volume  = np.array([[]])
        self.xarea   = np.array([[]])
        self.yarea  = np.array([[]])



class chunk_type:
    def __init__(self):

     self.task = 0   #mpi task

     self.chunk_neighbours = np.array([-1,-1,-1,-1]) # Chunks, not tasks, so we can overload in the future
     self.field = field_type()


chunks = []
number_of_chunks = 1
grid = grid_type()

