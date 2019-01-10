g_version=1.0
g_ibig=640000
g_small=1.0e-16
g_big  =1.0e+21

g_name_len_max=255 
g_xdir=1           
g_ydir=2

file_in = 0
flie_out = 0
CHUNK_LEFT   =1    
CHUNK_RIGHT  =2    
CHUNK_BOTTOM =3    
CHUNK_TOP    =4    
EXTERNAL_FACE=-1

FIELD_DENSITY0   = 0
FIELD_DENSITY1   = 1
FIELD_ENERGY0    = 2   
FIELD_ENERGY1    = 3   
FIELD_PRESSURE   = 4   
FIELD_VISCOSITY  = 5   
FIELD_SOUNDSPEED = 6   
FIELD_XVEL0      = 7   
FIELD_XVEL1      = 8   
FIELD_YVEL0      = 9   
FIELD_YVEL1      =10   
FIELD_VOL_FLUX_X =11   
FIELD_VOL_FLUX_Y =12   
FIELD_MASS_FLUX_X=13   
FIELD_MASS_FLUX_Y=14   
NUM_FIELDS       =15

CELL_DATA     = 1
VERTEX_DATA   = 2
X_FACE_DATA   = 3
y_FACE_DATA   = 4

SOUND = 1  
X_VEL = 2  
Y_VEL = 3  
DIVERG= 4

g_rect=1
g_circ=2
g_point=3

parallel={
        "boss": False,
        "parallel": False,
        "task": 1,
        "max_task": 1,
        "boss_task": 1
        }
