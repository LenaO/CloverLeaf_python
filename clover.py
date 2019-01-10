from mpi4py import MPI
import data
import definitions
from numba import jit
import numpy
chunk_left=0
chunk_right=1
chunk_top =2
chunk_bottom = 3
from definitions import number_of_chunks
from initialise import log
external_face = -1
def clover_barrier():
    MPI.COMM_WORLD.barrier()

def clover_abort():
    MPI.abbort()

def clover_finalize():
    # do to: close io files, print
    data.file_out.close()



def clover_init_comms():
    rank=MPI.COMM_WORLD.Get_rank()
    size=MPI.COMM_WORLD.Get_size()
    data.parallel['parallel']=True
    data.parallel['task']= rank
    if rank is 0:
        data.parallel['boss'] = True
    data.parallel['boss_task'] = 0
    data.parallel['max_task'] = size

def clover_get_rank():
    return MPI.COMM_WORLD.Get_rank()

def clover_get_num_chunks():
    return data.parallel['max_task']


def clover_decompose(x_cells,y_cells,left,right,bottom,top):
    mesh_ratio=float(x_cells)/float(y_cells)
    chunk_x = definitions.number_of_chunks
    chunk_y = 1
    split_found = 0
    factor_x=1.0
    factor_y=1.0
    print("x_cells {} and y_cells{}".format(x_cells, y_cells))
    for c in range(1,definitions.number_of_chunks+1):
        if(definitions.number_of_chunks%c) is 0:
            factor_x= float( definitions.number_of_chunks)/float(c)
            factor_y = float(c)
            if float(factor_x)/float(factor_y) <= mesh_ratio:
                chunk_y=c
                chunk_x= definitions.number_of_chunks/c
                split_found=1
                break
    if (split_found == 0) or (chunk_y== definitions.number_of_chunks): # Prime number or 1D decomp detected
        if mesh_ratio >=1.0:
            chunk_x=definitions.number_of_chunks
            chunk_y=1
        else:
            chunk_x=1
            chunk_y=definitions.number_of_chunks
    delta_x=x_cells/chunk_x
    delta_y=y_cells/chunk_y

    mod_x=x_cells%chunk_x
    mod_y=y_cells%chunk_y
#et up chunk mesh ranges and chunk connectivity

    add_x_prev=0
    add_y_prev=0
    chunk=0

    for cy in range(1, int(chunk_y)+1):
        for cx in range (1, int(chunk_x)+1):
            add_x=0
            add_y=0
            if(cx <= mod_x):add_x=1
            if(cy <= mod_y):add_y=1
            left[chunk]=(cx-1)*delta_x+1+add_x_prev
            right[chunk]=left[chunk]+delta_x-1+add_x
            bottom[chunk]=(cy-1)*delta_y+1+add_y_prev
            top[chunk]=bottom[chunk]+delta_y-1+add_y
            definitions.chunks[chunk].chunk_neighbours[chunk_left]=chunk_x*(cy-1)+cx-1
            definitions.chunks[chunk].chunk_neighbours[chunk_right]=chunk_x*(cy-1)+cx+1
            definitions.chunks[chunk].chunk_neighbours[chunk_bottom]=chunk_x*(cy-2)+cx
            definitions.chunks[chunk].chunk_neighbours[chunk_top]=chunk_x*(cy)+cx
            if (cx == 1):
                definitions.chunks[chunk].chunk_neighbours[chunk_left]=external_face
            if (cx == chunk_x):
                definitions.chunks[chunk].chunk_neighbours[chunk_right]=external_face
            if (cy ==1):
                definitions.chunks[chunk].chunk_neighbours[chunk_bottom]=external_face
            if(cy == chunk_y):
                definitions.chunks[chunk].chunk_neighbours[chunk_top]=external_face
            if(cx <mod_x ):
                add_x_prev=add_x_prev+1
            chunk=chunk+1
        add_x_prev=0
        if cy < mod_y:
                add_y_prev=add_y_prev+1
        log("Mesh ratio of {}".format(mesh_ratio))
        log("Decomposing the mesh into {} by {} chunks".format(chunk_x, chunk_y))


#def clover_exchange(fields,depth)

def clover_sum(value):
    result = numpy.zeros(1, value.dtype)
    MPI.COMM_WORLD.Reduce(value, result, op=MPI.SUM, root=0)
    #print(result)
    return result[0]

def clover_min(value):
    result = numpy.zeros(1, value.dtype)
    MPI.COMM_WORLD.Allreduce(value, result, op=MPI.MIN)
    return result[0]

def clover_max(value):
    result = numpy.zeros(1, value.dtype)
    MPI.COMM_WORLD.Allreduce(value, result, op=MPI.MAX)
    return result[0]
