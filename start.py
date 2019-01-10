import clover
import data
import numpy as np
import definitions as defs
def log(string):
    if data.parallel['boss']:
        data.file_out.write(string)
        data.file_out.write("\n")
        print(string)

def init_chunk(chunk):
    from definitions import chunks, grid
    dx=np.array((grid.xmax-grid.xmin)/float(grid.x_cells))
    dy=(grid.ymax-grid.ymin)/float(grid.y_cells)

    xmin=grid.xmin+dx*float(chunks[chunk].field.left-1)
    ymin=grid.ymin+dy*float(chunks[chunk].field.bottom-1)
    from CloverLeafCuda import init_chunks
    init_chunks(xmin,ymin,dx,dy)

def ideal_gas(chunk, predict):
    from definitions import chunks
    if chunks[chunk].task is data.parallel["task"]:
        import CloverLeafCuda
        CloverLeafCuda.ideal_gas(predict)

def generate_chunk(chunks):
    from definitions import states
    density = np.zeros(data.number_of_states)
    energy  = np.zeros(data.number_of_states)
    xvel    = np.zeros(data.number_of_states)
    yvel    = np.zeros(data.number_of_states)
    xmin    = np.zeros(data.number_of_states)
    xmax    = np.zeros(data.number_of_states)
    ymin    = np.zeros(data.number_of_states)
    ymax    = np.zeros(data.number_of_states)
    radius    = np.zeros(data.number_of_states)
    geometry = np.zeros(data.number_of_states, np.int32)

    for state in range(0,data.number_of_states):
        density[state]=states[state].density
        energy[state]=states[state].energy
        xvel[state]=states[state].xvel
        yvel[state]=states[state].yvel
        xmin[state]=states[state].xmin
        xmax[state]=states[state].xmax
        ymin[state]=states[state].ymin
        ymax[state]=states[state].ymax
        radius[state]=states[state].radius
        geometry[state]=states[state].geometry
    import CloverLeafCuda
    CloverLeafCuda.generate_chunks(data.number_of_states, density, energy, xvel, yvel, xmin, xmax, ymin, ymax, radius, geometry, data.g_rect, data.g_circ, data.g_point)



def update_halos(fields, depth):
    from CloverLeafCuda import exchange_halo, update_halo
    for c in range(0,defs.number_of_chunks):
        if defs.chunks[c].task == data.parallel["task"]:
            exchange_halo(depth, fields,defs.chunks[c].chunk_neighbours,c)
            update_halo(depth, fields,  defs.chunks[c].chunk_neighbours)

def start():
    log('Setting up initial geometry')

    defs.time  = 0.0
    defs.step  = 0
    defs.dtold = defs.dtinit
    defs.dt    = defs.dtinit

    clover.clover_barrier()

    defs.number_of_chunks=clover.clover_get_num_chunks()
    for i in range(defs.number_of_chunks):
        defs.chunks.append(defs.chunk_type())

    left = np.zeros(defs.number_of_chunks)
    right = np.zeros(defs.number_of_chunks)
    bottom = np.zeros(defs.number_of_chunks)
    top = np.zeros(defs.number_of_chunks)

    clover.clover_decompose(defs.grid.x_cells, defs.grid.y_cells, left, right, bottom, top)
    for c in range(0,defs.number_of_chunks):

        from definitions import chunks
    #Needs changing so there can be more than 1 chunk per task
        chunks[c].task = c
        x_cells = right[c] -left[c]  +1
        y_cells = top[c]   -bottom[c]+1

        if chunks[c].task == data.parallel["task"]:
            from build_field import build_field
            build_field(c,x_cells,y_cells)

        chunks[c].field.left    = left[c]
        chunks[c].field.bottom  = bottom[c]
        chunks[c].field.right   = right[c]
        chunks[c].field.top     = top[c]
        chunks[c].field.left_boundary   = 1
        chunks[c].field.bottom_boundary = 1
        chunks[c].field.right_boundary  = defs.grid.x_cells
        chunks[c].field.top_boundary    = defs.grid.y_cells
        chunks[c].field.x_min = np.array(1, np.int)
        chunks[c].field.y_min = np.array(1, np.int)
        chunks[c].field.x_max = np.array(right[c]-left[c]+1, np.int)
        chunks[c].field.y_max = np.array(top[c]-bottom[c]+1, np.int)

    clover.clover_barrier()

    for c in range(0,defs.number_of_chunks):
        if chunks[c].task == data.parallel["task"]:
            init_chunk(c)
    for c in range(0,defs.number_of_chunks):
        if chunks[c].task == data.parallel["task"]:
            generate_chunk(c)



    defs.advect_x=True
    clover.clover_barrier()

    for c in range(0,defs.number_of_chunks):
        ideal_gas(c, False)

    fields = np.zeros(data.NUM_FIELDS, np.int32)
    fields[data.FIELD_DENSITY0]=1
    fields[data.FIELD_ENERGY0]=1
    fields[data.FIELD_PRESSURE]=1
    fields[data.FIELD_VISCOSITY]=1
    fields[data.FIELD_DENSITY1]=1
    fields[data.FIELD_ENERGY1]=1
    fields[data.FIELD_XVEL0]=1
    fields[data.FIELD_YVEL0]=1
    fields[data.FIELD_XVEL1]=1
    fields[data.FIELD_YVEL1]=1


    update_halos(fields,2)
    if data.parallel['boss']:
        print('Problem initialised and generated')
    from CloverLeafCuda import field_summary
    field_summary()

    if(defs.visit_frequency!= 0):
        import visist
        visit.visit()
