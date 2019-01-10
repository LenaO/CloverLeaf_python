import clover
import data
import definitions
from start import start

def log(string):
    if data.parallel['boss']:
        data.file_out.write(string)
        data.file_out.write("\n")

def find_digit(line):
    import string
    for i in range(1,len(line)):
        if line[i].isdigit():
            digit = max(state_max,int(line[i]))
            return digit

def read_input():
    from data import parallel, file_in
    from definitions import grid,profiler, states
    definitions.test_problem=0

    state_max=0

    grid.xmin=  0.0
    grid.ymin=  0.0
    grid.xmax=100.0
    grid.ymax=100.0

    grid.x_cells=10
    grid.y_cells=10

    definitions.end_time=10.0
    definitions.end_step=data.g_ibig
    definitions.complete=False

    definitions.visit_frequency=0
    definitions.summary_frequency=10

    definitions.dtinit=0.1
    definitions.dtmax=1.0
    definitions.dtmin=0.0000001
    definitions.dtrise=1.5

    definitions.dtinit=0.1
    definitions.dtmax=1.0
    definitions.dtmin=0.0000001
    definitions.dtrise=1.5
    definitions.dtc_safe=0.7
    definitions.dtu_safe=0.5
    definitions.dtv_safe=0.5
    definitions.dtdiv_safe=0.7

    definitions.profiler_on=False
    profiler.timestep=0.0
    profiler.acceleration=0.0
    profiler.PdV=0.0
    profiler.cell_advection=0.0
    profiler.mom_advection=0.0
    profiler.viscosity=0.0
    profiler.ideal_gas=0.0
    profiler.visit=0.0
    profiler.summary=0.0
    profiler.reset=0.0
    profiler.revert=0.0
    profiler.flux=0.0
    profiler.halo_exchange=0.0
    if parallel["boss"]:
        log('Reading input File')
    import string
    import re
    state_lines = []
    for line in data.in_file: 
        if "state" in line:
            for i in range(1,len(line)):
                if line[i].isdigit():
                    state_max= max(state_max,int(line[i]))
                    break
            state_lines.append(line)
            continue
        if 'initial_timestep' in line:
            definitions.dtinit=float(re.search("\d+(\.\d+)", line).group())
            if parallel['boss']: log('initial_timestep {}'.format(definitions.dtinit))

        if 'max_timestep' in line:
            definitions.dtmax=float(re.search("\d+(\.\d+)?", line).group())
            if parallel['boss']: log('max_timestep {}'.format(definitions.dtmax))

        if 'timestep_rise' in line:
            definitions.dtrise=float(re.search("\d+(\.\d+)", line).group())
            if parallel['boss']: log('timestep_rise {}'.format(definitions.dtrise))
        if 'end_time' in line:
            definitions.end_time=float(re.search("\d+(\.\d+)?", line).group())
            if parallel['boss']: log('end_time {}'.format(definitions.end_time))

        if 'end_step' in line:
            definitions.end_step=float(re.search("\d+(\.\d+)?", line).group())
            if parallel['boss']: log('end_step {}'.format(definitions.end_step))
        if  'xmin' in line:
            grid.xmin= float(re.search("\d+(\.\d+)?", line).group())
            if parallel['boss']: log('xmin {}'.format(grid.xmin))
        if 'xmax' in line:
            grid.xmax= float(re.search("\d+(\.\d+)?", line).group())
            if parallel['boss']: log('xmax {}'.format(grid.xmax))
        if 'ymin' in line:
            grid.ymin= float(re.search("\d+(\.\d+)?", line).group())
            if parallel['boss']: log('ymin {}'.format(grid.ymin))
        if 'ymax' in line:
            grid.ymax= float(re.search("\d+(\.\d+)?", line).group())
            if parallel['boss']: log('ymax {}'.format(grid.ymax))
        if 'x_cells' in line:
            grid.x_cells= float(re.search("\d+(\.\d+)?", line).group())
            if parallel['boss']: log('x_cells {}'.format(grid.x_cells))
        if 'y_cells' in line:
            grid.y_cells= float(re.search("\d+(\.\d+)?", line).group())
            if parallel['boss']: log('y_cells {}'.format(grid.y_cells))
        if 'visit_frequency' in line:
            definitions.visit_frequency= float(re.search("\d+(\.\d+)?", line).group())
            if parallel['boss']: log('visit_frequenncy{}'.format( definitions.visit_frequency))
        if 'summary_freqyency' in line:
            definitions.summary_frequency== float(re.search("\d+(\.\d+)?", line).group())
            if parallel['boss']: log('summary_frequency {}'.format( definitions.summary_frequency))

     
    data.number_of_states= state_max
    definitions.number_of_states = state_max
    if data.number_of_states< 1:  log('read_input','No states defined.')

    from definitions import state_type
    for i in range(0,data.number_of_states):
            states.append(state_type())
    for line in state_lines:
        print(line)
    #first get state number
        state=re.search('state \d', line).group() 
        state= int(state.split(" ")[1])
        if 'xvel' in line:
            tmp = re.search('xvel=\d+(\.\d+)?', line).group()
            states[state-1].xvel = float(tmp.split('=')[1])
            if parallel["boss"]: log("state xvel {}".format(states[state-1].xvel))        
        if 'yvel' in line:
            tmp = re.search('yvel=\d+(\.\d+)?', line).group()
            states[state-1].yvel = float(tmp.split('=')[1])
            if parallel["boss"]: log("state yvel {}".format(states[state-1].yvel))        
        if 'xmin' in line:
            tmp = re.search('xmin=\d+(\.\d+)?', line).group()
            states[state-1].xmin = float(tmp.split('=')[1])
            if parallel["boss"]: log("state xmin {}".format(states[state-1].xmin))        
        if 'ymin' in line:
            tmp = re.search('ymin=\d+(\.\d+)?', line).group()
            states[state-1].ymin = float(tmp.split('=')[1])
            if parallel["boss"]: log("state ymin {}".format(states[state-1].ymin))        
        if 'xmax' in line:
            tmp = re.search('xmax=\d+(\.\d+)?', line).group()
            states[state-1].xmax = float(tmp.split('=')[1])
            if parallel["boss"]: log("state xmax {}".format(states[state-1].xmax))        
        if 'ymax' in line:
            tmp = re.search('ymax=\d+(\.\d+)?', line).group()
            states[state-1].ymax = float(tmp.split('=')[1])
            if parallel["boss"]: log("state ymax {}".format(states[state-1].ymax))        
        if 'radius' in line:
            tmp = re.search('radius=\d+(\.\d+)?', line).group()
            states[state-1].radius = float(tmp.split('=')[1])
            if parallel["boss"]: log("state radius {}".format(states[state-1].radius))        
        if 'density' in line:
            tmp = re.search('density=\d+(\.\d+)?', line).group()
            states[state-1].density = float(tmp.split('=')[1])
            if parallel["boss"]: log("state density {}".format(states[state-1].density))        
        if 'energy' in line:
            tmp = re.search('energy=\d+(\.\d+)?', line).group()
            states[state-1].energy = float(tmp.split('=')[1])
            if parallel["boss"]: log("state energy {}".format(states[state-1].energy))        
        if 'geometry' in line:
            tmp = re.search('geometry=[A-Za-z_]+', line).group()
            states[state-1].geometry = data.g_rect
            tmp = tmp.split("=")[1]
            if 'rectangle' in tmp:
                states[state-1].geometry = data.g_rect
                if parallel["boss"]: log("state geometry {}".format(tmp))        
            if 'circle' in tmp:
                states[state-1].geometry = data.g_circ
                if parallel["boss"]: log("state geometry {}".format(tmp))        
            if 'point' in tmp:
                states[state-1].geometry = data.g_point
                if parallel["boss"]: log("state geometry {}".format(tmp))


    if parallel["boss"]: log("Input read finished")





def initialise():
    from data import parallel
    data.file_out = open("clover.out", 'w') 
    
    if parallel['boss'] is True:
        log('Clover Version {}'.format(data.g_version))
        log('MPI Version')
        log('Task Count {} '.format(parallel["max_task"]))
        print('Output file clover.out opened. All output will go there.')
        
    clover.clover_barrier()

    data.in_file = open("clover.in", 'r')
    read_input()
    start()
