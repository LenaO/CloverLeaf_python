import data
import numpy as np
import definitions as defs
from timestep import timestep
from timestep import log

def do_pdv(predict):
    for c in range(0,defs.number_of_chunks):
        if defs.chunks[c].task == data.parallel["task"]:
            from CloverLeafCuda import PdV
            error = PdV(predict, defs.dt)
    from clover import clover_max
    #print("Error here {}".format(error))
    error=np.array(error, np.int32)
    error =  clover_max(error)
    if error != 0:
        print("Error")
    if predict:
        from start import ideal_gas, update_halos
        for c in range(0,defs.number_of_chunks):
            ideal_gas(c, True)

        fields = np.zeros(data.NUM_FIELDS, np.int32)
        fields[data.FIELD_PRESSURE]=1
        update_halos(fields,1)
    if predict:
        from CloverLeafCuda import revert
        revert()

def do_accelerate():
    from CloverLeafCuda import accelerate
    for c in range(0,defs.number_of_chunks):
        if defs.chunks[c].task == data.parallel["task"]:
            accelerate(defs.dt)

def do_flux_calc():
    from CloverLeafCuda import flux_calc
    for c in range(0,defs.number_of_chunks):
        if defs.chunks[c].task == data.parallel["task"]:
            flux_calc(defs.dt)

def do_advection():
    sweep_number =1
    if defs.advect_x : direction= data.g_xdir
    if not defs.advect_x: direction = data.g_ydir
    xvel = data.g_xdir
    yvel = data.g_ydir
    from start import update_halos

    fields = np.zeros(data.NUM_FIELDS, np.int32)
    fields[data.FIELD_ENERGY1]=1
    fields[data.FIELD_DENSITY1]=1
    fields[data.FIELD_VOL_FLUX_X]=1
    fields[data.FIELD_VOL_FLUX_Y]=1
    update_halos(fields,2)
    for c in range(0,defs.number_of_chunks):
        if defs.chunks[c].task == data.parallel["task"]:
            from CloverLeafCuda import advec_cell
            advec_cell(direction,sweep_number)

    fields = np.zeros(data.NUM_FIELDS, np.int32)

    fields[data.FIELD_DENSITY1]=1
    fields[data.FIELD_ENERGY1]=1
    fields[data.FIELD_XVEL1]=1
    fields[data.FIELD_YVEL1]=1
    fields[data.FIELD_MASS_FLUX_X]=1
    fields[data.FIELD_MASS_FLUX_Y]=1
    update_halos(fields,2)

    for c in range(0,defs.number_of_chunks):
        if defs.chunks[c].task == data.parallel["task"]:
            from CloverLeafCuda import advec_mom
            advec_mom(xvel,sweep_number,direction)
    for c in range(0,defs.number_of_chunks):
        if defs.chunks[c].task == data.parallel["task"]:
            from CloverLeafCuda import advec_mom
            advec_mom(yvel,sweep_number,direction)

    sweep_number=2
    if defs.advect_x:   direction=data.g_ydir
    if  not defs.advect_x: direction=data.g_xdir

     # IF(profiler_on) kernel_time=timer()
    for c in range(0,defs.number_of_chunks):
        if defs.chunks[c].task == data.parallel["task"]:
            from CloverLeafCuda import advec_cell
            advec_cell(direction, sweep_number)

      #IF(profiler_on) profiler%cell_advection=profiler%cell_advection+(timer()-kernel_time)
    fields = np.zeros(data.NUM_FIELDS, np.int32)
    fields[data.FIELD_DENSITY1]=1
    fields[data.FIELD_ENERGY1]=1
    fields[data.FIELD_XVEL1]=1
    fields[data.FIELD_YVEL1]=1
    fields[data.FIELD_MASS_FLUX_X]=1
    fields[data.FIELD_MASS_FLUX_Y]=1
    update_halos(fields,2)
      #IF(profiler_on) profiler%halo_exchange=profiler%halo_exchange+(timer()-kernel_time)

      #IF(profiler_on) kernel_time=timer()
    for c in range(0,defs.number_of_chunks):
        if defs.chunks[c].task == data.parallel["task"]:
            from CloverLeafCuda import advec_mom
            advec_mom(xvel,sweep_number,direction)
    for c in range(0,defs.number_of_chunks):
        if defs.chunks[c].task == data.parallel["task"]:
            from CloverLeafCuda import advec_mom
            advec_mom(yvel,sweep_number,direction)

     # IF(profiler_on) profiler%mom_advection=profiler%mom_advection+(timer()-kernel_time)

def do_reset_field():
    for c in range(0,defs.number_of_chunks):
        if defs.chunks[c].task == data.parallel["task"]:
            from CloverLeafCuda import reset_field
            reset_field()
def hydro():
    from Timer import Timer
    complete = False
    wall_clock = Timer()
    wall_clock.start()
    while(not complete):
        step_timer = Timer()
        step_timer.start()
        defs.step = defs.step + 1
        timestep()
        do_pdv(True)
        do_accelerate()
        do_pdv(False)
        do_flux_calc()
        do_advection()
        do_reset_field()

        defs.advect_x = not(defs.advect_x)

        defs.time = defs.time + defs.dt
        if defs.summary_frequency != 0:
            if defs.step%defs.summary_frequency == 0:
                from CloverLeafCuda import field_summary
                field_summary()
        if defs.visit_frequency != 0 :
            if defs.step % defs.visit_frequency == 0:
                from visit import visit
                visit()
        step_timer.stop()
        if defs.step == 1 :  first_step = step_timer.elapsed_time
        if defs.step == 2 :  second_step = step_timer.elapsed_time

    # Sometimes there can be a significant start up cost that appears in the first step.
    # Sometimes it is due to the number of MPI tasks, or OpenCL kernel compilation.
    # On the short test runs, this can skew the results, so should be taken into account
    #  in recorded run times.
    #if (defs.step == 1) first_step=(timer() - step_time)
    #IF(step.EQ.2) second_step=(timer() - step_time)

        if defs.time+data.g_small >defs.end_time or defs.step >= defs.end_step:

            complete=True
            wall_clock.stop()
            from CloverLeafCuda import field_summary

            field_summary()
            if(defs.visit_frequency!=0 ):
                from visit import visit
                visit()
      #wall_clock=timer() - timerstart
            log('Calculation complete')
            log('Clover is finishing')
            log('Wall clock {}'.format(wall_clock.elapsed_time))
            log('First step overhead {}'.format(first_step-second_step))
        else:
            cells = defs.grid.x_cells * defs.grid.y_cells
            rstep = defs.step
            wall_clock.stop()
            grind_time   = wall_clock.elapsed_time/(rstep * cells)
            step_grind   = step_timer.elapsed_time/cells
            log("Average time per cell {} ".format(grind_time))
            log("Step time per cell  {}  ".format(step_grind))
