import clover
import data
import numpy as np
import definitions as defs

from start import ideal_gas, update_halos,log


def do_viscosity():
    from CloverLeafCuda import viscosity
    for c in range(0,defs.number_of_chunks):
        if defs.chunks[c].task == data.parallel["task"]:
            viscosity()
def do_calc_dt(chunk):
    if defs.chunks[chunk].task!=data.parallel["task"]: return data.g_big, 0,0,0,0,0

    from CloverLeafCuda import calc_dt
    return calc_dt()


def timestep():

    defs.dt= data.g_big
    small=0
    dt_control=0
    x_pos=0
    y_pos=0

    for c in range(0,defs.number_of_chunks):
        ideal_gas(c, False)
    fields = np.zeros(data.NUM_FIELDS, np.int32)
    fields[data.FIELD_PRESSURE]=1
    fields[data.FIELD_ENERGY0]=1
    fields[data.FIELD_DENSITY0]=1
    fields[data.FIELD_XVEL0]=1
    fields[data.FIELD_YVEL0]=1

    update_halos(fields,1)

    do_viscosity()
    fields = np.zeros(data.NUM_FIELDS, np.int32)
    fields[data.FIELD_VISCOSITY]=1
    update_halos(fields,1)

    for c in range(0,defs.number_of_chunks):
        dltp, dtl_control, xl_pos, yl_pos, jldt, kldt =  do_calc_dt(c)
            #print(dltp)
        if dltp <= defs.dt:
            defs.dt=dltp
            dt_control = dtl_control
            x_pos=xl_pos
            y_pos = yl_pos
            defs.jdt=jldt
            defs.kdt=kldt

    dt=min(min(defs.dt, (defs.dtold * defs.dtrise)), defs.dtmax)
    dt = np.array(dt, np.float64)
    defs.dt=clover.clover_min(dt)

    if defs.dt < defs.dtmin: small=1
    log("step {:3d} , time {:.8f}, controll {}, timestep {:.2e}, {:1d} {:1d}, x {:.2e}, y {:.2e}".format(defs.step, defs.time, dt_control, defs.dt, defs.jdt, defs.kdt, x_pos, y_pos))
    defs.dtold=defs.dt
