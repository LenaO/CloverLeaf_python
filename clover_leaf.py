import clover
import data
from data import parallel
from initialise import initialise, log
from hydro import hydro
from initialise import log
import click
import numpy as np
def cloverleaf():
    import profiling
    clover.clover_init_comms()

    if parallel["boss"] is True:
        print('Clover Version {}'.format(data.g_version))
        print('MPI Version')
        print('Task Count {} '.format(parallel["max_task"]))
    initialise()
    hydro()

    clover.clover_finalize()

    kernel = profiling.get("CloverLeafCuda")
    for key in kernel.keys():
        times= kernel[key]
        max = clover.clover_max(np.max(times))
        print("{}:\t {:.2f} {:.2f} {:.2f} {:.2f}".format(key, np.min(times), max, np.average(times), np.sum(times)))

    filename="profile_{}of{}.log".format(data.parallel['task'], data.parallel['max_task'])
    profiling.save(filename, "./", 'w')
if __name__ == '__main__':
    cloverleaf()
