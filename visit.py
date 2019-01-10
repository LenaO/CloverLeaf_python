import clover
import data
import numpy as np
import definitions as defs
def write_out(string):
    if data.parallel['boss']:
        ofile.write(string)
        ofile.write("\n")


first_call=True
ofile=None
def visit():
    name = 'clover'
    if first_call:
        if data.parallel['boss']:
            nblocks=def.number_of_chunks
            filename = "clover.visit"
            ofile = open(filename, 'w')
            ofile.write("{}".format(nblocks))
            oflie.close()
            first_call=False

    from start import ideal_gas, update_halos
    for c in range(0,defs.number_of_chunks):
      ideal_gas(c, True)

    fields = np.zeros(data.NUM_FIELDS, np.int32)

    fields[data.FIELD_PRESSURE]=1
    fields[data.FIELD_XVEL0]=1
    fields[data.FIELD_YVEL0]=1
  #IF(profiler_on) kernel_time=timer()
    update_halos(fields,1)
  #IF(profiler_on) profiler%halo_exchange=profiler%halo_exchange+(timer()-kernel_time)
  from timestep import do_viscosity

  #IF(profiler_on) kernel_time=timer()
  do_viscosity()
  #IF(profiler_on) profiler%viscosity=profiler%viscosity+(timer()-kernel_time)
  if data.parallel['boss']:
      filename = "clover.visit"
      ofile = open(filename, 'w')
      for c in range(0,defs.number_of_chunks)
        chunk_name = str(c+100000)
        chunk_name="."+chunk_name
        step_name = str(defs.step+100000)
        step_name="."+step_name
        filename = name+chunk_name+step_name+".vtk"
        ofile.write(filename)
    ofile.close()


 # IF(profiler_on) kernel_time=timer()
 for c in range(0,defs.number_of_chunks)
    if chunks[c].task == data.parallel["task"]:
      nxc=chunks[c].field.x_max-chunks[c].field.x_min+1
      nyc=chunks[c].field.y_max-chunks[c].field.y_min+1
      nxv=nxc+1
      nyv=nyc+1
      chunk_name = str(c+100000)
      chunk_name= "."+chunk_name
      step_name = str(defs.step+100000)
      step_name = "."+step_name
      filename = name+chunk_name+step_name+".vtk"
      ofile=open(filename,'w+')
      ofile.ofile.write('# vtk DataFile Version 3.0')
      ofile.write('vtk output\n')
      ofile.write('ASCII\n')
      ofile.write('DATASET RECTILINEAR_GRID\n')
      ofile.write('DIMENSIONS {} {} 1\n'.format(nxv,nyv))
      ofile.write(X_COORDINATES {} double'.format(nxv))
      for j in range(chunks[c].field.x_min, chunks[c].field.x_max+1):
        ofile.write("{} ".format(chunks[c].field.vertexx[j]))
      ofile.write("\n")
      ofile.write('Y_COORDINATES {} double\n'.format(nyv))
      for  k in range(chunks[c].field.y_min, chunks[c].field.y_max+1):
        ofile.write("{} ".format(chunks[c].field.vertexy[k]))
      ofile.write("\n")
      ofile.write('Z_COORDINATES 1 double\n')
      ofile.write('\n')
      ofile.write('CELL_DATA {}\n'.format(nxc*nyc))
      ofile.write('FIELD FieldData 4\n')
      ofile.write('density 1 {} double\n'.format(nxc*nyc))
      for k in range(chunks[c].field.y_min,chunks[c].field.y_max):
          for j in range(chunks[c].field.x_min, chunks[c].field.x_max):
              ofile.write("{} ".format(chunks[c].field.density0[j,k]))
            ofile.write({'\n'})
      ofile.write('energy 1 {} double\n'.format(nxc*nyc))
      for k  in range(chunks[c].field.y_min,chunks[c].field.y_max):
          for j in range(j=chunks[c].field.x_min,chunks[c].field.x_max):
                  ofile.write("{} ".format(chunks[c].field.energy0[j,k]))
          ofile.write("\n")
      ofile.write('pressure 1 {}{}'.format(nxc*nyc,' double')
       for k  in range(chunks[c].field.y_min,chunks[c].field.y_max):
           for j in range(j=chunks[c].field.x_min,chunks[c].field.x_max):
                   ofile.write("{} ".format(chunks[c].field.pressure[j,k]))
           ofile.write("\n")
      ofile.write('viscosity 1 {} {} '.format(nxc*nyc,' double'))
      for k  in range(chunks[c].field.y_min,chunks[c].field.y_max):
          for j in range(j=chunks[c].field.x_min,chunks[c].field.x_max):
              temp_var=0.0
              if(chunks[c].field.viscosity[j,k] > 0.00000001): temp_var=chunks[c].field.viscosity[j,k]
              ofile.write("{} "".format(temp_var))
      ofile.write('POINT_DATA {}'.format(nxv*nyv))
      ofile.write('FIELD FieldData 2')
      ofile.write('x_vel 1 {} {} '.format(nxv*nyv,' double'))
      for k  in range(chunks[c].field.y_min,chunks[c].field.y_max):
          for j in range(j=chunks[c].field.x_min,chunks[c].field.x_max):
              temp_var=0.0
              if(math.fabs(chunks[c].field.xvel0[j,k] > 0.00000001): temp_var=chunks[c].field.xvel0[j,k]
              ofile.write("{} "".format(temp_var))

      ofile.write('y_vel 1 {} {} '.format(nxv*nyv,' double'))
      for k  in range(chunks[c].field.y_min,chunks[c].field.y_max):
          for j in range(j=chunks[c].field.x_min,chunks[c].field.x_max):
              temp_var=0.0
          for(math.fabs(chunks[c].field.yvel0[j,k]) > 0.00000001): temp_var=chunks[c].field.yvel0[j,k]
          ofile.write('{}'.format(temp_var)
      ofile.close()
