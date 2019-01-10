from numba import cuda, jit, float32
import numpy
import math
import clover
import data
import definitions
from clover import external_face, chunk_top, chunk_bottom, chunk_left, chunk_right
from Timer import GPUTimer
BLOCK_SIZE=16
T_BLOCK_SIZE=int(BLOCK_SIZE*BLOCK_SIZE)
cuda_chunk = 0

_DIAGNOSTICS_KEY = "CloverLeafCuda"

def _append_diagnostics(key, value):
    """
    Small helper function to append value to diagnostics with a common key.
    """
    from os.path import join
    import profiling

    profiling.append(join(_DIAGNOSTICS_KEY, key), value)


class CloverLeafCudaChunk:

    def mallocSZX(self,x_extra):
        return cuda.device_array(int(x_extra+self.x_max+4), dtype=numpy.float64)

    def mallocSZY(self,y_extra):
        return cuda.device_array(int(y_extra+self.y_max+4), dtype=numpy.float64)

    def mallocSZ2D(self,x_extra, y_extra):
        return cuda.device_array((int(x_extra+self.x_max+4), int(y_extra+self.y_max+4)), dtype=numpy.float64)
    def __init__(self):

        self.x_max =0
        self.y_max =0
        self.x_min =0
        self.y_min =0
        self.num_blocks = 0
        self.num_blocks_y = 0
        self.num_blocks_x = 0
        self.rank = 0

        self.volume =    0
        self.soundspeed =0
        self.pressure  = 0
        self.viscosity = 0

        self.density0 = 0

        self.density1 = 0
        self.energy0 =  0
        self.energy1 =  0

        self.xvel0 =  0
        self.xvel1 =  0
        self.yvel0 =  0
        self.yvel1 =  0


        self.xarea       =  0
        self.vol_flux_x  =  0
        self.mass_flux_x =  0


        self.yarea      = 0
        self.vol_flux_y = 0
        self.mass_flux_y = 0

        self.cellx    = 0
        self.celldx   = 0
        self.vertexx  = 0
        self.vertexdx = 0

        self.celly     = 0
        self.celldy    = 0
        self.vertexy   = 0
        self.vertexdy  = 0

        self.work_array_1= 0
        self.work_array_2= 0
        self.work_array_3= 0
        self.work_array_4= 0
        self.work_array_5= 0

        self.reduce_buf_1 =0
        self.reduce_buf_2 =0
        self.reduce_buf_3 =0
        self.reduce_buf_4 =0
        self.reduce_buf_5 =0
        self.reduce_buf_6 =0
        self.pdv_reduce_array = 0
        self.dev_left_send_buffer = 0
        self.dev_rigt_send_buffer = 0
        self.dev_top_send_buffer  =  0
        self.dev_bottom_send_buffer = 0

        self.dev_left_recv_buffer = 0
        self.dev_rigt_recv_buffer=  0
        self.dev_top_recv_buffer =  0
        self.dev_bottom_recv_buffer = 0


    def init(self,x_min, x_max, y_min, y_max, chunk):
            #choose device 0 unless specified
        cuda.close()

        num_devices = len(cuda.gpus)
        rank = data.parallel["task"]
        print("{} devices available in rank {}".format(num_devices, rank ))
    #fflush(stdout);
        device_id = rank % num_devices
        print("rank {} uses GPU {}".format(rank, device_id))
        cuda.select_device(device_id)


        self.x_max = int(x_max)
        self.y_max = int(y_max)
        self.x_min =int(x_min)
        self.y_min = int(y_min)
        self.num_blocks =int((((x_max)+5)*((y_max)+5))/BLOCK_SIZE)
        self.num_blocks_x = int((self.x_max+5)/BLOCK_SIZE)
        self.num_blocks_y = int((self.y_max+5)/BLOCK_SIZE)
        if not  (self.x_max+5) % BLOCK_SIZE == 0: self.num_blocks_x+=1
        if not  (self.x_max+5) % BLOCK_SIZE == 0: self.num_blocks_y+=1
       # if not  ((self.x_max+5) * (self.y_max+5)) % BLOCK_SIZE == 0: self.num_blocks+=1
        self.num_blocks=self.num_blocks_x*self.num_blocks_y
        self.rank = clover.clover_get_rank()

        self.volume = self.mallocSZ2D(0,0)
        self.soundspeed = self.mallocSZ2D(0,0)
        self.pressure  = self.mallocSZ2D(0,0)
        self.viscosity =  self.mallocSZ2D(0,0)

        self.density0 =  self.mallocSZ2D(0,0)

        self.density1 = self.mallocSZ2D(0,0)
        self.energy0 =  self.mallocSZ2D(0,0)
        self.energy1 =  self.mallocSZ2D(0,0)

        self.xvel0 =  self.mallocSZ2D(1,1)
        self.xvel1 =  self.mallocSZ2D(1,1)
        self.yvel0 =  self.mallocSZ2D(1,1)
        self.yvel1 =  self.mallocSZ2D(1,1)


        self.xarea =  self.mallocSZ2D(1,0)
        self.vol_flux_x =  self.mallocSZ2D(1,0)
        self.mass_flux_x =  self.mallocSZ2D(1,0)


        self.yarea =  self.mallocSZ2D(0,1)
        self.vol_flux_y =  self.mallocSZ2D(0,1)
        self.mass_flux_y =  self.mallocSZ2D(0,1)

        self.cellx = self.mallocSZX(0)
        self.celldx = self.mallocSZX(0)
        self.vertexx = self.mallocSZX(1)
        self.vertexdx = self.mallocSZX(1)

        self.celly = self.mallocSZY(0)
        self.celldy = self.mallocSZY(0)
        self.vertexy = self.mallocSZY(1)
        self.vertexdy = self.mallocSZY(1)
        self.thr_cellx = cuda.pinned_array(int(self.x_max+4))
        self.thr_celly = cuda.pinned_array(int(self.y_max+4))
        self.work_array_1= self.mallocSZ2D(1,1)
        self.work_array_2= self.mallocSZ2D(1,1)
        self.work_array_3= self.mallocSZ2D(1,1)
        self.work_array_4= self.mallocSZ2D(1,1)
        self.work_array_5= self.mallocSZ2D(1,1)

        self.reduce_buf_1 = cuda.device_array(int(self.num_blocks))
        self.reduce_buf_2 = cuda.device_array(int(self.num_blocks))
        self.reduce_buf_3 = cuda.device_array(int(self.num_blocks))
        self.reduce_buf_4 = cuda.device_array(int(self.num_blocks))
        self.reduce_buf_5 = cuda.device_array(int(self.num_blocks))
        self.reduce_buf_6 = cuda.device_array(int(self.num_blocks))
        self.pdv_reduce_array = cuda.device_array(int(self.num_blocks), dtype=numpy.int)

        self.dev_left_send_buffer=cuda.device_array(int((self.y_max+5)*2))
        self.dev_rigt_send_buffer=  cuda.device_array(int((self.y_max+5)*2))
        self.dev_top_send_buffer = cuda.device_array(int((self.x_max+5)*2))
        self.dev_bottom_send_buffer = cuda.device_array(int((self.x_max+5)*2))

        self.dev_left_recv_buffer = cuda.device_array(int ((self.y_max+5)*2))
        self.dev_rigt_recv_buffer=  cuda.device_array(int ((self.y_max+5)*2))
        self.dev_top_recv_buffer =  cuda.device_array(int ((self.x_max+5)*2))
        self.dev_bottom_recv_buffer = cuda.device_array(int ((self.x_max+5)*2))

        self.left_snd_buffer =  cuda.pinned_array(int(2*(self.y_max+5)))
        self.left_rcv_buffer =  cuda.pinned_array(int(2*(self.y_max+5)))

        self.right_snd_buffer = cuda.pinned_array(int(2*(self.y_max+5)))
        self.right_rcv_buffer = cuda.pinned_array(int(2*(self.y_max+5)))

        self.bottom_snd_buffer = cuda.pinned_array(int(2*(self.x_max+5)))
        self.bottom_rcv_buffer = cuda.pinned_array(int(2*(self.x_max+5)))

        self.top_snd_buffer= cuda.pinned_array(int(2*(self.x_max+5)))
        self.top_rcv_buffer= cuda.pinned_array(int(2*(self.x_max+5)))



    def _init_chunks(self,xmin, ymin, dx, dy):
        import initchunks
        #xmin=0
        kernel1_timer =GPUTimer()
        kernel2_timer =GPUTimer()
        with kernel1_timer:
            initchunks.initialise_chunk_vertex[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE,BLOCK_SIZE) ](xmin,ymin,dx,dy, self.vertexx, self.vertexdx, self.vertexy, self.vertexdy)
        _append_diagnostics("initialise_chunk_vertex", kernel1_timer.elapsed_time )
        with kernel2_timer:
            initchunks.initialise_chunk[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE)]\
            (dx,dy, self.vertexx,self.vertexy, self.cellx, self.celldx, self.celly, self.celldy,self.volume, self.xarea, self.yarea )
        _append_diagnostics("initialise_chunk", kernel2_timer.elapsed_time )

    def _generate_chunks(self,number_of_states, density, energy, xvel, yvel, x_min, x_max, y_min, y_max, radius, geometry, g_rect, g_circ, g_point):
        import generate_chunk

        kernel1_timer =GPUTimer()
        with kernel1_timer:
            generate_chunk.generate_chunk_kernel_init[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE)]\
                (self.density0,self.energy0,  self.xvel0, self.yvel0, density[0], energy[0], xvel[0], yvel[0])

        _append_diagnostics("generate_chunk_init", kernel1_timer.elapsed_time )
        for state in range(1, number_of_states):
            kernel2_timer =GPUTimer()
            with kernel2_timer:
                generate_chunk.generate_chunk_kernel[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE)]\
                (self.vertexx, self.vertexy, self.cellx, self.celly, self.density0, self.energy0, self.xvel0, self.yvel0, \
                density[state], energy[state], xvel[state], yvel[state], x_min[state], x_max[state], y_min[state], y_max[state], radius[state],\
                geometry[state],  g_rect, g_circ, g_point)

            _append_diagnostics("generate_chunk", kernel2_timer.elapsed_time )

    def _ideal_gas(self,predict):
        from  ideal_gas_kernel import ideal_gas_kernel
        kernel1_timer =GPUTimer()
        if predict is True:
            with kernel1_timer:
                ideal_gas_kernel[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE)](self.density1, self.energy1, self.pressure, self.soundspeed)
        else:
            with kernel1_timer:
                ideal_gas_kernel[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE)](self.density0, self.energy0, self.pressure, self.soundspeed)

        _append_diagnostics("ideal_gas", kernel1_timer.elapsed_time )

    def _update_halo_array(self, x_extra, y_extra, x_invert, y_invert, x_face, y_face, grid_type, array, depth, neighbours):
        import update_halo_kernel
        left_timer= GPUTimer()
        right_timer= GPUTimer()
        top_timer= GPUTimer()
        btm_timer= GPUTimer()

        if(neighbours[chunk_bottom]==external_face):
            with btm_timer:
                update_halo_kernel.update_halo_kernel_bottom[(self.num_blocks_x,1),(BLOCK_SIZE, BLOCK_SIZE)]\
                (self.x_max, self.y_max, x_extra, y_invert, grid_type, array, depth)
            _append_diagnostics("update_halo_kernel_bottom", btm_timer.elapsed_time)

        if(neighbours[chunk_top]==external_face):
            with top_timer:
                update_halo_kernel.update_halo_kernel_top[(self.num_blocks_x, 1),(BLOCK_SIZE, BLOCK_SIZE)]\
                (self.x_max, self.y_max, x_extra, y_extra, y_invert, x_face, array,depth)
            _append_diagnostics("update_halo_kernel_top", top_timer.elapsed_time)
        if(neighbours[chunk_left]==external_face):
            with left_timer:
                update_halo_kernel.update_halo_kernel_left[(1, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE)]\
                (self.x_max, self.y_max, x_invert, y_extra, grid_type, array, depth)
            _append_diagnostics("update_halo_kernel_left", left_timer.elapsed_time)
        if(neighbours[chunk_right]==external_face):
            with right_timer:
                update_halo_kernel.update_halo_kernel_right[(1, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE)]\
                (self.x_max, self.y_max, x_extra,y_extra, x_invert, y_face, array, depth)
            _append_diagnostics("update_halo_kernel_right", btm_timer.elapsed_time)

    def _exchange_halo(self, x_inc, y_inc, depth, array, chunk, neighbours):
        from definitions import chunks
        import pack_buffer
        stream = cuda.stream()
        num_blocks_y= math.ceil((self.y_max+4+y_inc)/T_BLOCK_SIZE) * depth
        num_blocks_x= math.ceil((self.x_max+4+x_inc)/T_BLOCK_SIZE) * depth

        if(neighbours[chunk_left]!=external_face):

            pack_buffer.pack_left_buffer[num_blocks_y,(T_BLOCK_SIZE),stream]( array, self.dev_left_send_buffer, depth, x_inc, y_inc, self.y_max)
            self.dev_left_send_buffer.copy_to_host(self.left_snd_buffer, stream=stream)
        if(neighbours[chunk_right]!=external_face):
            pack_buffer.pack_right_buffer[num_blocks_y,(T_BLOCK_SIZE),stream]( array, self.dev_rigt_send_buffer, depth, x_inc, y_inc, self.x_max, self.y_max)
            self.dev_rigt_send_buffer.copy_to_host(self.right_snd_buffer, stream=stream)

        from mpi4py import MPI
        requests = []
        message_count = 0

        stream.synchronize()
        if chunks[chunk].chunk_neighbours[chunk_left]!= external_face:
            tag=4*(chunk)+1 # 4 because we have 4 faces, 1 because it is leaving the left face
            receiver= chunks[neighbours[chunk_left]-1].task
            req=MPI.COMM_WORLD.Isend(self.left_snd_buffer, receiver, tag=tag)
            requests.append(req)
            tag=4*(chunks[neighbours[chunk_left]-1].task)+2 # 4 because we have 4 faces, 1 because it is coming from the right face of the left neighbour
            sender=chunks[neighbours[chunk_left]-1].task
            req=MPI.COMM_WORLD.Irecv(self.left_rcv_buffer, sender, tag=tag)
            requests.append(req)
            message_count=message_count+2
        if chunks[chunk].chunk_neighbours[chunk_right]!= external_face:
            tag=4*(chunk)+2 # 4 because we have 4 faces, 1 because it is leaving the left face
            receiver= chunks[neighbours[chunk_right]-1].task
            req=MPI.COMM_WORLD.Isend(self.right_snd_buffer, receiver, tag=tag)
            requests.append(req)
            tag=4*int(chunks[neighbours[chunk_right]-1].task)+1 # 4 because we have 4 faces, 1 because it is coming from the right face of the left neighbour
            sender= chunks[neighbours[chunk_right]-1].task
            req=MPI.COMM_WORLD.Irecv(self.right_rcv_buffer, sender, tag=tag)
            requests.append(req)
            message_count=message_count+2

        if(neighbours[chunk_bottom]!=external_face):
            pack_buffer.pack_bottom_buffer[num_blocks_x,(T_BLOCK_SIZE),stream]( array, self.dev_bottom_send_buffer, depth, x_inc,y_inc, self.x_max)
        if(neighbours[chunk_top]!=external_face):
            pack_buffer.pack_top_buffer[num_blocks_x,(T_BLOCK_SIZE),stream]( array, self.dev_top_send_buffer, depth,x_inc,y_inc, self.x_max, self.y_max)
        stream.synchronize()

        requests2 = []
        message_count2=0
        if chunks[chunk].chunk_neighbours[chunk_top]!= external_face:
            tag=4*(chunk)+3  # 4 because we have 4 faces, 1 because it is leaving the left face
#            print("Send Tag top {}".format(tag))
            receiver= chunks[neighbours[chunk_top]-1].task
            req=MPI.COMM_WORLD.Isend(self.top_snd_buffer, receiver, tag=tag)
            requests2.append(req)
            tag=4*(chunks[neighbours[chunk_top]-1].task)+4 # 4 because we have 4 faces, 1 because it is coming from the right face of the left neighbour
            sender=chunks[neighbours[chunk_top]-1].task
            req=MPI.COMM_WORLD.Irecv(self.top_rcv_buffer, sender, tag=tag)
            requests2.append(req)
            message_count2=message_count2+2

        if chunks[chunk].chunk_neighbours[chunk_bottom]!= external_face:
            tag=4*(chunk)+4 # 4 because we have 4 faces, 1 because it is leaving the left face
            receiver= chunks[neighbours[chunk_bottom]-1].task
            req=MPI.COMM_WORLD.Isend(self.bottom_snd_buffer, receiver, tag=tag)
            requests2.append(req)
            tag=4*int(chunks[neighbours[chunk_bottom]-1].task)+3 # 4 because we have 4 faces, 1 because it is coming from the right face of the left neighbour
 #           print("recive Tag bottom {}".format(tag))
            sender= chunks[neighbours[chunk_bottom]-1].task
            req=MPI.COMM_WORLD.Irecv(self.bottom_rcv_buffer, sender, tag=tag)
            requests2.append(req)
            message_count2=message_count2+2

        MPI.Request.Waitall(requests)
        if(neighbours[chunk_left]!=external_face):
            self.dev_left_recv_buffer.copy_to_device(self.left_rcv_buffer, stream=stream)
            pack_buffer.unpack_left_buffer[num_blocks_y,(T_BLOCK_SIZE),stream]( array, self.dev_left_recv_buffer, depth, x_inc,y_inc, self.y_max)
            #print("my task  left {} , remote taskt {} chunk {}".format( data.parallel["task"], chunks[neighbours[chunk_left]-1].task, chunk))
        if(neighbours[chunk_right]!=external_face):
            self.dev_rigt_recv_buffer.copy_to_device(self.right_rcv_buffer, stream=stream)
            pack_buffer.unpack_right_buffer[num_blocks_y,(T_BLOCK_SIZE),stream]( array, self.dev_rigt_recv_buffer, depth, x_inc, y_inc, self.x_max, self.y_max)
          #  self.dev_rigt_send_buffer.copy_to_host(self.right_snd_buffer, stream=stream)

        MPI.Request.Waitall(requests2)

        if(neighbours[chunk_top]!=external_face):
            self.dev_top_recv_buffer.copy_to_device(self.top_rcv_buffer, stream=stream)
            pack_buffer.unpack_top_buffer[num_blocks_x,(T_BLOCK_SIZE),stream]( array, self.dev_top_recv_buffer, depth, x_inc, y_inc, self.x_max, self.y_max)
            #print("my task  left {} , remote taskt {} chunk {}".format( data.parallel["task"], chunks[neighbours[chunk_left]-1].task, chunk))
        if(neighbours[chunk_bottom]!=external_face):
            self.dev_bottom_recv_buffer.copy_to_device(self.bottom_rcv_buffer, stream=stream)
            pack_buffer.unpack_bottom_buffer[num_blocks_x,(T_BLOCK_SIZE),stream]( array, self.dev_bottom_recv_buffer, depth, x_inc,y_inc, self.x_max, self.y_max)
          #  self.dev_rigt_send_buffer.copy_to_host(self.right_snd_buffer, stream=stream)

        stream.synchronize()


    def _update_halo(self, depth, fields, neighbours):
        if fields[data.FIELD_DENSITY0] == 1:
            self._update_halo_array(0,0,1,1,0,0,1, self.density0, depth, neighbours)
        if fields[data.FIELD_DENSITY1] == 1:
            self._update_halo_array(0,0,1,1,0,0,1, self.density1, depth, neighbours)


        if  fields[data.FIELD_ENERGY0] == 1:
            self._update_halo_array(0,0,1,1,0,0,1, self.energy0, depth, neighbours)
        if fields[data.FIELD_ENERGY1]== 1:
            self._update_halo_array(0,0,1,1,0,0,1, self.energy1, depth, neighbours)

        if fields[data.FIELD_PRESSURE] == 1:
            self._update_halo_array(0,0,1,1,0,0,1, self.pressure, depth, neighbours)
        if fields[data.FIELD_VISCOSITY]== 1:
            self._update_halo_array(0,0,1,1,0,0,1, self.viscosity, depth, neighbours)
        if fields[data.FIELD_SOUNDSPEED]== 1:
            self._update_halo_array(0,0,1,1,0,0,1, self.soundpeed, depth, neighbours)

        if fields[data.FIELD_XVEL0]    == 1:
            self._update_halo_array(1,1,-1,1,0,0,2, self.xvel0, depth, neighbours)
        if fields[data.FIELD_YVEL0]    == 1:
            self._update_halo_array(1,1,1,-1,0,0,2, self.yvel0, depth,  neighbours)
        if fields[data.FIELD_XVEL1]    == 1:
            self._update_halo_array(1,1,-1,1,0,0,2, self.xvel1, depth,  neighbours)
        if fields[data.FIELD_YVEL1]    == 1:
            self._update_halo_array(1,1,1,-1,0,0,2, self.yvel1, depth,  neighbours)


        if fields[data.FIELD_VOL_FLUX_X]    == 1:
            self._update_halo_array( 1, 0, -1,  1, 1, 0,3, self.vol_flux_x, depth,  neighbours)
        if fields[data.FIELD_VOL_FLUX_Y]    == 1:
            self._update_halo_array(0, 1,  1, -1, 0, 1,4, self.vol_flux_y, depth,  neighbours)


        if fields[data.FIELD_MASS_FLUX_X]    == 1:
            self._update_halo_array( 1, 0, -1,  1, 1, 0,3,  self.mass_flux_x, depth, neighbours)
        if fields[data.FIELD_MASS_FLUX_Y]    == 1:
            self._update_halo_array(0, 1,  1, -1, 0, 1,4, self.mass_flux_y, depth, neighbours)

    def _exchange_halos(self, depth, fields, chunk, neighbours):

        if fields[data.FIELD_DENSITY0] == 1:
            self._exchange_halo(0, 0, depth, self.density0, chunk, neighbours)
        if fields[data.FIELD_DENSITY1] == 1:
            self._exchange_halo(0, 0, depth, self.density1, chunk, neighbours)

        if  fields[data.FIELD_ENERGY0] == 1:
            self._exchange_halo(0, 0, depth, self.energy0, chunk, neighbours)
        if fields[data.FIELD_ENERGY1]== 1:
            self._exchange_halo(0, 0, depth, self.energy1, chunk, neighbours)
        if fields[data.FIELD_PRESSURE] == 1:
            self._exchange_halo(0, 0, depth, self.pressure, chunk, neighbours)
        if fields[data.FIELD_VISCOSITY]== 1:
            self._exchange_halo(0, 0, depth, self.viscosity, chunk, neighbours)
        if fields[data.FIELD_SOUNDSPEED]== 1:
            self._exchange_halo(0, 0, depth, self.soundspeed, chunk, neighbours)

        if fields[data.FIELD_XVEL0]    == 1:
            self._exchange_halo(1, 1, depth, self.xvel0, chunk, neighbours)
        if fields[data.FIELD_YVEL0]    == 1:
            self._exchange_halo(1, 1, depth, self.yvel0, chunk, neighbours)
        if fields[data.FIELD_XVEL1]    == 1:
            self._exchange_halo(1, 1, depth, self.xvel1, chunk, neighbours)
        if fields[data.FIELD_YVEL1]    == 1:
            self._exchange_halo(1, 1, depth, self.yvel1, chunk, neighbours)

        if fields[data.FIELD_VOL_FLUX_X]    == 1:
            self._exchange_halo(1, 0, depth, self.vol_flux_x, chunk, neighbours)
        if fields[data.FIELD_VOL_FLUX_Y]    == 1:
            self._exchange_halo(0, 1, depth, self.vol_flux_y, chunk, neighbours)

        if fields[data.FIELD_MASS_FLUX_X]    == 1:
            self._exchange_halo(1, 0, depth, self.mass_flux_x, chunk, neighbours)
        if fields[data.FIELD_MASS_FLUX_Y]    == 1:
            self._exchange_halo(0, 1, depth, self.mass_flux_y, chunk, neighbours)

    def _field_summary(self):
        import field_summary
        field_timer = GPUTimer()
        with field_timer:
            field_summary.field_summary_kernel[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE), ](\
                    self.volume, self.density0,self.energy0, self.pressure, self.xvel0, self.yvel0, \
                    self.reduce_buf_1, self.reduce_buf_2, self.reduce_buf_3, self.reduce_buf_4, self.reduce_buf_5)

        _append_diagnostics("field_summary", field_timer.elapsed_time )
        reduce_timer = GPUTimer()
        with reduce_timer:
            vol = field_summary.sum_reduce(self.reduce_buf_1)

        _append_diagnostics("vol_reduce", reduce_timer.elapsed_time )

        with reduce_timer:
            mass = field_summary.sum_reduce(self.reduce_buf_2)
        _append_diagnostics("mass_reduce", reduce_timer.elapsed_time )

        with reduce_timer:
            ie = field_summary.sum_reduce(self.reduce_buf_3)
        _append_diagnostics("ie_reduce", reduce_timer.elapsed_time )
        with reduce_timer:
            ke = field_summary.sum_reduce(self.reduce_buf_4)
        _append_diagnostics("ke_reduce", reduce_timer.elapsed_time )
        with reduce_timer:
            press = field_summary.sum_reduce(self.reduce_buf_5)
        _append_diagnostics("press_reduce", reduce_timer.elapsed_time )
        return(vol, mass, ie, ke, press)

    def _viscosity(self):
        import viscosity_kernel
        timer = GPUTimer()
        with timer:
            viscosity_kernel.viscosity_kernel[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE), ]\
                (self.celldx, self.celldy, self.density0, self.pressure, self.viscosity, self.xvel0,self.yvel0)

        _append_diagnostics("viscocity_kernel", timer.elapsed_time )
    def _calc_dt(self):
        import calc_dt_kernel
        timer = GPUTimer()
        with timer:
            calc_dt_kernel.calc_dt_kernel[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE), ]\
                (data.g_small, data.g_big, definitions.dtmin, definitions.dtc_safe, definitions.dtu_safe, definitions.dtv_safe, definitions.dtdiv_safe, \
                self.xarea,  self.yarea, self.celldx, self.celldy, self.volume, self.density0, self.viscosity,self.soundspeed, self.xvel0,\
                self.yvel0, self.reduce_buf_1, self.reduce_buf_2)

        _append_diagnostics("calc_dt_kernel", timer.elapsed_time)
        tmp = self.reduce_buf_2.copy_to_host()
        with timer:
            dt_min_val = calc_dt_kernel.min_reduce(self.reduce_buf_2, init=data.g_big)
        _append_diagnostics("min_reduce", timer.elapsed_time)
        with timer:
            jk_control = calc_dt_kernel.max_reduce(self.reduce_buf_1)
        _append_diagnostics("max_reduce", timer.elapsed_time)

        self.cellx.copy_to_host(self.thr_cellx)
        self.celly.copy_to_host(self.thr_celly)

        dtl_control = int(10.01 * (jk_control - int(jk_control)))

        jk_control = jk_control - (jk_control - int(jk_control))

        tmp_jldt = jldt = int(jk_control) % self.x_max
        tmp_kldt = kldt = 1+ int(jk_control/self.x_max)
        xl_pos = self.thr_cellx[tmp_jldt]
        yl_pos = self.thr_celly[tmp_kldt]
        small = 0
        if (dt_min_val < definitions.dtmin): small = 1

        if (small != 0):
            print("Timestep information: {}".format(dt_min_val))
            print("j, k     : {} {} ".format( tmp_jldt,tmp_kldt))
            print("x, y     : {} {}".format(self.thr_cellx[tmp_jldt], self.thr_celly[tmp_kldt]))
            print("timestep : {} ".format(dt_min_val))
#           print("Cell velocities;")
#        print("{}\t{}".format(thr_xvel0[tmp_jldt+(self.x_max+5)*tmp_kldt], thr_yvel0[tmp_jldt  +(self.x_max+5)*tmp_kldt  ]))
#        print("{}\t{}".format(thr_xvel0[tmp_jldt+1+(self.x_max+5)*tmp_kldt  ],thr_yvel0[tmp_jldt+1+(self.x_max+5)*tmp_kldt  ]))
#        print("{}\t{}".format(thr_xvel0[tmp_jldt+1+(self.x_max+5)*(tmp_kldt+1)], thr_yvel0[tmp_jldt+1+(self.x_max+5)*(tmp_kldt+1)]))
#        print("{}\t{}".format(thr_xvel0[tmp_jldt  +(dx_max+5)*(tmp_kldt+1)], thr_yvel0[tmp_jldt  +(x_max+5)*(tmp_kldt+1)] << std::endl;
#
#       "density, energy, pressure, soundspeed " << std::endl;
#       thr_density0[tmp_jldt+(x_max+5)*tmp_kldt] << "\t";
#       thr_energy0[tmp_jldt+(x_max+5)*tmp_kldt] << "\t";
#       thr_pressure[tmp_jldt+(x_max+5)*tmp_kldt] << "\t";
#       thr_soundspeed[tmp_jldt+(x_max+5)*tmp_kldt] << std::endl;
#
        if dtl_control == 1:
            dtl_control = 'sound'
        if dtl_control == 2:
            dtl_control = 'xvel'
        if dtl_control == 3:
            dtl_control = 'yvel'
        if dtl_control == 4:
            dtl_control = 'div'


        return dt_min_val, dtl_control, xl_pos, yl_pos, jldt, kldt

    def _PdV_kernel(self,prdct, dtbyt):
        import PdV_kernel
        timer = GPUTimer()
        if prdct is True:
            with timer:
                PdV_kernel.PdV_kernel_predict[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE), ]\
                    (dtbyt, self.pdv_reduce_array,  self.xarea,  self.yarea,  self.volume,  self.density0,  self.density1,  self.energy0,  self.energy1,\
                    self.pressure, self.viscosity, self.xvel0, self.yvel0, self.xvel1, self.yvel1)
            _append_diagnostics("PdV_kernel_predict", timer.elapsed_time)
        else:
            with timer:
                PdV_kernel.PdV_kernel_not_predict[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE), ]\
                    (dtbyt, self.pdv_reduce_array, self.xarea,  self.yarea,  self.volume,  self.density0, self.density1,  self.energy0,  self.energy1,\
                    self.pressure, self.viscosity, self.xvel0, self.yvel0, self.xvel1, self.yvel1)
            _append_diagnostics("PdV_kernel_not_predict", timer.elapsed_time)


        from  calc_dt_kernel import max_reduce
        error_condition = max_reduce(self.pdv_reduce_array)
        if (1 == error_condition):
            print("Negative volume in PdV kernel")
        elif (2 == error_condition):
            print("Negative cell volume in PdV kernel")
        return error_condition

    def _revert(self):
        import revert_kernel

        timer = GPUTimer()
        with timer:
            revert_kernel.revert_kernel[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE), ]\
                (self.density0, self.density1, self.energy0, self.energy1)
        _append_diagnostics("revert_kernel", timer.elapsed_time)

    def _accelerate(self,dt):
        import accelerate_kernel
        timer = GPUTimer()
        with timer:
            accelerate_kernel.accelerate_kernel[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE), ]\
                (dt,self.xarea, self.yarea, self.volume, self.density0, self.pressure, self.viscosity, self.xvel0, self.yvel0, self.xvel1, self.yvel1)
        _append_diagnostics("accelerate_kernel", timer.elapsed_time)

    def _flux_calc(self, dt):
        import flux_calc_kernel
        timer=GPUTimer()
        with timer:
            flux_calc_kernel.flux_calc_kernel [(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE), ]\
                (self.x_max, self.y_max, dt, self.xarea, self.yarea, self.xvel0, self.yvel0, self.xvel1, self.yvel1, self.vol_flux_x, self.vol_flux_y)
        _append_diagnostics("flux_calc_kernel", timer.elapsed_time)


    def _advec_cell(self, dr, swp_nmpr):
        import advec_cell_kernel
        swp = numpy.array(swp_nmpr, numpy.int32)
        timer = GPUTimer()
        if dr == 1:
            with timer:
                advec_cell_kernel.pre_vol_kernel_x[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE), ]\
                    (swp, self.work_array_1,  self.work_array_2,  self.volume, self.vol_flux_x,  self.vol_flux_y)
            _append_diagnostics("pre_vol_kernel_x", timer.elapsed_time)
            with timer:
                advec_cell_kernel.ener_flux_kernel_x[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE), ]\
                    (swp,self.volume, self.vol_flux_x, self.vol_flux_y, self.work_array_1, self.density1, self.energy1,\
                    self.work_array_2, self.vertexdx, self.mass_flux_x)
            _append_diagnostics("ener_flux_kernel_x", timer.elapsed_time)
            with timer:
                advec_cell_kernel.advec_cell_kernel_x[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE), ]\
                    (swp,self.volume, self.vol_flux_x, self.vol_flux_y, self.work_array_1, self.density1, self.energy1,\
                    self.work_array_2, self.mass_flux_x)
            _append_diagnostics("advec_cell_kernel_x", timer.elapsed_time)

        elif dr== 2:
            with timer:
                advec_cell_kernel.pre_vol_kernel_y[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE), ]\
                    (swp, self.work_array_1,  self.work_array_2,  self.volume, self.vol_flux_x,  self.vol_flux_y)
            _append_diagnostics("pre_vol_kernel_y", timer.elapsed_time)
            with timer:
                advec_cell_kernel.ener_flux_kernel_y[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE), ]\
                    (swp,self.volume, self.vol_flux_x, self.vol_flux_y, self.work_array_1, self.density1, self.energy1,\
                    self.work_array_2, self.vertexdy, self.mass_flux_y)
            _append_diagnostics("ener_flux_kernel_y", timer.elapsed_time)
            with timer:
                advec_cell_kernel.advec_cell_kernel_y[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE), ]\
                    (swp,self.volume, self.vol_flux_x, self.vol_flux_y, self.work_array_1, self.density1, self.energy1,\
                    self.work_array_2, self.mass_flux_y)
            _append_diagnostics("advec_cell_kernel_y", timer.elapsed_time)

    def _advec_mom(self,which_vel, sweep_number, direction):
        mom_sweep = numpy.array(direction + (2 * (sweep_number - 1)), numpy.int32)
        import advec_mom_kernel
        timer=GPUTimer()
        with timer:
            advec_mom_kernel.advec_mom_vol_kernel[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE), ]\
                (mom_sweep, self.work_array_1, self.work_array_2, self.volume, self.vol_flux_x, self.vol_flux_y)
        _append_diagnostics("advec_mom_vol_kernel", timer.elapsed_time)
        vel1 = 0
        if (which_vel == 1):
            vel1 =  self.xvel1
        else:
            vel1 =  self.yvel1
        if direction == 1:
            with timer:
                advec_mom_kernel.advec_mom_node_flux_post_x_kernel[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE), ]\
                (self.work_array_2, self.work_array_3, self.mass_flux_x, self.work_array_1, self.density1)
            _append_diagnostics("advec_mom_node_flux_post_x_kernel", timer.elapsed_time)
            with timer:
                advec_mom_kernel.advec_mom_node_pre_x_kernel[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE), ]\
                (self.x_max, self.y_max, self.work_array_2, self.work_array_3, self.work_array_4)
            _append_diagnostics("advec_mom_node_pre_x_kernel", timer.elapsed_time)
            with timer:
                advec_mom_kernel.advec_mom_flux_x_kernel[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE)]\
                (self.x_max, self.y_max, self.work_array_2, self.work_array_3, self.work_array_4, vel1, self.celldx, self.work_array_5)
            _append_diagnostics("advec_mom_flux_x_kernel", timer.elapsed_time)
            with timer:
                advec_mom_kernel.advec_mom_xvel_kernel[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE)]\
                (self.x_max, self.y_max,self.work_array_3, self.work_array_4, self.work_array_5, vel1)
            _append_diagnostics("advec_mom_xvel_kernel", timer.elapsed_time)


        elif (direction == 2):
            with timer:
                advec_mom_kernel.advec_mom_node_flux_post_y_kernel[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE), ]\
                (self.work_array_2, self.work_array_3, self.mass_flux_y, self.work_array_1, self.density1)
                _append_diagnostics("advec_mom_node_flux_post_y_kernel", timer.elapsed_time)
            with timer:
                advec_mom_kernel.advec_mom_node_pre_y_kernel[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE), ]\
                (self.x_max, self.y_max, self.work_array_2, self.work_array_3, self.work_array_4)
            _append_diagnostics("advec_mom_node_pre_y_kernel", timer.elapsed_time)
            with timer:
                advec_mom_kernel.advec_mom_flux_y_kernel[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE)]\
                (self.x_max, self.y_max, self.work_array_2, self.work_array_3, self.work_array_4, vel1, self.celldy, self.work_array_5)
            _append_diagnostics("advec_mom_flux_y_kernel", timer.elapsed_time)
            with timer:
                 advec_mom_kernel.advec_mom_yvel_kernel[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE)]\
                 (self.x_max, self.y_max, self.work_array_3, self.work_array_4, self.work_array_5, vel1)
            _append_diagnostics("advec_mom_yvel_kernel", timer.elapsed_time)


    def _reset_field(self):
        import reset_kernel
        timer=GPUTimer()
        with timer:
            reset_kernel.reset_field_kernel[(self.num_blocks_x, self.num_blocks_y),(BLOCK_SIZE, BLOCK_SIZE)]\
            (self.density0, self.density1, self.energy0, self.energy1, self.xvel0, self.xvel1, self.yvel0, self.yvel1)
        _append_diagnostics("reset_field_kernel", timer.elapsed_time)

cuda_chunk = CloverLeafCudaChunk()

def init_chunks(xmin, ymin, dx, dy):
    cuda_chunk._init_chunks(xmin, ymin,dx,dy)

def generate_chunks(number_of_states, density, energy, xvel, yvel, x_min, x_max, y_min, y_max, radius, geometry, g_rect, g_circ, g_point):
    cuda_chunk._generate_chunks(number_of_states, density, energy, xvel, yvel, x_min, x_max, y_min, y_max, radius, geometry, g_rect, g_circ, g_point)
def ideal_gas(predict):
    cuda_chunk._ideal_gas(predict)

def update_halo(depth,fields, neighbours):
    cuda_chunk._update_halo(depth, fields, neighbours)

def exchange_halo(depth, fields, neighbours, chunk):
    cuda_chunk._exchange_halos(depth, fields, chunk, neighbours)

def field_summary():
    cuda_chunk._ideal_gas(False)
    (vol, mass, ie, ke, press)= cuda_chunk._field_summary()
    vol=clover.clover_sum(numpy.array(vol, numpy.float64))
    mass=clover.clover_sum(numpy.array(mass, numpy.float64))
    press=clover.clover_sum(numpy.array(press, numpy.float64))
    ie=clover.clover_sum(numpy.array(ie, numpy.float64))
    ke=clover.clover_sum(numpy.array(ke, numpy.float64))
    if(data.parallel["boss"]):
        print("step {:1d} {:.2f} {:.4e} {:.4e} {:.4e} {:.2e} {:.2e} {:.2e}".format(definitions.step, vol, mass, mass/vol, press/vol, ie, ke, ie+ke))

def viscosity():
    cuda_chunk._viscosity()

def calc_dt():
    return cuda_chunk._calc_dt()

def PdV(predict,dtbyt):
    return cuda_chunk._PdV_kernel(predict,dtbyt)

def revert():
    cuda_chunk._revert()


def accelerate(dt):
    cuda_chunk._accelerate(dt)

def flux_calc(dt):
    cuda_chunk._flux_calc(dt)

def advec_cell(dr,swp_nmpr):
    cuda_chunk._advec_cell(dr,swp_nmpr)

def advec_mom(which_vel, sweep_number, direction):
    cuda_chunk._advec_mom(which_vel, sweep_number, direction)

def reset_field():
    cuda_chunk._reset_field()
