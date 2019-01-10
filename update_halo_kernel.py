from numba import cuda, jit, float64, int32
import numpy as np
import math
CELL_DATA=1
@cuda.jit('void(int32,int32,int32, int32, int32, float64[:,:],int32)')
def update_halo_kernel_bottom( x_max,y_max,x_extra, y_invert,grid_type, cur_array, depth):
    column, row= cuda.grid(2)
    #offset by 1 if it is anything but a CELL grid
    b_offset = 0
    if not grid_type == CELL_DATA: b_offset=1
    if column >= 2 - depth and column <= (x_max + 1) + x_extra + depth:
        if (row < depth):
            offset = 2 + b_offset
            cur_array[column, 1 -  row] =\
                y_invert * cur_array[column, row+offset]

@cuda.jit('void(int32,int32,int32,int32, int32, int32, float64[:,:],int32)')
def update_halo_kernel_top(x_max, y_max,\
 x_extra, y_extra, y_invert, x_face, cur_array,depth):
    column, row = cuda.grid(2)
    # if x face data, offset source/dest by - 1
    x_f_offset = 0
    if x_face: x_f_offset=1
    if column >= 2 - depth and column <= (x_max + 1) + x_extra + depth:
        if row < depth:
            offset = (- row) * 2 - 1 - x_f_offset
            cur_array[column, row+y_extra+y_max+2] =\
                y_invert * cur_array[column, row+y_max + 2 + offset]
@cuda.jit('void(int32,int32,int32,int32,int32, float64[:,:],int32)')
def update_halo_kernel_left(x_max, y_max, x_invert, y_extra, grid_type,\
cur_array, depth):

    column, row = cuda.grid(2)
    # offset by 1 if it is anything but a CELL grid
    l_offset = 0
    if grid_type != CELL_DATA:  l_offset= 1

    if row >= 2 - depth and row <= (y_max + 1) + y_extra + depth:
        if column<depth:
            cur_array[(1 - column), row] = x_invert * cur_array[2 + column + l_offset,row]

@cuda.jit('void(int32,int32,int32,int32,int32,int32, float64[:,:],int32)')
def update_halo_kernel_right( x_max, y_max, x_extra, y_extra, x_invert, y_face,\
cur_array, depth):
    # offset source by -1 if its a y face
    y_f_offset =  0
    if y_face: y_f_offset=1
    column, row = cuda.grid(2)
    if row >= 2 - depth and row <= (y_max + 1) + y_extra + depth :
        if column<depth:
            cur_array[x_max + 2 + x_extra + column,row] = \
            x_invert * cur_array[x_max + 1 - (column + y_f_offset),row]
