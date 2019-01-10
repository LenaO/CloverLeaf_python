from numba import cuda, jit, float64, int32
import numpy as np
import math


@cuda.jit('void(float64[:,:],float64[:], int32, int32, int32, int32)')
def pack_bottom_buffer(array, buf, depth, x_inc, y_inc, x_max):
    glob_id  =  cuda.grid(1)
    row = int(glob_id / (x_max+4))
    column = glob_id %(x_max+4)
    if (column > 1 - depth) and (column <= x_max +1 +x_inc + depth):
        if row < depth:
            index= column + depth + row *((x_max + 1) + x_inc + (2 * depth)) - 2
            buf[index]=array[column][row+2 +y_inc]

@cuda.jit('void(float64[:,:],float64[:], int32, int32, int32, int32)')
def unpack_bottom_buffer(array, buf, depth, x_inc, y_inc, x_max):
    glob_id  =  cuda.grid(1)
    row = int(glob_id /(x_max+4))
    column = glob_id %(x_max+4)
    if (column > 1 - depth) and (column <= x_max +1 +x_inc + depth):
        if row < depth:
            index= column + depth + row *((x_max + 1) + x_inc + (2 * depth)) - 2
            array[column][1 -row]=buf[index]



@cuda.jit('void(float64[:,:],float64[:], int32, int32, int32, int32, int32)')
def pack_top_buffer(array, buf, depth,x_inc, y_inc, x_max, y_max):
    glob_id  =  cuda.grid(1)
    row = int(glob_id / (x_max+4))
    column = glob_id % (x_max+4)
    if (column > 1 - depth) and (column <= x_max +1 +x_inc + depth):
        if row < depth:
            index=column + depth + row *((x_max + 1) + x_inc + (2 * depth)) - 2
            buf[index]=array[column][y_max + 1 - row]

@cuda.jit('void(float64[:,:],float64[:], int32, int32,  int32, int32, int32)')
def unpack_top_buffer(array, buf, depth, x_inc, y_inc, x_max, y_max):
    glob_id  =  cuda.grid(1)
    row = int(glob_id / (x_max+4))
    column = glob_id % (x_max+4)
    if (column > 1 - depth) and (column <= x_max +1+ x_inc + depth):
        if row < depth:
            index=column + depth + row *((x_max + 1) + x_inc + (2 * depth)) - 2
            array[column][ (y_max + 2) + y_inc + 2*row]=buf[index]



@cuda.jit('void(float64[:,:],float64[:], int32, int32, int32,int32)')
def pack_left_buffer(array, buf, depth, x_inc, y_inc, y_max):
    glob_id  =  cuda.grid(1)
    row = int(glob_id / depth)
    column = glob_id % depth;
    if (row > 1 - depth) and (row < (y_max + 2) + y_inc + depth):
            index=column+  (row + depth - 1)*depth - 2
            buf[index]=array[x_inc+2+column][row]

@cuda.jit('void(float64[:,:],float64[:], int32, int32, int32, int32)')
def unpack_left_buffer(array, buf, depth, x_inc, y_inc, y_max):
    glob_id  =  cuda.grid(1)
    row = int(glob_id / depth)
    column = glob_id % depth;
    if (row > 1 - depth) and (row <  (y_max + 2) + y_inc + depth):
            index=column + (row + depth - 1)*depth - 2
            array[1-column][row] = buf[index]


@cuda.jit('void(float64[:,:],float64[:], int32, int32,int32, int32, int32)')
def pack_right_buffer(array, buf, depth, x_inc, y_inc, x_max, y_max):
    glob_id  =  cuda.grid(1)
    row = int(glob_id / depth)
    column = glob_id % depth
    if (row > 1 - depth) and (row < (y_max + 2) + y_inc + depth):
            index=column+ (row + depth - 1)*depth - 2
            buf[index]=array[x_max+1-column][row]


@cuda.jit('void(float64[:,:],float64[:], int32, int32,int32,int32,int32)')
def unpack_right_buffer(array, buf, depth, x_inc, y_inc, x_max, y_max):
    glob_id  =  cuda.grid(1)
    row = int(glob_id / depth)
    column = glob_id % depth;
    if (row > 1 - depth) and (row < (y_max + 2) + y_inc + depth):
            index=column+ (row + depth - 1)*depth - 2
            array[x_max+2+x_inc+column][row] = buf[index]
