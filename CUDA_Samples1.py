
import sys
import numba
import numpy
from numba import cuda
import numpy as np
import math
import cv2
import time

print("Python version:", sys.version)
print("Numba version:", numba.__version__)
print("Numpy version:", numpy.__version__)



def cpu_process(h_img,dst_cpu):
    rows,cols,channels = h_img.shape
    for row in range(rows):
        for col in range(cols):
            for channel in range(channels):
                color = h_img[row,col][channel] * 3.0 + 30
                if color > 255:
                    dst_cpu[row,col][channel] = 255
                elif color <0:
                    dst_cpu[row, col][channel] = 0
                else:
                    dst_cpu[row,col][channel] = color

@cuda.jit
def gpu_process(d_img):
    index_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    index_y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    rows, cols, channels = d_img.shape
    for channel in range(channels):
        color = d_img[index_x, index_y][channel] * 3.0 + 30
        if color > 255:
            d_img[index_x, index_y][channel] = 255
        elif color < 0:
            d_img[index_x, index_y][channel] = 0
        else:
            d_img[index_x, index_y][channel] = color


if __name__ == "__main__":

    img_path = "Wallpaper2.jpg"
    h_img = cv2.imread(img_path)
    dst_cpu = h_img.copy()
    dst_gpu = h_img.copy()

    # CPU process data
    cpu_start_time = time.time()
    cpu_process(h_img,dst_cpu)
    cpu_end_time = time.time()
    print("CPU process time:%s"%(str(cpu_end_time-cpu_start_time)))
    cv2.imwrite("verificationCPU.jpg",dst_cpu)
    print("CPU image process has been saved")

    # 1st CUDA initial
    rows, cols, channels = h_img.shape
    d_img = cuda.to_device(h_img)
    threadsPerblock = (12,12)  # error numba.cuda.cudadrv.driver.CudaAPIError: 【700】 Call to cuCtxSynchronize results in UNKNOWN_CUDA_ERROR would happen if threadsPerblock = (16,16)
    blockPergrid_x = int(math.ceil(rows/threadsPerblock[0]))
    blockPergrid_y = int(math.ceil(cols/threadsPerblock[1]))
    blocksPergrid = (blockPergrid_x,blockPergrid_y)
    cuda.synchronize()

    # 2nd start the GPU process
    gpu_start_time = time.time()
    gpu_process[blocksPergrid,threadsPerblock](d_img)
    #3rd copy the data from GPU to CPU
    dst_gpu = d_img.copy_to_host()
    gpu_end_time = time.time()

    #4th log the time duration
    print("GPU process time:%s"%(str(gpu_end_time-gpu_start_time)))
    cv2.imwrite("verificationGPU.jpg",dst_gpu)
    print("GPU image process has been saved")
    print("Process has been finished")


















