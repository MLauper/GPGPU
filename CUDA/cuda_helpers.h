#ifndef cuda_helpers
#define cuda_helpers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*! \file cuda_helpers.h
 *	\brief Provides helper functions and kernels for CUDA
 *	
 *	This header file contains helper functions for CUDA to make
 *	certain tasks easier and remove some overhead to make the 
 *	code cleaner. It also contains most of the kernels.
 */

/*! \brief Dummy CUDA kernel with no practical use
 *
 * This kernel does not run anything on the GPU, but can be used
 * to test if it is possible to properly lunch a kernel on the GPU.
 */
__global__ void dummyKernel();

/*! \brief CUDA Kernel to add two integers
 *
 * This kernel calculates the sum of two integers and writes the 
 * result in global memory.
 */
__global__ void addInt(int* a, int* b, int* c);

/*! \brief CUDA Kernel to add two integers with shared memory
 *
 * This kernel calculates the sum of two integers and writes the 
 * result in global memory. Before the calculation, it copies the 
 * values in shared memory.
 */
__global__ void addIntSharedMemory(int* a, int* b, int* c);

#endif // cuda_helpers