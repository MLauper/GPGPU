#include "cuda_helpers.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void dummyKernel()
{
	return;
}

__global__ void addInt(int* d_a, int* d_b, int* d_c)
{
	*d_c = *d_a + *d_b;
}

__global__ void addIntSharedMemory(int* d_a, int* d_b, int* d_c)
{
	__shared__ int a_shared;
	__shared__ int b_shared;
	__shared__ int c_shared;

	a_shared = *d_a;
	b_shared = *d_b;

	c_shared = a_shared + b_shared;
	
	*d_c = c_shared;
}
