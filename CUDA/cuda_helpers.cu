#include "cuda_helpers.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int answerToEverything(){
	return 42;
}

__global__ void dummyKernel()
{
	return;
}

__global__ void addInt(int* d_a, int* d_b, int* d_c)
{
	*d_c = *d_a + *d_b;
	return;
}