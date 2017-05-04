#ifndef cuda_helpers
#define cuda_helpers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int answerToEverything();

__global__ void dummyKernel();
__global__ void addInt(int* a, int* b, int* c);
__global__ void addIntSharedMemory(int* a, int* b, int* c);

#endif // cuda_helpers