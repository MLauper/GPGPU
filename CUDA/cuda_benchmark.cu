#include "cuda_benchmark.h"
#include <iostream>
#include <ratio>
#include <chrono>

__global__ void BM_addInt(int* d_a, int* d_b, int* d_c)
{
	*d_c = *d_a + *d_b;
}
static void BM_CUDABasicLatencyTest(benchmark::State& state)
{
	// Setting Host Memory Variables
	auto h_a = 1, h_b = 1, h_c = 0;

	// Reserve pointers on Host and allocate memory on device
	int *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, sizeof(int));
	cudaMalloc(&d_b, sizeof(int));
	cudaMalloc(&d_c, sizeof(int));

	// Move input values to the device
	cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice);

	while(state.KeepRunning())
	{
		BM_addInt << <1, 1 >> >(d_a, d_b, d_c);
	}

	// Move output value to the host
	cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

	// Free memory on the device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

BENCHMARK(BM_CUDABasicLatencyTest)
->Unit(benchmark::kMillisecond)
->MinTime(1.0);


int main(int argc, char** argv)
{
	::benchmark::Initialize(&argc, argv);
	::benchmark::RunSpecifiedBenchmarks();
}
