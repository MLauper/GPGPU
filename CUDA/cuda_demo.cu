#include "cuda_demo.h"
#include "cuda_helpers.h"
#include <iostream>

int main(int argc, char** argv)
{
	cuda_demo::device_memory::demonstrateDeviceMemory();

	return 0;
}

void cuda_demo::device_memory::demonstrateDeviceMemory()
{
	std::cout << "Device Memory Demo\n";
	std::cout << "==================\n\n";

	linear_memory::demonstrateLinearDeviceMemory();
	linear_memory::demonstrateSharedDeviceMemory();
}

void cuda_demo::device_memory::linear_memory::demonstrateLinearDeviceMemory()
{
	std::cout << "Linear Device Memory Demo\n";
	std::cout << "\tAdding numbers on the GPU:\n";

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

	// Calculate result on the device
	addInt << <1, 1 >> >(d_a, d_b, d_c);

	// Move output value to the host
	cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

	// Free memory on the device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	std::cout << "\t" << h_a << " + " << h_b << " = " << h_c << "\n";
}

void cuda_demo::device_memory::linear_memory::demonstrateSharedDeviceMemory()
{
	std::cout << "Shared Device Memory Demo\n";
	std::cout << "\tAdding numbers on the GPU via Shared Memory:\n";

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

	// Calculate result on the device
	addIntSharedMemory << <1, 1 >> >(d_a, d_b, d_c);

	// Move output value to the host
	cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

	// Free memory on the device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	std::cout << "\t" << h_a << " + " << h_b << " = " << h_c << "\n";
}
