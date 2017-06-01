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
	while (state.KeepRunning())
	{
		// Set CUDA deivce
		cudaSetDevice(benchmarkingDevice);

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

		BM_addInt << <1, 1 >> >(d_a, d_b, d_c);

		// Move output value to the host
		cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

		// Free memory on the device
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);

		cudaDeviceReset();
	}
}

BENCHMARK(BM_CUDABasicLatencyTest)
	->MinTime(1.0);

__global__ void BM_convergedKernel(float* d_in, float* d_out)
{
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;

	float x = float(d_in[globalId]);
	float y = float(threadIdx.x);

	if (x <= 10.0f)
	{
		for (int i = 0; i < (gridDim.x * blockDim.x - 1); i++)
		{
			y = y + d_in[i];
		}
	}
	else
	{
		for (int i = 0; i < (gridDim.x * blockDim.x - 1); i++)
		{
			y = y - d_in[i];
		}
	}
}

static void BM_CUDAConvergedExecution(benchmark::State& state)
{
	// Dynamic Data Input
	int dataSize = state.range(0);

	// Set CUDA deivce
	cudaSetDevice(benchmarkingDevice);

	// Get max threads per block
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, benchmarkingDevice);
	int threadsPerBlock = prop.maxThreadsPerBlock;

	// Calculate Grid size
	int gridSize = int(dataSize / threadsPerBlock);

	// Generate input data
	std::vector<float> inputVector(dataSize, 0);
	for (int i = 0; i < dataSize; i++)
	{
		inputVector[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
	}
	float* h_input = &inputVector[0];

	// Allocate memory on device
	float *d_input, *d_output;
	cudaMalloc(&d_input, dataSize * sizeof(float));
	cudaMalloc(&d_output, dataSize * sizeof(float));

	// Copy data to device
	cudaMemcpy(d_input, h_input, dataSize * sizeof(float), cudaMemcpyHostToDevice);

	while (state.KeepRunning())
	{
		auto start = std::chrono::high_resolution_clock::now();

		BM_convergedKernel << <gridSize, threadsPerBlock >> >(d_input, d_output);

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(
				end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}


	// Copy Data from Host to Device
	std::vector<float> outputVector(dataSize, 0);
	float* h_output = &outputVector[0];
	cudaMemcpy(h_output, d_output, dataSize * sizeof(float), cudaMemcpyDeviceToHost);

	// Free memory and reset device
	cudaFree(d_input);
	cudaFree(d_output);
	cudaDeviceReset();

	// Calculate Elements processed
	state.SetBytesProcessed(state.iterations() * dataSize);
}

BENCHMARK(BM_CUDAConvergedExecution)
->MinTime(1.0)
->UseManualTime()
->Args({ 2048 })
->Args({ 8192 })
->Args({ 65536 })
->Args({ 524288 })
->Args({ 8388608 })
->Args({ 16777216 });

__global__ void BM_divergedKernel(float* d_in, float* d_out)
{
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;

	float x = float(d_in[globalId]);
	float y = float(threadIdx.x);

	if (x < 0.5f)
	{
		for (int i = 0; i < (gridDim.x * blockDim.x - 1); i++)
		{
			y = y + d_in[i];
		}
	}
	else
	{
		for (int i = 0; i < (gridDim.x * blockDim.x - 1); i++)
		{
			y = y - d_in[i];
		}
	}
}

static void BM_CUDADivergedExecution(benchmark::State& state)
{
	// Dynamic Data Input
	int dataSize = state.range(0);

	// Set CUDA deivce
	cudaSetDevice(benchmarkingDevice);

	// Get max threads per block
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, benchmarkingDevice);
	int threadsPerBlock = prop.maxThreadsPerBlock;

	// Calculate Grid size
	int gridSize = int(dataSize / threadsPerBlock);

	// Generate input data
	std::vector<float> inputVector(dataSize, 0);
	for (int i = 0; i < dataSize; i++)
	{
		inputVector[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
	}
	float* h_input = &inputVector[0];

	// Allocate memory on device
	float *d_input, *d_output;
	cudaMalloc(&d_input, dataSize * sizeof(float));
	cudaMalloc(&d_output, dataSize * sizeof(float));

	// Copy data to device
	cudaMemcpy(d_input, h_input, dataSize * sizeof(float), cudaMemcpyHostToDevice);

	while (state.KeepRunning())
	{
		auto start = std::chrono::high_resolution_clock::now();

		BM_divergedKernel << <gridSize, threadsPerBlock >> >(d_input, d_output);

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(
				end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}


	// Copy Data from Host to Device
	std::vector<float> outputVector(dataSize, 0);
	float* h_output = &outputVector[0];
	cudaMemcpy(h_output, d_output, dataSize * sizeof(float), cudaMemcpyDeviceToHost);

	// Free memory and reset device
	cudaFree(d_input);
	cudaFree(d_output);
	cudaDeviceReset();

	// Calculate Elements processed
	state.SetBytesProcessed(state.iterations() * dataSize);
}

BENCHMARK(BM_CUDADivergedExecution)
->MinTime(1.0)
->UseManualTime()
->Args({2048})
->Args({8192})
->Args({65536})
->Args({524288})
->Args({8388608})
->Args({16777216});

__global__ void BM_multiDivergedKernel(float* d_in, float* d_out)
{
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;

	float x = float(d_in[globalId]);
	float y = float(threadIdx.x);

	int threadId = threadIdx.x;

	switch (threadId)
	{
	case 1: y = 32;
		break;
	case 2: y = 33;
		break;
	case 3: y = 34;
		break;
	case 4: y = 35;
		break;
	case 5: y = 36;
		break;
	case 6: y = 37;
		break;
	case 7: y = 38;
		break;
	case 8: y = 39;
		break;
	case 9: y = 40;
		break;
	case 10: y = 41;
		break;
	default: y = 42;
		break;
	}

	y = y * x;

	d_out[globalId] = y;
}

static void BM_CUDAMultiDivergedExecution(benchmark::State& state)
{
	// Dynamic Data Input
	int dataSize = state.range(0);

	// Set CUDA deivce
	cudaSetDevice(benchmarkingDevice);

	// Get max threads per block
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, benchmarkingDevice);
	int threadsPerBlock = prop.maxThreadsPerBlock;

	// Calculate Grid size
	int gridSize = int(dataSize / threadsPerBlock);

	// Generate input data
	std::vector<float> inputVector(dataSize, 0);
	for (int i = 0; i < dataSize; i++)
	{
		inputVector[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
	}
	float* h_input = &inputVector[0];

	// Allocate memory on device
	float *d_input, *d_output;
	cudaMalloc(&d_input, dataSize * sizeof(float));
	cudaMalloc(&d_output, dataSize * sizeof(float));

	// Copy data to device
	cudaMemcpy(d_input, h_input, dataSize * sizeof(float), cudaMemcpyHostToDevice);

	while (state.KeepRunning())
	{
		auto start = std::chrono::high_resolution_clock::now();

		BM_divergedKernel << <gridSize, threadsPerBlock >> >(d_input, d_output);

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(
				end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}


	// Copy Data from Host to Device
	std::vector<float> outputVector(dataSize, 0);
	float* h_output = &outputVector[0];
	cudaMemcpy(h_output, d_output, dataSize * sizeof(float), cudaMemcpyDeviceToHost);

	// Free memory and reset device
	cudaFree(d_input);
	cudaFree(d_output);
	cudaDeviceReset();

	// Calculate Elements processed
	state.SetBytesProcessed(state.iterations() * dataSize);
}

BENCHMARK(BM_CUDAMultiDivergedExecution)
->MinTime(1.0)
->UseManualTime()
->Args({2048})
->Args({8192})
->Args({65536})
->Args({524288})
->Args({8388608})
->Args({16777216});

__global__ void BM_multFloat(float* d_in, float* d_out)
{
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;

	float x = d_in[globalId];
	float y = float(threadIdx.x);

	y = y * x;

	d_out[globalId] = y;
}

static void BM_CUDAFLOPS_GeneratedData(benchmark::State& state)
{
	// Dynamic Data Input
	int dataSize = state.range(0);

	// Set CUDA deivce
	cudaSetDevice(benchmarkingDevice);

	// Get max threads per block
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, benchmarkingDevice);
	int threadsPerBlock = prop.maxThreadsPerBlock;

	// Calculate Grid size
	int gridSize = int(dataSize / threadsPerBlock);

	// Generate input data
	std::vector<float> inputVector(dataSize, 0);
	for (int i = 0; i < dataSize; i++)
	{
		inputVector[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
	}
	float* h_input = &inputVector[0];

	// Allocate memory on device
	float *d_input, *d_output;
	cudaMalloc(&d_input, dataSize * sizeof(float));
	cudaMalloc(&d_output, dataSize * sizeof(float));

	// Copy data to device
	cudaMemcpy(d_input, h_input, dataSize * sizeof(float), cudaMemcpyHostToDevice);

	while (state.KeepRunning())
	{
		auto start = std::chrono::high_resolution_clock::now();

		BM_multFloat << <gridSize, threadsPerBlock >> >(d_input, d_output);

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(
				end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}


	// Copy Data from Host to Device
	std::vector<float> outputVector(dataSize, 0);
	float* h_output = &outputVector[0];
	cudaMemcpy(h_output, d_output, dataSize * sizeof(float), cudaMemcpyDeviceToHost);

	// Free memory and reset device
	cudaFree(d_input);
	cudaFree(d_output);
	cudaDeviceReset();

	// Calculate Elements processed
	state.SetBytesProcessed(state.iterations() * dataSize);
}

BENCHMARK(BM_CUDAFLOPS_GeneratedData)
->UseManualTime()
->MinTime(1.0)
->Args({ 2048 })
->Args({ 8192 })
->Args({ 65536 })
->Args({ 524288 })
->Args({ 8388608 })
->Args({ 16777216 })
->Args({ 33554432 });

__global__ void BM_multInt(int* d_in, int* d_out)
{
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;

	int x = d_in[globalId];
	int y = threadIdx.x;

	y = y * x;

	d_out[globalId] = y;
}

static void BM_CUDAIntOPS_GeneratedData(benchmark::State& state)
{
	// Dynamic Data Input
	int dataSize = state.range(0);

	// Set CUDA deivce
	cudaSetDevice(benchmarkingDevice);

	// Get max threads per block
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, benchmarkingDevice);
	int threadsPerBlock = prop.maxThreadsPerBlock;

	// Calculate Grid size
	int gridSize = int(dataSize / threadsPerBlock);

	// Generate input data
	std::vector<int> inputVector(dataSize, 0);
	for (int i = 0; i < dataSize; i++)
	{
		inputVector[i] = static_cast<int>(rand());
	}
	int* h_input = &inputVector[0];

	// Allocate memory on device
	int *d_input, *d_output;
	cudaMalloc(&d_input, dataSize * sizeof(int));
	cudaMalloc(&d_output, dataSize * sizeof(int));

	// Copy data to device
	cudaMemcpy(d_input, h_input, dataSize * sizeof(int), cudaMemcpyHostToDevice);

	while (state.KeepRunning())
	{
		auto start = std::chrono::high_resolution_clock::now();

		BM_multInt << <gridSize, threadsPerBlock >> >(d_input, d_output);

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(
				end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}


	// Copy Data from Host to Device
	std::vector<int> outputVector(dataSize, 0);
	int* h_output = &outputVector[0];
	cudaMemcpy(h_output, d_output, dataSize * sizeof(int), cudaMemcpyDeviceToHost);

	// Free memory and reset device
	cudaFree(d_input);
	cudaFree(d_output);
	cudaDeviceReset();

	// Calculate Elements processed
	state.SetBytesProcessed(state.iterations() * dataSize);
}

BENCHMARK(BM_CUDAIntOPS_GeneratedData)
->UseManualTime()
->MinTime(1.0)
->Args({ 2048 })
->Args({ 8192 })
->Args({ 65536 })
->Args({ 524288 })
->Args({ 8388608 })
->Args({ 16777216 })
->Args({ 33554432 });

__global__ void BM_multFloat2(float2* d_in, float2* d_out)
{
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;

	float2 x = d_in[globalId];
	float2 y = {1.01010101f, 1.01010101f};

	y.x = y.x * x.x;
	y.y = y.y * x.y;

	d_out[globalId] = y;
}

static void BM_CUDAFloat2OPS_GeneratedData(benchmark::State& state)
{
	// Dynamic Data Input
	int dataSize = state.range(0);

	// Set CUDA deivce
	cudaSetDevice(benchmarkingDevice);

	// Get max threads per block
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, benchmarkingDevice);
	int threadsPerBlock = prop.maxThreadsPerBlock;

	// Calculate Grid size
	int gridSize = int(dataSize / threadsPerBlock);

	// Generate input data
	std::vector<float2> inputVector(dataSize, {0, 0});
	for (int i = 0; i < dataSize; i++)
	{
		inputVector[i] = {static_cast<float>(rand()) * 1.010101f, static_cast<float>(rand()) * 1.010101f};
	}
	float2* h_input = &inputVector[0];

	// Allocate memory on device
	float2 *d_input, *d_output;
	cudaMalloc(&d_input, dataSize * sizeof(float2));
	cudaMalloc(&d_output, dataSize * sizeof(float2));

	// Copy data to device
	cudaMemcpy(d_input, h_input, dataSize * sizeof(float2), cudaMemcpyHostToDevice);

	while (state.KeepRunning())
	{
		auto start = std::chrono::high_resolution_clock::now();

		BM_multFloat2 << <gridSize, threadsPerBlock >> >(d_input, d_output);

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(
				end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}


	// Copy Data from Host to Device
	std::vector<float2> outputVector(dataSize, {0,0});
	float2* h_output = &outputVector[0];
	cudaMemcpy(h_output, d_output, dataSize * sizeof(float2), cudaMemcpyDeviceToHost);

	// Free memory and reset device
	cudaFree(d_input);
	cudaFree(d_output);
	cudaDeviceReset();

	// Calculate Elements processed
	state.SetBytesProcessed(state.iterations() * dataSize);
}

BENCHMARK(BM_CUDAFloat2OPS_GeneratedData)
->UseManualTime()
->MinTime(1.0)
->Args({ 2048 })
->Args({ 8192 })
->Args({ 65536 })
->Args({ 524288 })
->Args({ 8388608 })
->Args({ 16777216 })
->Args({ 33554432 });

static void BM_CUDABandwidthHostToDevice(benchmark::State& state)
{
	// Dynamic Data Input
	int dataSize = state.range(0);

	// Set CUDA deivce
	cudaSetDevice(benchmarkingDevice);

	// Get max threads per block
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, benchmarkingDevice);
	int threadsPerBlock = prop.maxThreadsPerBlock;

	// Generate input data
	std::vector<int> inputVector(dataSize, 0);
	int* h_input = &inputVector[0];

	// Allocate memory on device
	int* d_input;
	cudaMalloc(&d_input, dataSize * sizeof(int));

	while (state.KeepRunning())
	{
		auto start = std::chrono::high_resolution_clock::now();

		cudaMemcpy(d_input, h_input, dataSize * sizeof(int), cudaMemcpyHostToDevice);

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(
				end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}

	// Free memory and reset device
	cudaFree(d_input);
	cudaDeviceReset();

	// Calculate Elements processed
	state.SetBytesProcessed(state.iterations() * dataSize * sizeof(int));
}

BENCHMARK(BM_CUDABandwidthHostToDevice)
->UseManualTime()
->MinTime(1.0)
->Args({ 1 })
->Args({ 8 })
->Args({ 16 })
->Args({ 512 })
->Args({ 1024 })
->Args({ 16384 })
->Args({ 131072 })
->Args({ 1048576 })
->Args({ 8388608 })
->Args({ 16777216 })
->Args({ 33554432 })
->Args({ 67108864 })
->Args({ 134217728 });

static void BM_CUDABandwidthDeviceToHost(benchmark::State& state)
{
	// Dynamic Data Input
	int dataSize = state.range(0);

	// Set CUDA deivce
	cudaSetDevice(benchmarkingDevice);

	// Get max threads per block
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, benchmarkingDevice);
	int threadsPerBlock = prop.maxThreadsPerBlock;

	// Generate input data
	std::vector<int> inoutVector(dataSize, 0);
	int* h_inout = &inoutVector[0];

	// Allocate memory on device
	int* d_device;
	cudaMalloc(&d_device, dataSize * sizeof(int));

	// Copy Data from Host to Device
	cudaMemcpy(d_device, h_inout, dataSize * sizeof(int), cudaMemcpyHostToDevice);

	while (state.KeepRunning())
	{
		auto start = std::chrono::high_resolution_clock::now();

		cudaMemcpy(h_inout, d_device, dataSize * sizeof(int), cudaMemcpyDeviceToHost);

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(
				end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}

	// Free memory and reset device
	cudaFree(d_device);
	cudaDeviceReset();

	// Calculate Elements processed
	state.SetBytesProcessed(state.iterations() * dataSize * sizeof(int));
}

BENCHMARK(BM_CUDABandwidthHostToDevice)
->UseManualTime()
->MinTime(1.0)
->Args({ 1 })
->Args({ 8 })
->Args({ 16 })
->Args({ 512 })
->Args({ 1024 })
->Args({ 16384 })
->Args({ 131072 })
->Args({ 1048576 })
->Args({ 8388608 })
->Args({ 16777216 })
->Args({ 33554432 })
->Args({ 67108864 })
->Args({ 134217728 });

static void BM_CUDABandwidthDeviceToDevice(benchmark::State& state)
{
	// Dynamic Data Input
	int dataSize = state.range(0);

	// Set CUDA deivce
	cudaSetDevice(benchmarkingDevice);

	// Get max threads per block
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, benchmarkingDevice);
	int threadsPerBlock = prop.maxThreadsPerBlock;

	// Generate input data
	std::vector<int> inputVector(dataSize, 0);
	int* h_input = &inputVector[0];

	// Allocate memory on device
	int *d_device, *d_input;
	cudaMalloc(&d_device, dataSize * sizeof(int));
	cudaMalloc(&d_input, dataSize * sizeof(int));

	// Copy Data from Host to Device
	cudaMemcpy(d_input, h_input, dataSize * sizeof(int), cudaMemcpyHostToDevice);

	while (state.KeepRunning())
	{
		auto start = std::chrono::high_resolution_clock::now();

		cudaMemcpy(d_device, d_input, dataSize * sizeof(int), cudaMemcpyDeviceToDevice);

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(
				end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}

	// Free memory and reset device
	cudaFree(d_device);
	cudaDeviceReset();

	// Calculate Elements processed
	state.SetBytesProcessed(state.iterations() * dataSize * sizeof(int));
}

BENCHMARK(BM_CUDABandwidthDeviceToDevice)
->UseManualTime()
->MinTime(1.0)
->Args({ 1 })
->Args({ 8 })
->Args({ 16 })
->Args({ 512 })
->Args({ 1024 })
->Args({ 16384 })
->Args({ 131072 })
->Args({ 1048576 })
->Args({ 8388608 })
->Args({ 16777216 })
->Args({ 33554432 })
->Args({ 67108864 })
->Args({ 134217728 });

__global__ void BM_gridStrideKernel(const int n, const int* d_in, int* d_out)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = id; i < n; i += blockDim.x * gridDim.x)
	{
		d_out[i] = d_in[i] * id;
	}
}

static void BM_CUDAKernelCreation(benchmark::State& state)
{
	// Dynamic Data Input
	int kernels = state.range(0);
	int dataSize = state.range(1);
	int dataPerKernel = static_cast<int>(dataSize / kernels);

	// Set CUDA deivce
	cudaSetDevice(benchmarkingDevice);

	// Get max threads per block
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, benchmarkingDevice);
	int threadsPerBlock = prop.maxThreadsPerBlock;

	// Calculate Grid and Block size
	int blockSize = threadsPerBlock > kernels ? kernels : threadsPerBlock;
	int gridSize = int(kernels / blockSize);

	// Generate input data
	std::vector<int> inputVector(dataSize, 1);
	int* h_input = &inputVector[0];

	// Allocate memory on device
	int *d_input, *d_output;
	cudaMalloc(&d_input, dataSize * sizeof(int));
	cudaMalloc(&d_output, dataSize * sizeof(int));

	// Copy data to device
	cudaMemcpy(d_input, h_input, dataSize * sizeof(int), cudaMemcpyHostToDevice);

	while (state.KeepRunning())
	{
		auto start = std::chrono::high_resolution_clock::now();

		BM_gridStrideKernel << <gridSize, blockSize >> >(dataSize, d_input, d_output);

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(
				end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}


	// Copy Data from Host to Device
	std::vector<int> outputVector(dataSize, 0);
	int* h_output = &outputVector[0];
	cudaMemcpy(h_output, d_output, dataSize * sizeof(int), cudaMemcpyDeviceToHost);

	// Free memory and reset device
	cudaFree(d_input);
	cudaFree(d_output);
	cudaDeviceReset();

	// Calculate Elements processed
	state.SetBytesProcessed(state.iterations() * dataSize * sizeof(int));
}

BENCHMARK(BM_CUDAKernelCreation)
->UseManualTime()
->MinTime(1.0)
->Args({ 1, 1 })
->Args({ 1, 10 })
->Args({ 8, 8 })
->Args({ 8, 80 })
->Args({ 16, 16 })
->Args({ 16, 160 })
->Args({ 512, 512 })
->Args({ 512, 5120 })
->Args({ 1024, 1024 })
->Args({ 1024, 10240 })
->Args({ 16384, 16384 })
->Args({ 16384, 163840 })
->Args({ 131072, 131072 })
->Args({ 131072, 1310720 })
->Args({ 1048576, 1048576 })
->Args({ 1048576, 10485760 })
->Args({ 1048576, 104857600 })
->Args({ 131072, 131072 })
->Args({ 65536, 131072 })
->Args({ 32768, 131072 })
->Args({ 16384, 131072 })
->Args({ 8192, 131072 })
->Args({ 4096, 131072 })
->Args({ 2048, 131072 })
->Args({ 1024, 131072 });

__global__ void BM_badGridStrideKernel(const int objectsPerKernel, const int* d_in, int* d_out)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	int startElement = id * objectsPerKernel;

	for (int i = startElement; i < (startElement + objectsPerKernel); i++)
	{
		d_out[i] = d_in[i] * id;
	}
}

static void BM_CUDABadMemoryCoalescence(benchmark::State& state)
{
	// Dynamic Data Input
	int kernels = state.range(0);
	int dataSize = state.range(1);
	int dataPerKernel = static_cast<int>(dataSize / kernels);

	// Set CUDA deivce
	cudaSetDevice(benchmarkingDevice);

	// Get max threads per block
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, benchmarkingDevice);
	int threadsPerBlock = prop.maxThreadsPerBlock;

	// Calculate Grid and Block size
	int blockSize = threadsPerBlock > kernels ? kernels : threadsPerBlock;
	int gridSize = int(kernels / blockSize);

	// Generate input data
	std::vector<int> inputVector(dataSize, 1);
	int* h_input = &inputVector[0];

	// Allocate memory on device
	int *d_input, *d_output;
	cudaMalloc(&d_input, dataSize * sizeof(int));
	cudaMalloc(&d_output, dataSize * sizeof(int));

	// Copy data to device
	cudaMemcpy(d_input, h_input, dataSize * sizeof(int), cudaMemcpyHostToDevice);

	while (state.KeepRunning())
	{
		auto start = std::chrono::high_resolution_clock::now();

		BM_badGridStrideKernel << <gridSize, blockSize >> >(dataPerKernel, d_input, d_output);

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(
				end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}


	// Copy Data from Host to Device
	std::vector<int> outputVector(dataSize, 0);
	int* h_output = &outputVector[0];
	cudaMemcpy(h_output, d_output, dataSize * sizeof(int), cudaMemcpyDeviceToHost);

	// Free memory and reset device
	cudaFree(d_input);
	cudaFree(d_output);
	cudaDeviceReset();

	// Calculate Elements processed
	state.SetBytesProcessed(state.iterations() * dataSize * sizeof(int));
}

BENCHMARK(BM_CUDABadMemoryCoalescence)
->UseManualTime()
->MinTime(1.0)
->Args({512, 4096})
->Args({4096, 4096})
->Args({1024, 8192})
->Args({8192, 8192})
->Args({16384, 131072})
->Args({131072, 131072})
->Args({16384, 131072})
->Args({131072, 131072})
->Args({131072, 1048576})
->Args({1048576, 1048576})
->Args({131072, 1048576})
->Args({1048576, 1048576})
->Args({1048576, 8388608})
->Args({8388608, 8388608});

__global__ void BM_goodGridStrideKernel(const int n, const int* d_in, int* d_out)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = id; i < n; i += blockDim.x * gridDim.x)
	{
		d_out[i] = d_in[i] * id;
	}
}

static void BM_CUDAGoodMemoryCoalescence(benchmark::State& state)
{
	// Dynamic Data Input
	int kernels = state.range(0);
	int dataSize = state.range(1);
	int dataPerKernel = static_cast<int>(dataSize / kernels);

	// Set CUDA deivce
	cudaSetDevice(benchmarkingDevice);

	// Get max threads per block
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, benchmarkingDevice);
	int threadsPerBlock = prop.maxThreadsPerBlock;

	// Calculate Grid and Block size
	int blockSize = threadsPerBlock > kernels ? kernels : threadsPerBlock;
	int gridSize = int(kernels / blockSize);

	// Generate input data
	std::vector<int> inputVector(dataSize, 1);
	int* h_input = &inputVector[0];

	// Allocate memory on device
	int *d_input, *d_output;
	cudaMalloc(&d_input, dataSize * sizeof(int));
	cudaMalloc(&d_output, dataSize * sizeof(int));

	// Copy data to device
	cudaMemcpy(d_input, h_input, dataSize * sizeof(int), cudaMemcpyHostToDevice);

	while (state.KeepRunning())
	{
		auto start = std::chrono::high_resolution_clock::now();

		BM_goodGridStrideKernel << <gridSize, blockSize >> >(dataSize, d_input, d_output);

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(
				end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}


	// Copy Data from Host to Device
	std::vector<int> outputVector(dataSize, 0);
	int* h_output = &outputVector[0];
	cudaMemcpy(h_output, d_output, dataSize * sizeof(int), cudaMemcpyDeviceToHost);

	// Free memory and reset device
	cudaFree(d_input);
	cudaFree(d_output);
	cudaDeviceReset();

	// Calculate Elements processed
	state.SetBytesProcessed(state.iterations() * dataSize * sizeof(int));
}

BENCHMARK(BM_CUDAGoodMemoryCoalescence)
->UseManualTime()
->MinTime(1.0)
->Args({512, 4096})
->Args({4096, 4096})
->Args({1024, 8192})
->Args({8192, 8192})
->Args({16384, 131072})
->Args({131072, 131072})
->Args({16384, 131072})
->Args({131072, 131072})
->Args({131072, 1048576})
->Args({1048576, 1048576})
->Args({131072, 1048576})
->Args({1048576, 1048576})
->Args({1048576, 8388608})
->Args({8388608, 8388608});

void selectBenchmarkDevice()
{
	std::cout << "Select Benchmarking Device:\n";

	int nDevices;
	cudaGetDeviceCount(&nDevices);

	std::vector<int> allDevices;

	for (int i = 0; i < nDevices; i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);

		std::cout << "[" << i << "] " << prop.name << "\n";
	}

	std::cout << "\nSelect Device: ";

	int deviceSelection;
	if (autoSelectDevice == -1)
	{
		std::cin >> deviceSelection;
	}
	else
	{
		deviceSelection = autoSelectDevice;
	}

	benchmarkingDevice = deviceSelection;

	std::cout << "\nSelected Device: " << deviceSelection << "\n\n";
}

int main(int argc, char** argv)
{
	selectBenchmarkDevice();

	::benchmark::Initialize(&argc, argv);
	::benchmark::RunSpecifiedBenchmarks();
}
