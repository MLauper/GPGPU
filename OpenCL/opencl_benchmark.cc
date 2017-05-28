#include "opencl_benchmark.h"
#include <iostream>
#include <ratio>
#include <chrono>

cl::Device benchmarkingDevice;

static void BM_StringCreation(benchmark::State& state)
{
	while (state.KeepRunning())
		std::string empty_string;
}

// Register the function as a benchmark
//BENCHMARK(BM_StringCreation);

// Define another benchmark
static void BM_StringCopy(benchmark::State& state)
{
	std::string x = "hello";
	while (state.KeepRunning())
		std::string copy(x);
}

//BENCHMARK(BM_StringCopy);

static void BM_memcpy(benchmark::State& state)
{
	char* src = new char[state.range(0)];
	char* dst = new char[state.range(0)];
	memset(src, 'x', state.range(0));
	while (state.KeepRunning())
		memcpy(dst, src, state.range(0));
	state.SetBytesProcessed(int64_t(state.iterations()) *
		int64_t(state.range(0)));
	delete[] src;
	delete[] dst;
}

//BENCHMARK(BM_memcpy)->Arg(8)->Arg(64)->Arg(512)->Arg(1 << 10)->Arg(8 << 10);

static void BM_OpenCLBasicTest(benchmark::State& state)
{
	while (state.KeepRunning())
	{
		// Create Context on Device
		cl::Context context({benchmarkingDevice});

		// Create Program source Object
		cl::Program::Sources sources;

		// Provide Kernel Code
		std::string kernelCode =
			R"CLC(
			void kernel addIntBasic(global const int* A, global const int* B, global int* C){       
				C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];                 
			}
		)CLC";
		sources.push_back({kernelCode.c_str() , kernelCode.length()});

		// Create Program with Source in the created Context and Build the Program
		cl::Program program(context, sources);
		program.build({benchmarkingDevice});

		// Create Buffer Objects
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * 10);

		// Input Data
		int A[] = {1, 1, 2, 3, 4, 5, 6, 7, 8, 9};
		int B[] = {2, 1, 2, 0, 1, 2, 0, 1, 2, 0};

		// Create Command Queue
		cl::CommandQueue queue(context, benchmarkingDevice);

		// Copy Data from Host to Device
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * 10, A);
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int) * 10, B);

		// Execute the Kernel
		cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&> addIntBasic(cl::Kernel(program, "addIntBasic"));
		cl::EnqueueArgs eargs(queue, cl::NullRange, cl::NDRange(10), cl::NullRange);
		addIntBasic(eargs, buffer_A, buffer_B, buffer_C).wait();

		// Output Data
		int C[10];

		// Copy Data from Device to Host
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * 10, C);
	}
}

//BENCHMARK(BM_OpenCLBasicTest)
//	->Unit(benchmark::kMillisecond)
//	->MinTime(1.0);

static void BM_OpenCLConvergedExecution(benchmark::State& state)
{
	// Create Context on Device
	cl::Context context({benchmarkingDevice});

	// Create Program source Object
	cl::Program::Sources sources;

	// Provide Kernel Code
	std::string kernelCode =
		R"CLC(
			void kernel convergedKernel(global const float* in, global float* out){
				
				float x = in[get_global_id(0)];
				float y = (float)get_local_id(0);

				if (x < 0.5f) {
					y = (float)sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(y * x)))))));
				} else {
					y = (float)sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(y / x)))))));
				}

				out[get_global_id(0)] = y;
			}
		)CLC";
	sources.push_back({kernelCode.c_str() , kernelCode.length()});

	// Create Program with Source in the created Context and Build the Program
	cl::Program program(context, sources);
	program.build({benchmarkingDevice});

	// Create Buffer Objects
	cl::Buffer buffer_in(context, CL_MEM_READ_WRITE, sizeof(float) * 65536);
	cl::Buffer buffer_out(context, CL_MEM_READ_WRITE, sizeof(float) * 65536);

	// Create Command Queue
	cl::CommandQueue queue(context, benchmarkingDevice);

	// Copy Data from Host to Device
	int zeroes[65536] = {0};
	queue.enqueueWriteBuffer(buffer_in, CL_TRUE, 0, sizeof(float) * 65536, zeroes);

	size_t maxWorkGroupSize;
	benchmarkingDevice.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);

	cl::NDRange globalSize, localSize;
	globalSize = 65536;
	localSize = maxWorkGroupSize;

	cl::make_kernel<cl::Buffer&, cl::Buffer&> convergedKernel(cl::Kernel(program, "convergedKernel"));
	cl::EnqueueArgs eargs(queue, globalSize, localSize);

	while (state.KeepRunning())
	{
		convergedKernel(eargs, buffer_in, buffer_out).wait();
	}

	float output[65536];
	queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, sizeof(float) * 65536, output);
	queue.finish();
}

//BENCHMARK(BM_OpenCLConvergedExecution)
//->MinTime(1.0);

static void BM_OpenCLDivergedExecution(benchmark::State& state)
{
	// Create Context on Device
	cl::Context context({benchmarkingDevice});

	// Create Program source Object
	cl::Program::Sources sources;

	// Provide Kernel Code
	std::string kernelCode =
		R"CLC(
			void kernel divergedKernel(global const float* in, global float* out){
				
				float x = in[get_global_id(0)];
				float y = (float)get_local_id(0);

				if (x < 0.5f) {
					y = (float)sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(y * x)))))));
				} else {
					y = (float)sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(y / x)))))));
				}

				out[get_global_id(0)] = y;
			}
		)CLC";
	sources.push_back({kernelCode.c_str() , kernelCode.length()});

	// Create Program with Source in the created Context and Build the Program
	cl::Program program(context, sources);
	program.build({benchmarkingDevice});

	// Create Buffer Objects
	cl::Buffer buffer_in(context, CL_MEM_READ_WRITE, sizeof(float) * 65536);
	cl::Buffer buffer_out(context, CL_MEM_READ_WRITE, sizeof(float) * 65536);

	// Create Command Queue
	cl::CommandQueue queue(context, benchmarkingDevice);

	// Copy Data from Host to Device
	queue.enqueueWriteBuffer(buffer_in, CL_TRUE, 0, sizeof(float) * 65536, randomFloats);

	size_t maxWorkGroupSize;
	benchmarkingDevice.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);

	cl::NDRange globalSize, localSize;
	globalSize = 65536;
	localSize = maxWorkGroupSize;

	cl::make_kernel<cl::Buffer&, cl::Buffer&> divergedKernel(cl::Kernel(program, "divergedKernel"));
	cl::EnqueueArgs eargs(queue, globalSize, localSize);

	while (state.KeepRunning())
	{
		divergedKernel(eargs, buffer_in, buffer_out).wait();
	}

	float output[65536];
	queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, sizeof(float) * 65536, output);
	queue.finish();
}

//BENCHMARK(BM_OpenCLDivergedExecution)
//->MinTime(1.0);

static void BM_OpenCLFLOPS(benchmark::State& state)
{
	// Create Context on Device
	cl::Context context({benchmarkingDevice});

	// Create Program source Object
	cl::Program::Sources sources;

	// Provide Kernel Code
	std::string kernelCode =
		R"CLC(
			void kernel multFloat(global const float* in, global float* out){
				
				float x = in[get_global_id(0)];
				float y = (float)get_local_id(0);

				y = y * x;

				out[get_global_id(0)] = y;
			}
		)CLC";
	sources.push_back({kernelCode.c_str() , kernelCode.length()});

	// Create Program with Source in the created Context and Build the Program
	cl::Program program(context, sources);
	program.build({benchmarkingDevice});

	// Create Buffer Objects
	cl::Buffer buffer_in(context, CL_MEM_READ_WRITE, sizeof(float) * 65536);
	cl::Buffer buffer_out(context, CL_MEM_READ_WRITE, sizeof(float) * 65536);

	// Create Command Queue
	cl::CommandQueue queue(context, benchmarkingDevice);

	// Copy Data from Host to Device
	queue.enqueueWriteBuffer(buffer_in, CL_TRUE, 0, sizeof(float) * 65536, randomFloats);

	size_t maxWorkGroupSize;
	benchmarkingDevice.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);

	cl::NDRange globalSize, localSize;
	globalSize = 65536;
	localSize = maxWorkGroupSize;

	cl::make_kernel<cl::Buffer&, cl::Buffer&> multFloat(cl::Kernel(program, "multFloat"));
	cl::EnqueueArgs eargs(queue, globalSize, localSize);

	while (state.KeepRunning())
	{
		multFloat(eargs, buffer_in, buffer_out).wait();
	}

	float output[65536];
	queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, sizeof(float) * 65536, output);
	queue.finish();
}

//BENCHMARK(BM_OpenCLFLOPS)
//->MinTime(1.0);

static void BM_OpenCLIntOPS(benchmark::State& state)
{
	// Create Context on Device
	cl::Context context({benchmarkingDevice});

	// Create Program source Object
	cl::Program::Sources sources;

	// Provide Kernel Code
	std::string kernelCode =
		R"CLC(
			void kernel multInt(global const int* in, global int* out){
				
				float x = in[get_global_id(0)];
				float y = (int)get_local_id(0);

				y = y * x;

				out[get_global_id(0)] = y;
			}
		)CLC";
	sources.push_back({kernelCode.c_str() , kernelCode.length()});

	// Create Program with Source in the created Context and Build the Program
	cl::Program program(context, sources);
	program.build({benchmarkingDevice});

	// Create Buffer Objects
	cl::Buffer buffer_in(context, CL_MEM_READ_WRITE, sizeof(int) * 65536);
	cl::Buffer buffer_out(context, CL_MEM_READ_WRITE, sizeof(int) * 65536);

	// Create Command Queue
	cl::CommandQueue queue(context, benchmarkingDevice);

	// Copy Data from Host to Device
	queue.enqueueWriteBuffer(buffer_in, CL_TRUE, 0, sizeof(int) * 65536, randomIntegers);

	size_t maxWorkGroupSize;
	benchmarkingDevice.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);

	cl::NDRange globalSize, localSize;
	globalSize = 65536;
	localSize = maxWorkGroupSize;

	cl::make_kernel<cl::Buffer&, cl::Buffer&> multInt(cl::Kernel(program, "multInt"));
	cl::EnqueueArgs eargs(queue, globalSize, localSize);

	// Benchmark
	while (state.KeepRunning())
	{
		auto start = std::chrono::high_resolution_clock::now();

		multInt(eargs, buffer_in, buffer_out).wait();

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(
				end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}

	float output[65536];
	queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, sizeof(int) * 65536, output);
	queue.finish();
}

//BENCHMARK(BM_OpenCLIntOPS)
//->MinTime(1.0);

static void BM_OpenCLBandwidthHostToDevice(benchmark::State& state)
{
	// Transfer size
	int chunkSize = state.range(0);

	// Create Context on Device
	cl::Context context({benchmarkingDevice});

	// Create Buffer Object
	cl::Buffer buffer_in(context, CL_MEM_READ_WRITE, sizeof(int) * chunkSize);

	// Create Command Queue
	cl::CommandQueue queue(context, benchmarkingDevice);

	// Create input data
	std::vector<int> inputVector(chunkSize, 0);
	int* input = &inputVector[0];

	// Copy Data from Host to Device
	while (state.KeepRunning())
	{
		queue.enqueueWriteBuffer(buffer_in, CL_TRUE, 0, sizeof(int) * chunkSize, input);
	}

	state.SetBytesProcessed(state.iterations() * chunkSize * sizeof(int));
}

//BENCHMARK(BM_OpenCLBandwidthHostToDevice)
//->MinTime(1.0)
//->Args({ 1 })
//->Args({ 8 })
//->Args({ 16 })
//->Args({ 512 })
//->Args({ 1024 })
//->Args({ 16384 })
//->Args({ 131072 })
//->Args({ 1048576 })
//->Args({ 8388608 })
//->Args({ 16777216 })
//->Args({ 33554432 })
//->Args({ 67108864 })
//->Args({ 134217728 });

static void BM_OpenCLBandwidthDeviceToDevice(benchmark::State& state)
{
	// Transfer size
	int chunkSize = state.range(0);

	// Create Context on Device
	cl::Context context({benchmarkingDevice});

	// Create Buffer Object
	cl::Buffer buffer_in(context, CL_MEM_READ_WRITE, sizeof(int) * chunkSize);
	cl::Buffer buffer_device(context, CL_MEM_READ_WRITE, sizeof(int) * chunkSize);

	// Create Command Queue
	cl::CommandQueue queue(context, benchmarkingDevice);

	// Create input data
	std::vector<int> inputVector(chunkSize, 0);
	int* input = &inputVector[0];

	// Copy Data from Host to Device
	queue.enqueueWriteBuffer(buffer_in, CL_TRUE, 0, sizeof(int) * chunkSize, input);

	// Copy Data from Device to Device
	while (state.KeepRunning())
	{
		queue.enqueueCopyBuffer(buffer_in, buffer_device, 0, 0, sizeof(int) * chunkSize);
		queue.finish();
	}

	state.SetBytesProcessed(state.iterations() * chunkSize * sizeof(int));
}

//BENCHMARK(BM_OpenCLBandwidthDeviceToDevice)
//->MinTime(1.0)
//->Args({ 1 })
//->Args({ 8 })
//->Args({ 16 })
//->Args({ 512 })
//->Args({ 1024 })
//->Args({ 16384 })
//->Args({ 131072 })
//->Args({ 1048576 })
//->Args({ 8388608 })
//->Args({ 16777216 })
//->Args({ 33554432 })
//->Args({ 67108864 })
//->Args({ 134217728 });

static void BM_OpenCLBandwidthDeviceToHost(benchmark::State& state)
{
	// Transfer size
	int chunkSize = state.range(0);

	// Create Context on Device
	cl::Context context({benchmarkingDevice});

	// Create Buffer Object
	cl::Buffer buffer_inout(context, CL_MEM_READ_WRITE, sizeof(int) * chunkSize);

	// Create Command Queue
	cl::CommandQueue queue(context, benchmarkingDevice);

	// Create input data
	std::vector<int> inputVector(chunkSize, 0);
	int* inout = &inputVector[0];

	// Write data once to device
	queue.enqueueWriteBuffer(buffer_inout, CL_TRUE, 0, sizeof(int) * chunkSize, inout);

	// Copy Data from Host to Device
	while (state.KeepRunning())
	{
		queue.enqueueReadBuffer(buffer_inout, CL_TRUE, 0, sizeof(int) * chunkSize, inout);
	}

	state.SetBytesProcessed(state.iterations() * chunkSize * sizeof(int));
}

//BENCHMARK(BM_OpenCLBandwidthDeviceToHost)
//->MinTime(1.0)
//->Args({ 1 })
//->Args({ 8 })
//->Args({ 16 })
//->Args({ 512 })
//->Args({ 1024 })
//->Args({ 16384 })
//->Args({ 131072 })
//->Args({ 1048576 })
//->Args({ 8388608 })
//->Args({ 16777216 })
//->Args({ 33554432 })
//->Args({ 67108864 })
//->Args({ 134217728 });

static void BM_OpenCLKernelCreation(benchmark::State& state)
{
	int kernels = state.range(0);
	int data = state.range(1);
	int dataPerKernel = static_cast<int>(data / kernels);
	std::cout << "Kernels: " << kernels << ", data: " << data << ", dataPerKernel: " << dataPerKernel << "\n";

	// Create Context on Device
	cl::Context context({benchmarkingDevice});

	// Create Program source Object
	cl::Program::Sources sources;

	// Provide Kernel Code
	std::string kernelCode =
		R"CLC(
			void kernel gridStrideKernel(const int objectsPerKernel, global const int* input, global int* output){
				
				int id = get_global_id(0);
				int startElement = id * objectsPerKernel;
				
				for (int i = startElement; i < (startElement+objectsPerKernel); i++){
					output[i] = input[i] * id;
				}
			}
		)CLC";
	sources.push_back({kernelCode.c_str() , kernelCode.length()});

	// Create Program with Source in the created Context and Build the Program
	cl::Program program(context, sources);
	program.build({benchmarkingDevice});

	// Create Buffer Objects
	cl::Buffer input(context, CL_MEM_READ_WRITE, sizeof(int) * data);
	cl::Buffer output(context, CL_MEM_READ_WRITE, sizeof(int) * data);

	// Input Data
	std::vector<int> inputVector(data, 1);
	int* inputData = &inputVector[0];

	// Create Command Queue
	cl::CommandQueue queue(context, benchmarkingDevice);

	// Copy Data from Host to Device
	queue.enqueueWriteBuffer(input, CL_TRUE, 0, sizeof(int) * data, inputData);

	// Get global and local size
	size_t maxWorkGroupSize;
	benchmarkingDevice.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);

	cl::NDRange globalSize, localSize;
	globalSize = kernels;
	localSize = maxWorkGroupSize>kernels ? kernels : maxWorkGroupSize;

	// Create Kernel
	cl::make_kernel<int, cl::Buffer&, cl::Buffer&> gridStrideKernel(cl::Kernel(program, "gridStrideKernel"));
	cl::EnqueueArgs eargs(queue, globalSize, localSize);
	//std::cout << "Global Size: " << kernels << ", Local Size: " << (maxWorkGroupSize>kernels ? kernels : maxWorkGroupSize) << "\n";

	while (state.KeepRunning())
	{
		gridStrideKernel(eargs, dataPerKernel, input, output).wait();
	}

	// Ouptut Data
	std::vector<int> outputVector(data, 0);
	int* outputData = &inputVector[0];

	// Copy Data from Device to Host
	queue.enqueueReadBuffer(output, CL_TRUE, 0, sizeof(int) * data, outputData);

	// Report speed
	state.SetBytesProcessed(state.iterations() * data * sizeof(int));

	for (auto i = 0; i < 10; i++)
	{
		std::cout << outputData[i] << " ";
	}
	std::cout << "\n";
}

BENCHMARK(BM_OpenCLKernelCreation)
->MinTime(1.0)
->Args({ 1, 1 })
->Args({ 1, 10 })
->Args({ 8, 8 })
->Args({ 8, 80 })
->Args({16, 16})
->Args({16, 160})
->Args({512, 512})
->Args({512, 5120})
->Args({ 1024, 1024 })
->Args({ 1024, 10240 })
->Args({ 16384, 16384 })
->Args({ 16384, 163840 })
->Args({ 131072, 131072 })
->Args({ 131072, 1310720 })
->Args({ 1048576, 1048576 })
->Args({ 1048576, 10485760 })
->Args({ 1048576, 104857600 })
->Args({ 134217728, 134217728 })
->Args({ 67108864, 134217728 })
->Args({ 33554432, 134217728 })
->Args({ 16777216, 134217728 })
->Args({ 8388608, 134217728 })
->Args({ 4194304, 134217728 })
->Args({ 2097152, 134217728 })
->Args({ 1048576, 134217728 })
->Args({ 524288, 134217728 })
->Args({ 262144, 134217728 })
->Args({ 131072, 134217728 })
->Args({ 65536, 134217728 })
->Args({ 32768, 134217728 })
->Args({ 16384, 134217728 })
->Args({ 8192, 134217728 })
->Args({ 4096, 134217728 })
->Args({ 2048, 134217728 })
->Args({ 1024, 134217728 });
/* The following Kernels overloaded my GPU and caused a crash of the system
->Args({ 512, 134217728 })
->Args({ 256, 134217728 })
->Args({ 128, 134217728 })
->Args({ 64, 134217728 })
->Args({ 32, 134217728 })
->Args({ 16, 134217728 })
->Args({ 8, 134217728 })
->Args({ 4, 134217728 })
->Args({ 2, 134217728 })
->Args({ 1, 134217728 })*/


float dummyOut_BM_CPUFLOPs;
static void BM_CPUFLOPs(benchmark::State& state)
{
	float x = 1.01010101f;
	while (state.KeepRunning())
	{
		for (auto i = 0; i < 1000000; i++)
		{
			x *= 1.01010101f;
		}
	}
}

//BENCHMARK(BM_CPUFLOPs)
//	->Unit(benchmark::kNanosecond)
//	->MinTime(1.0);

static void BM_OpenCLContextCreation(benchmark::State& state)
{
	while (state.KeepRunning())
	{
	}
	state.SetLabel("fefe");
}


int autoSelectDevice = 0;

void selectBenchmarkDevice()
{
	std::cout << "Select Benchmarking Device:\n";

	std::vector<cl::Device> allDevices;

	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	int i = 0;
	for (auto platform : platforms)
	{
		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

		std::string platformName;
		platform.getInfo(CL_PLATFORM_NAME, &platformName);

		for (auto device : devices)
		{
			std::string deviceName;
			device.getInfo(CL_DEVICE_NAME, &deviceName);

			std::cout << "[" << i++ << "] " << platformName << ": " << deviceName << "\n";
			allDevices.push_back(device);
		}
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

	benchmarkingDevice = allDevices[deviceSelection];
	std::string deviceName;
	benchmarkingDevice.getInfo(CL_DEVICE_NAME, &deviceName);

	std::cout << "\nSelected Device: " << deviceName << "\n\n";
}

int main(int argc, char** argv)
{
	selectBenchmarkDevice();

	::benchmark::Initialize(&argc, argv);
	::benchmark::RunSpecifiedBenchmarks();
}
