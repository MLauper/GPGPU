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
			void kernel addInt(global const int* A, global const int* B, global int* C){       
				C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];                 
			}
		)CLC";
		sources.push_back({kernelCode.c_str() , kernelCode.length()});

		// Create Program with Source in the created Context and Build the Program
		cl::Program program(context, sources);
		program.build({ benchmarkingDevice });

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
		cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&> addInt(cl::Kernel(program, "addInt"));
		cl::EnqueueArgs eargs(queue, cl::NullRange, cl::NDRange(10), cl::NullRange);
		addInt(eargs, buffer_A, buffer_B, buffer_C).wait();

		// Output Data
		int C[10];

		// Copy Data from Device to Host
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * 10, C);
	}
}
BENCHMARK(BM_OpenCLBasicTest)
	->Unit(benchmark::kMillisecond)
	->MinTime(1.0);

static void BM_OpenCLConvergedExecution(benchmark::State& state)
{
	// Create Context on Device
	cl::Context context({ benchmarkingDevice });

	// Create Program source Object
	cl::Program::Sources sources;

	// Provide Kernel Code
	std::string kernelCode =
		R"CLC(
			void kernel multFloat(global const float* in, global float* out){
				
				float x = in[get_global_id(0)];
				float y = (float)get_local_id(0);

				if (x < 32768) {
					y = y * x;
				} else {
					y = y / x;
				}

				out[get_global_id(0)] = y;
			}
		)CLC";
	sources.push_back({ kernelCode.c_str() , kernelCode.length() });

	// Create Program with Source in the created Context and Build the Program
	cl::Program program(context, sources);
	program.build({ benchmarkingDevice });

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
BENCHMARK(BM_OpenCLConvergedExecution)
->Unit(benchmark::kMicrosecond)
->MinTime(1.0);

static void BM_OpenCLDivergedExecution(benchmark::State& state)
{
	// Create Context on Device
	cl::Context context({ benchmarkingDevice });

	// Create Program source Object
	cl::Program::Sources sources;

	// Provide Kernel Code
	std::string kernelCode =
		R"CLC(
			void kernel multFloat(global const float* in, global float* out){
				
				float x = in[get_global_id(0)];
				float y = (float)get_local_id(0);

				if (x % 2) {
					y = y * x;
				} else {
					y = y / x;
				}

				out[get_global_id(0)] = y;
			}
		)CLC";
	sources.push_back({ kernelCode.c_str() , kernelCode.length() });

	// Create Program with Source in the created Context and Build the Program
	cl::Program program(context, sources);
	program.build({ benchmarkingDevice });

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
BENCHMARK(BM_OpenCLDivergedExecution)
->Unit(benchmark::kMicrosecond)
->MinTime(1.0);

static void BM_OpenCLFLOPS(benchmark::State& state)
{
	// Create Context on Device
	cl::Context context({ benchmarkingDevice });

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
	sources.push_back({ kernelCode.c_str() , kernelCode.length() });

	// Create Program with Source in the created Context and Build the Program
	cl::Program program(context, sources);
	program.build({ benchmarkingDevice });

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
BENCHMARK(BM_OpenCLFLOPS)
->MinTime(1.0);

static void BM_OpenCLIntOPS(benchmark::State& state)
{
	// Create Context on Device
	cl::Context context({ benchmarkingDevice });

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
	sources.push_back({ kernelCode.c_str() , kernelCode.length() });

	// Create Program with Source in the created Context and Build the Program
	cl::Program program(context, sources);
	program.build({ benchmarkingDevice });

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
BENCHMARK(BM_OpenCLIntOPS)
->MinTime(1.0);

float dummyOut_BM_CPUFLOPs;
static void BM_CPUFLOPs(benchmark::State& state)
{
	float x = 1.01010101f;
	while (state.KeepRunning())
	{
		for (auto i = 0; i < 1000000; i++) {
			x *= 1.01010101f;
		}
	}
}
BENCHMARK(BM_CPUFLOPs)
	->Unit(benchmark::kNanosecond)
	->MinTime(1.0);

static void BM_OpenCLContextCreation(benchmark::State& state)
{
	while (state.KeepRunning()) {
		
	}
	state.SetLabel("fefe");
}


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
	std::cin >> deviceSelection;

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
