#include "opencl_benchmark.h"
#include <iostream>

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

static void BM_OpenCLInitialization(benchmark::State& state)
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
BENCHMARK(BM_OpenCLInitialization)->Unit(benchmark::kMillisecond);

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
