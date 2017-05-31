#include "opencl_benchmark.h"
#include <iostream>
#include <ratio>
#include <chrono>

cl::Device benchmarkingDevice;

static void BM_OpenCLBasicLatencyTest(benchmark::State& state)
{
	while (state.KeepRunning())
	{
		// Create Context on Device
		cl::Context context(benchmarkingDevice);

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

BENCHMARK(BM_OpenCLBasicLatencyTest)
	->MinTime(1.0);

static void BM_OpenCLConvergedExecution(benchmark::State& state)
{
	// Dynamic Data Input
	int dataSize = state.range(0);

	// Create Context on Device
	cl::Context context(benchmarkingDevice);

	// Create Program source Object
	cl::Program::Sources sources;

	// Provide Kernel Code
	std::string kernelCode =
		R"CLC(
			void kernel convergedKernel(global const float* in, global float* out){
				
				float x = in[get_global_id(0)];
				float y = (float)get_local_id(0);

				if (x <= 10.0f) {
					for (int i = 0; i < (get_work_dim() - 1); i++){
						y = y + in[i];
					}
				} else {
					for (int i = 0; i < (get_work_dim() - 1); i++){
						y = y - in[i];
					}
				}

				out[get_global_id(0)] = y;
			}
		)CLC";
	sources.push_back({ kernelCode.c_str() , kernelCode.length() });

	// Create Program with Source in the created Context and Build the Program
	cl::Program program(context, sources);
	program.build({ benchmarkingDevice });

	// Create Buffer Objects
	cl::Buffer buffer_in(context, CL_MEM_READ_WRITE, sizeof(float) * dataSize);
	cl::Buffer buffer_out(context, CL_MEM_READ_WRITE, sizeof(float) * dataSize);

	// Create Command Queue
	cl::CommandQueue queue(context, benchmarkingDevice);

	// Generate input data
	std::vector<cl_float> inputVector(dataSize, 0);
	for (int i = 0; i < dataSize; i++)
	{
		inputVector[i] = static_cast <float> (rand()) / static_cast<float>(RAND_MAX);
	}
	cl_float* input = &inputVector[0];

	// Copy Data from Host to Device
	queue.enqueueWriteBuffer(buffer_in, CL_TRUE, 0, sizeof(float) * dataSize, input);

	size_t maxWorkGroupSize;
	benchmarkingDevice.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);

	cl::NDRange globalSize, localSize;
	globalSize = dataSize;
	localSize = maxWorkGroupSize;

	cl::make_kernel<cl::Buffer&, cl::Buffer&> convergedKernel(cl::Kernel(program, "convergedKernel"));
	cl::EnqueueArgs eargs(queue, globalSize, localSize);

	while (state.KeepRunning())
	{
		auto start = std::chrono::high_resolution_clock::now();

		convergedKernel(eargs, buffer_in, buffer_out).wait();

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(
				end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}

	std::vector<float> outputVector(dataSize, 0);
	float* output = &outputVector[0];
	queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, sizeof(cl_float) * dataSize, output);
	queue.finish();

	// The returned "Bytes" processed is realy the number of Elements Processed!
	state.SetBytesProcessed(state.iterations()*dataSize);
}

BENCHMARK(BM_OpenCLConvergedExecution)
->MinTime(1.0)
->UseManualTime()
->Args({ 2048 })
->Args({ 8192 })
->Args({ 65536 })
->Args({ 524288 })
->Args({ 8388608 })
->Args({ 16777216 })
->Args({ 67108864 });

static void BM_OpenCLDivergedExecution(benchmark::State& state)
{
	// Dynamic Data Input
	int dataSize = state.range(0);

	// Create Context on Device
	cl::Context context(benchmarkingDevice);

	// Create Program source Object
	cl::Program::Sources sources;

	// Provide Kernel Code
	std::string kernelCode =
		R"CLC(
			void kernel divergedKernel(global const float* in, global float* out){
				
				float x = in[get_global_id(0)];
				float y = (float)get_local_id(0);

				if (x < 0.5f) {
					for (int i = 0; i < (get_work_dim() - 1); i++){
						y = y + in[i];
					}
				} else {
					for (int i = 0; i < (get_work_dim() - 1); i++){
						y = y - in[i];
					}
				}

				out[get_global_id(0)] = y;
			}
		)CLC";
	sources.push_back({kernelCode.c_str() , kernelCode.length()});

	// Create Program with Source in the created Context and Build the Program
	cl::Program program(context, sources);
	program.build({benchmarkingDevice});

	// Create Buffer Objects
	cl::Buffer buffer_in(context, CL_MEM_READ_WRITE, sizeof(float) * dataSize);
	cl::Buffer buffer_out(context, CL_MEM_READ_WRITE, sizeof(float) * dataSize);

	// Create Command Queue
	cl::CommandQueue queue(context, benchmarkingDevice);

	// Generate input data
	std::vector<cl_float> inputVector(dataSize, 0);
	for (int i = 0; i < dataSize; i++)
	{
		inputVector[i] = static_cast <float> (rand()) / static_cast<float>(RAND_MAX);
	}
	cl_float* input = &inputVector[0];

	// Copy Data from Host to Device
	queue.enqueueWriteBuffer(buffer_in, CL_TRUE, 0, sizeof(float) * dataSize, input);

	size_t maxWorkGroupSize;
	benchmarkingDevice.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);

	cl::NDRange globalSize, localSize;
	globalSize = dataSize;
	localSize = maxWorkGroupSize;

	cl::make_kernel<cl::Buffer&, cl::Buffer&> divergedKernel(cl::Kernel(program, "divergedKernel"));
	cl::EnqueueArgs eargs(queue, globalSize, localSize);

	while (state.KeepRunning())
	{
		auto start = std::chrono::high_resolution_clock::now();

		divergedKernel(eargs, buffer_in, buffer_out).wait();

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(
				end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}

	std::vector<float> outputVector(dataSize, 0);
	float* output = &outputVector[0];
	queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, sizeof(cl_float) * dataSize, output);
	queue.finish();

	// The returned "Bytes" processed is realy the number of Elements Processed!
	state.SetBytesProcessed(state.iterations()*dataSize);
}

BENCHMARK(BM_OpenCLDivergedExecution)
->UseManualTime()
->MinTime(1.0)
->Args({ 2048 })
->Args({ 8192 })
->Args({ 65536 })
->Args({ 524288 })
->Args({ 8388608 })
->Args({ 16777216 })
->Args({ 67108864 });

static void BM_OpenCLFLOPS_RandomData(benchmark::State& state)
{
	// Create Context on Device
	cl::Context context(benchmarkingDevice);

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
		auto start = std::chrono::high_resolution_clock::now();

		multFloat(eargs, buffer_in, buffer_out).wait();

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(
				end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}

	float output[65536];
	queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, sizeof(float) * 65536, output);
	queue.finish();
}

BENCHMARK(BM_OpenCLFLOPS_RandomData)
->UseManualTime()
->MinTime(1.0);

static void BM_OpenCLFLOPS_GeneratedData(benchmark::State& state)
{
	// Dynamic Data Input
	int dataSize = state.range(0);

	// Create Context on Device
	cl::Context context(benchmarkingDevice);

	// Create Program source Object
	cl::Program::Sources sources;

	// Provide Kernel Code
	std::string kernelCode =
		R"CLC(
			void kernel multFloat(global const float* in, global float* out){
				
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
	cl::Buffer buffer_in(context, CL_MEM_READ_WRITE, sizeof(cl_float) * dataSize);
	cl::Buffer buffer_out(context, CL_MEM_READ_WRITE, sizeof(cl_float) * dataSize);

	// Create Command Queue
	cl::CommandQueue queue(context, benchmarkingDevice);

	// Generate Input Data
	std::vector<cl_float> inputVector(dataSize, 0);
	for (int i = 0; i < dataSize; i++)
	{
		inputVector[i] = static_cast<cl_float>(rand()) * 1.01010101f;
	}
	cl_float* input = &inputVector[0];

	// Copy Data from Host to Device
	queue.enqueueWriteBuffer(buffer_in, CL_TRUE, 0, sizeof(cl_float) * dataSize, input);

	size_t maxWorkGroupSize;
	benchmarkingDevice.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);

	cl::NDRange globalSize, localSize;
	globalSize = dataSize;
	localSize = maxWorkGroupSize;
	//std::cout << "Data Size: " << dataSize << " maxWorkGroupSize: " << maxWorkGroupSize << "\n";

	cl::make_kernel<cl::Buffer&, cl::Buffer&> multFloat(cl::Kernel(program, "multFloat"));
	cl::EnqueueArgs eargs(queue, globalSize, localSize);

	// Benchmark
	while (state.KeepRunning())
	{
		auto start = std::chrono::high_resolution_clock::now();

		multFloat(eargs, buffer_in, buffer_out).wait();

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(
				end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}

	std::vector<cl_float> outputVector(dataSize, 0);
	cl_float* output = &outputVector[0];
	queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, sizeof(cl_float) * dataSize, output);
	queue.finish();

	// The returned "Bytes" processed is realy the number of Integers Processed!
	state.SetBytesProcessed(state.iterations()*dataSize);
}

BENCHMARK(BM_OpenCLFLOPS_GeneratedData)
->UseManualTime()
->MinTime(1.0)
->Args({ 2048 })
->Args({ 8192 })
->Args({ 65536 })
->Args({ 524288 })
->Args({ 8388608 })
->Args({ 16777216 })
->Args({ 33554432 })
->Args({ 67108864 })
->Args({ 134217728 });

static void BM_OpenCLIntOPS_RandomData(benchmark::State& state)
{
	// Create Context on Device
	cl::Context context(benchmarkingDevice);

	// Create Program source Object
	cl::Program::Sources sources;

	// Provide Kernel Code
	std::string kernelCode =
		R"CLC(
			void kernel multInt(global const int* in, global int* out){

				int x = in[get_global_id(0)];
				int y = (int)get_local_id(0);

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

	int output[65536];
	queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, sizeof(int) * 65536, output);
	queue.finish();
}

BENCHMARK(BM_OpenCLIntOPS_RandomData)
->UseManualTime()
->MinTime(1.0);

static void BM_OpenCLIntOPS_GeneratedData(benchmark::State& state)
{
	// Dynamic Data Input
	int dataSize = state.range(0);

	// Create Context on Device
	cl::Context context(benchmarkingDevice);

	// Create Program source Object
	cl::Program::Sources sources;

	// Provide Kernel Code
	std::string kernelCode =
		R"CLC(
			void kernel multInt(global const int* in, global int* out){
				
				int x = in[get_global_id(0)];
				int y = (int)get_local_id(0);

				y = y * x;

				out[get_global_id(0)] = y;
			}
		)CLC";
	sources.push_back({ kernelCode.c_str() , kernelCode.length() });

	// Create Program with Source in the created Context and Build the Program
	cl::Program program(context, sources);
	program.build({ benchmarkingDevice });

	// Create Buffer Objects
	cl::Buffer buffer_in(context, CL_MEM_READ_WRITE, sizeof(cl_int) * dataSize);
	cl::Buffer buffer_out(context, CL_MEM_READ_WRITE, sizeof(cl_int) * dataSize);

	// Create Command Queue
	cl::CommandQueue queue(context, benchmarkingDevice);

	// Generate Input Data
	std::vector<cl_int> inputVector(dataSize, 0);
	for (int i = 0; i < dataSize; i++)
	{
		inputVector[i] = static_cast<cl_int>(rand());
	}
	int* input = &inputVector[0];

	// Copy Data from Host to Device
	queue.enqueueWriteBuffer(buffer_in, CL_TRUE, 0, sizeof(cl_int) * dataSize, input);

	size_t maxWorkGroupSize;
	benchmarkingDevice.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);

	cl::NDRange globalSize, localSize;
	globalSize = dataSize;
	localSize = maxWorkGroupSize;
	//std::cout << "Data Size: " << dataSize << " maxWorkGroupSize: " << maxWorkGroupSize << "\n";

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

	std::vector<cl_int> outputVector(dataSize, 0);
	int* output = &outputVector[0];
	queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, sizeof(cl_int) * dataSize, output);
	queue.finish();

	// The returned "Bytes" processed is realy the number of Integers Processed!
	state.SetBytesProcessed(state.iterations()*dataSize);
}

BENCHMARK(BM_OpenCLIntOPS_GeneratedData)
->UseManualTime()
->MinTime(1.0)
->Args({ 2048 })
->Args({ 8192 })
->Args({ 65536 })
->Args({ 524288 })
->Args({ 8388608 })
->Args({ 16777216 })
->Args({ 33554432 })
->Args({ 67108864 })
->Args({ 134217728 });

static void BM_OpenCLFloat2OPS_RandomData(benchmark::State& state)
{
	// Create Context on Device
	cl::Context context(benchmarkingDevice);

	// Create Program source Object
	cl::Program::Sources sources;

	// Provide Kernel Code
	std::string kernelCode =
		R"CLC(
			void kernel multFloat2(global const float2* in, global float2* out){
				
				float2 x = in[get_global_id(0)];
				float2 y = { 1.01010101f, 1.01010101f };

				y = y * x;

				out[get_global_id(0)] = y;
			}
		)CLC";
	sources.push_back({kernelCode.c_str() , kernelCode.length()});

	// Create Program with Source in the created Context and Build the Program
	cl::Program program(context, sources);
	program.build({benchmarkingDevice});

	// Create Buffer Objects
	cl::Buffer buffer_in(context, CL_MEM_READ_WRITE, sizeof(cl_float2) * 65536);
	cl::Buffer buffer_out(context, CL_MEM_READ_WRITE, sizeof(cl_float2) * 65536);

	// Create Command Queue
	cl::CommandQueue queue(context, benchmarkingDevice);

	// Create Input Data as 2D Vector
	cl_float2 input[65536];
	for (auto i = 0; i < 65536; i++)
	{
		input[i] = {randomFloats[i], randomFloats[i]};
	}

	// Copy Data from Host to Device
	queue.enqueueWriteBuffer(buffer_in, CL_TRUE, 0, sizeof(cl_float2) * 65536, input);

	// Define execution parameters
	size_t maxWorkGroupSize;
	benchmarkingDevice.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);

	cl::NDRange globalSize, localSize;
	globalSize = 65536;
	localSize = maxWorkGroupSize;

	// Create Kernel
	cl::make_kernel<cl::Buffer&, cl::Buffer&> multFloat2(cl::Kernel(program, "multFloat2"));
	cl::EnqueueArgs eargs(queue, globalSize, localSize);

	// Benchmark
	while (state.KeepRunning())
	{
		auto start = std::chrono::high_resolution_clock::now();

		multFloat2(eargs, buffer_in, buffer_out).wait();

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(
				end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}

	float output[65536];
	queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, sizeof(cl_float2) * 65536, output);
	queue.finish();
}

BENCHMARK(BM_OpenCLFloat2OPS_RandomData)
->UseManualTime()
->MinTime(1.0);

static void BM_OpenCLFloat2OPS_GeneratedData(benchmark::State& state)
{
	// Dynamic Data Input
	int dataSize = state.range(0);

	// Create Context on Device
	cl::Context context(benchmarkingDevice);

	// Create Program source Object
	cl::Program::Sources sources;

	// Provide Kernel Code
	std::string kernelCode =
		R"CLC(
			void kernel multFloat2(global const float2* in, global float2* out){
				
				float2 x = in[get_global_id(0)];
				float2 y = { 1.01010101f, 1.01010101f };

				y = y * x;

				out[get_global_id(0)] = y;
			}
		)CLC";
	sources.push_back({ kernelCode.c_str() , kernelCode.length() });

	// Create Program with Source in the created Context and Build the Program
	cl::Program program(context, sources);
	program.build({ benchmarkingDevice });

	// Create Buffer Objects
	cl::Buffer buffer_in(context, CL_MEM_READ_WRITE, sizeof(cl_float2) * dataSize);
	cl::Buffer buffer_out(context, CL_MEM_READ_WRITE, sizeof(cl_float2) * dataSize);

	// Create Command Queue
	cl::CommandQueue queue(context, benchmarkingDevice);

	// Generate Input Data
	std::vector<cl_float2> inputVector(dataSize, { 0, 0 });
	for (int i = 0; i < dataSize; i++)
	{
		inputVector[i] = { static_cast<cl_float>(rand()) * 1.010101f, static_cast<cl_float>(rand()) * 1.010101f };
	}
	cl_float2* input = &inputVector[0];

	// Copy Data from Host to Device
	queue.enqueueWriteBuffer(buffer_in, CL_TRUE, 0, sizeof(cl_float2) * dataSize, input);

	size_t maxWorkGroupSize;
	benchmarkingDevice.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);

	cl::NDRange globalSize, localSize;
	globalSize = dataSize;
	localSize = maxWorkGroupSize;
	//std::cout << "Data Size: " << dataSize << " maxWorkGroupSize: " << maxWorkGroupSize << "\n";

	cl::make_kernel<cl::Buffer&, cl::Buffer&> multFloat2(cl::Kernel(program, "multFloat2"));
	cl::EnqueueArgs eargs(queue, globalSize, localSize);

	// Benchmark
	while (state.KeepRunning())
	{
		auto start = std::chrono::high_resolution_clock::now();

		multFloat2(eargs, buffer_in, buffer_out).wait();

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(
				end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}

	std::vector<cl_float2> outputVector(dataSize, { 0, 0 });
	cl_float2* output = &outputVector[0];
	queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, sizeof(cl_float2) * dataSize, output);
	queue.finish();

	// The returned "Bytes" processed is realy the number of Float2 Processed!
	state.SetBytesProcessed(state.iterations()*dataSize);
}

BENCHMARK(BM_OpenCLFloat2OPS_GeneratedData)
->UseManualTime()
->MinTime(1.0)
->Args({ 2048 })
->Args({ 8192 })
->Args({ 65536 })
->Args({ 524288 })
->Args({ 8388608 })
->Args({ 16777216 })
->Args({ 33554432 })
->Args({ 67108864 })
->Args({ 134217728 });

static void BM_OpenCLFloat16OPS_GeneratedData(benchmark::State& state)
{
	// Dynamic Data Input
	int dataSize = state.range(0);

	// Create Context on Device
	cl::Context context(benchmarkingDevice);

	// Create Program source Object
	cl::Program::Sources sources;

	// Provide Kernel Code
	std::string kernelCode =
		R"CLC(
			void kernel multFloat16(global const float16* in, global float16* out){
				
				float16 x = in[get_global_id(0)];
				float16 y = { 1.01010101f, 1.01010101f, 1.01010101f, 1.01010101f,
							 1.01010101f, 1.01010101f, 1.01010101f, 1.01010101f,
							 1.01010101f, 1.01010101f, 1.01010101f, 1.01010101f,
							 1.01010101f, 1.01010101f, 1.01010101f, 1.01010101f };

				y = y * x;

				out[get_global_id(0)] = y;
			}
		)CLC";
	sources.push_back({ kernelCode.c_str() , kernelCode.length() });

	// Create Program with Source in the created Context and Build the Program
	cl::Program program(context, sources);
	program.build({ benchmarkingDevice });

	// Create Buffer Objects
	cl::Buffer buffer_in(context, CL_MEM_READ_WRITE, sizeof(cl_float16) * dataSize);
	cl::Buffer buffer_out(context, CL_MEM_READ_WRITE, sizeof(cl_float16) * dataSize);

	// Create Command Queue
	cl::CommandQueue queue(context, benchmarkingDevice);

	// Generate Input Data
	std::vector<cl_float16> inputVector(dataSize, { 0, 0 });
	for (int i = 0; i < dataSize; i++)
	{
		inputVector[i] = { 
			static_cast<cl_float>(rand()) * 1.010101f, static_cast<cl_float>(rand()) * 1.010101f,
			static_cast<cl_float>(rand()) * 1.010101f, static_cast<cl_float>(rand()) * 1.010101f,
			static_cast<cl_float>(rand()) * 1.010101f, static_cast<cl_float>(rand()) * 1.010101f,
			static_cast<cl_float>(rand()) * 1.010101f, static_cast<cl_float>(rand()) * 1.010101f, 
			static_cast<cl_float>(rand()) * 1.010101f, static_cast<cl_float>(rand()) * 1.010101f, 
			static_cast<cl_float>(rand()) * 1.010101f, static_cast<cl_float>(rand()) * 1.010101f, 
			static_cast<cl_float>(rand()) * 1.010101f, static_cast<cl_float>(rand()) * 1.010101f, 
			static_cast<cl_float>(rand()) * 1.010101f, static_cast<cl_float>(rand()) * 1.010101f 
		};
	}
	cl_float16* input = &inputVector[0];

	// Copy Data from Host to Device
	queue.enqueueWriteBuffer(buffer_in, CL_TRUE, 0, sizeof(cl_float16) * dataSize, input);

	size_t maxWorkGroupSize;
	benchmarkingDevice.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);

	cl::NDRange globalSize, localSize;
	globalSize = dataSize;
	localSize = maxWorkGroupSize;
	//std::cout << "Data Size: " << dataSize << " maxWorkGroupSize: " << maxWorkGroupSize << "\n";

	cl::make_kernel<cl::Buffer&, cl::Buffer&> multFloat16(cl::Kernel(program, "multFloat16"));
	cl::EnqueueArgs eargs(queue, globalSize, localSize);

	// Benchmark
	while (state.KeepRunning())
	{
		auto start = std::chrono::high_resolution_clock::now();

		multFloat16(eargs, buffer_in, buffer_out).wait();

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(
				end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}

	std::vector<cl_float16> outputVector(dataSize, { 0, 0 });
	cl_float16* output = &outputVector[0];
	queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, sizeof(cl_float16) * dataSize, output);
	queue.finish();

	// The returned "Bytes" processed is realy the number of Float16 Processed!
	state.SetBytesProcessed(state.iterations()*dataSize);
}

BENCHMARK(BM_OpenCLFloat16OPS_GeneratedData)
->UseManualTime()
->MinTime(1.0)
->Args({ 2048 })
->Args({ 8192 })
->Args({ 65536 })
->Args({ 524288 })
->Args({ 8388608 });

static void BM_OpenCLBandwidthHostToDevice(benchmark::State& state)
{
	// Transfer size
	int chunkSize = state.range(0);

	// Create Context on Device
	cl::Context context(benchmarkingDevice);

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
		auto start = std::chrono::high_resolution_clock::now();

		queue.enqueueWriteBuffer(buffer_in, CL_TRUE, 0, sizeof(int) * chunkSize, input);

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(
				end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}

	state.SetBytesProcessed(state.iterations() * chunkSize * sizeof(int));
}

BENCHMARK(BM_OpenCLBandwidthHostToDevice)
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

static void BM_OpenCLBandwidthDeviceToDevice(benchmark::State& state)
{
	// Transfer size
	int chunkSize = state.range(0);

	// Create Context on Device
	cl::Context context(benchmarkingDevice);

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
		auto start = std::chrono::high_resolution_clock::now();

		queue.enqueueCopyBuffer(buffer_in, buffer_device, 0, 0, sizeof(int) * chunkSize);
		queue.finish();

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(
				end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}

	state.SetBytesProcessed(state.iterations() * chunkSize * sizeof(int));
}

BENCHMARK(BM_OpenCLBandwidthDeviceToDevice)
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

static void BM_OpenCLBandwidthDeviceToHost(benchmark::State& state)
{
	// Transfer size
	int chunkSize = state.range(0);

	// Create Context on Device
	cl::Context context(benchmarkingDevice);

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
		auto start = std::chrono::high_resolution_clock::now();

		queue.enqueueReadBuffer(buffer_inout, CL_TRUE, 0, sizeof(int) * chunkSize, inout);

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(
				end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}

	state.SetBytesProcessed(state.iterations() * chunkSize * sizeof(int));
}

BENCHMARK(BM_OpenCLBandwidthDeviceToHost)
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

static void BM_OpenCLKernelCreation(benchmark::State& state)
{
	int kernels = state.range(0);
	int data = state.range(1);
	int dataPerKernel = static_cast<int>(data / kernels);
	//std::cout << "Kernels: " << kernels << ", data: " << data << ", dataPerKernel: " << dataPerKernel << "\n";

	// Create Context on Device
	cl::Context context(benchmarkingDevice);

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
	localSize = maxWorkGroupSize > kernels ? kernels : maxWorkGroupSize;

	// Create Kernel
	cl::make_kernel<int, cl::Buffer&, cl::Buffer&> gridStrideKernel(cl::Kernel(program, "gridStrideKernel"));
	cl::EnqueueArgs eargs(queue, globalSize, localSize);
	//std::cout << "Global Size: " << kernels << ", Local Size: " << (maxWorkGroupSize>kernels ? kernels : maxWorkGroupSize) << "\n";

	while (state.KeepRunning())
	{
		auto start = std::chrono::high_resolution_clock::now();

		gridStrideKernel(eargs, dataPerKernel, input, output).wait();

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(
				end - start);
		state.SetIterationTime(elapsed_seconds.count());
	}

	// Ouptut Data
	std::vector<int> outputVector(data, 0);
	int* outputData = &inputVector[0];

	// Copy Data from Device to Host
	queue.enqueueReadBuffer(output, CL_TRUE, 0, sizeof(int) * data, outputData);

	// Report speed
	state.SetBytesProcessed(state.iterations() * data * sizeof(int));
}

BENCHMARK(BM_OpenCLKernelCreation)
->UseManualTime()
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
