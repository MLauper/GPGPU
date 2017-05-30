#include "opencl_demo.h"
#include <iostream>

int main(int argc, char** argv)
{
	opencl_demo::PlatformDemo* platform = new opencl_demo::PlatformDemo();

	platform->printNumberOfAvailablePlatforms();
	platform->printPlatforms();
	platform->printPlatformInfo();

	platform->printDevices();
	platform->printDeviceInfo();

	opencl_demo::memory::demonstrateMemBuffer();

	// Free resources
	delete platform;

	return 0;
}

opencl_demo::PlatformDemo::PlatformDemo()
{
	this->numOfPlatforms_ = opencl_helpers::getNumOfPlatforms();
	this->platforms_ = opencl_helpers::getPlatforms();
}

void opencl_demo::PlatformDemo::printNumberOfAvailablePlatforms()
{
	std::cout << "Number of available OpenCL Platforms on this System: " << numOfPlatforms_ << "\n\n";
}

void opencl_demo::PlatformDemo::printPlatforms()
{
	for (auto it = this->platforms_.begin(); it != this->platforms_.end(); ++it)
	{
		std::cout << "Platform ID: " << (*it)->getId() << "\n";
	}
	std::cout << "\n";
}

void opencl_demo::PlatformDemo::printPlatformInfo()
{
	for (auto it = this->platforms_.begin(); it != this->platforms_.end(); ++it)
	{
		std::cout << "Platform Info for: " << (*it)->getId() << "\n";
		std::cout << "\t" << "Platform Profile: " << (*it)->getClPlatformProfile() << "\n";
		std::cout << "\t" << "Platform Version: " << (*it)->getClPlatformVersion() << "\n";
		std::cout << "\t" << "Platform Name: " << (*it)->getClPlatformName() << "\n";
		std::cout << "\t" << "Platform Vendor: " << (*it)->getClPlatformVendor() << "\n";
		std::cout << "\t" << "Platform Extensions: " << (*it)->getClPlatformExtensions() << "\n";

		std::cout << "\n";
	}
	std::cout << "\n";
}

void opencl_demo::PlatformDemo::printDevices()
{
	for (auto it = this->platforms_.begin(); it != this->platforms_.end(); ++it)
	{
		std::cout << "Devices for Platform: " << (*it)->getId() << ", " << (*it)->getClPlatformName() << "\n";
		auto devices = (*it)->getDevices();
		for (auto deviceIt = devices.begin(); deviceIt != devices.end(); ++deviceIt)
		{
			std::cout << "\t" << "Device: " << (*deviceIt)->getId() << "\n";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

void opencl_demo::PlatformDemo::printDeviceInfo()
{
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	for (auto& platform : platforms)
	{
		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

		for (auto& device : devices)
		{
			std::cout << "Info for Device: " << device.getInfo<CL_DEVICE_VENDOR_ID>() << "\n";

			std::cout << "\tDevice Available: " << device.getInfo<CL_DEVICE_AVAILABLE>() << "\n";
			std::cout << "\tDevice Name: " << device.getInfo<CL_DEVICE_NAME>() << "\n";
			std::cout << "\tDevice Platform: " << device.getInfo<CL_DEVICE_PLATFORM>() << "\n";
			std::cout << "\tDevice Profile: " << device.getInfo<CL_DEVICE_PROFILE>() << "\n";
			std::cout << "\tDevice Version: " << device.getInfo<CL_DEVICE_VERSION>() << "\n";
			std::cout << "\tDevice Type: " << device.getInfo<CL_DEVICE_TYPE>() << "\n";
			std::cout << "\tDevice Vendor ID: " << device.getInfo<CL_DEVICE_VENDOR_ID>() << "\n";
			std::cout << "\tDevice Driver Version: " << device.getInfo<CL_DRIVER_VERSION>() << "\n";
			std::cout << "\tDevice Global Mem Cache Size: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>() << "\n";
			std::cout << "\tDevice Global Mem Cache Type: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE>() << "\n";
			std::cout << "\tDevice Global Mem Size: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << "\n";
			std::cout << "\tDevice Local Mem Size: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << "\n";
			std::cout << "\tDevice Local Mem Type: " << device.getInfo<CL_DEVICE_LOCAL_MEM_TYPE>() << "\n";
			std::cout << "\tDevice Extentions: " << device.getInfo<CL_DEVICE_EXTENSIONS>() << "\n";

			std::cout << "\n";
		}
	}
}

void opencl_demo::memory::demonstrateMemBuffer()
{
	std::cout << "This is a Kernel exeuction with Buffer Memory" << "\n";

	// Get Platforms
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	
	// Select Platform
	cl::Platform platform = platforms[0];

	// Get Devices
	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

	// Select Device
	cl::Device device = devices[0];
	
	// Create Context on Device
	cl::Context context({ device });

	// Create Program source Object
	cl::Program::Sources sources;

	// Provide Kernel Code
	std::string kernelCode =
		R"CLC(
			void kernel addInt(global const int* A, global const int* B, global int* C){       
				C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];                 
			}
		)CLC";
	sources.push_back({ kernelCode.c_str() , kernelCode.length() });

	// Create Program with Source in the created Context and Build the Program
	cl::Program program(context, sources);
	program.build({ device });
	
	// Create Buffer Objects
	cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
	cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * 10);

	// Input Data
	int A[] = { 1, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	int B[] = { 2, 1, 2, 0, 1, 2, 0, 1, 2, 0 };

	// Create Command Queue
	cl::CommandQueue queue(context, device);

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

	std::cout << "\tCalculated " << A[0] << " + " << B[0] << " = " << C[0] << "\n";

	return;
}
