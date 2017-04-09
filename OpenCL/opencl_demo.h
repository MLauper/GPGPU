#ifndef GPGPU_OPENCL_OPENCL_DEMO_
#define GPGPU_OPENCL_OPENCL_DEMO_
#include <CL/cl.h>
#include <string>
#include "opencl_helpers.h"

namespace opencl_demo
{
	class PlatformDemo
	{
	public:
		PlatformDemo();
		void printPlatforms();
		void printNumberOfAvailablePlatforms();
		void printPlatformInfo();
	private:
		cl_uint numOfPlatforms_;
		std::vector<opencl_helpers::Platform*> platforms_;
	};

}


#endif // GPGPU_OPENCL_OPENCL_DEMO_