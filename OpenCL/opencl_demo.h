#ifndef GPGPU_OPENCL_OPENCL_DEMO_
#define GPGPU_OPENCL_OPENCL_DEMO_
#include <CL/cl.h>
#include <string>
#include "opencl_helpers.h"

/*! \file opencl_demo.h
*	\brief Provides functions to to interactively demonstrate OpenCL.
*
*	This header file provides all function for the namespace opencl_demo.
*	Under opencl_demo, there are several other namespaces that provide
*	appropriate demos for different OpenCL functionality, such as retrieval
*	of platforms and devices.
*/

/*! \brief Contains interactive demonstrations for OpenCL
 *
 * This namepsace contains all demonstrations for OpenCL demonstratoins.
 * The subsidiary namespaces are organized based on the purpose of the 
 * demonstration
 */
namespace opencl_demo
{
	class PlatformDemo
	{
	public:
		PlatformDemo();
		void printPlatforms();
		void printNumberOfAvailablePlatforms();
		void printPlatformInfo();
		void printDevices();
	private:
		cl_uint numOfPlatforms_;
		std::vector<opencl_helpers::Platform*> platforms_;
	};

	class DeviceDemo
	{
	public:
		
	};

}


#endif // GPGPU_OPENCL_OPENCL_DEMO_