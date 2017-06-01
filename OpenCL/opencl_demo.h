#ifndef GPGPU_OPENCL_OPENCL_DEMO_
#define GPGPU_OPENCL_OPENCL_DEMO_
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

		/*! \brief Print all available OpenCL Platforms */
		void printPlatforms();
		/*! \brief Print the number of available OpenCL Platforms*/
		void printNumberOfAvailablePlatforms();
		/*! \brief Print platform information*/
		void printPlatformInfo();
		/*! \brief Print all available OpenCL Devices */
		void printDevices();
		/*! \brief Print device information */
		void printDeviceInfo();
	private:
		cl_uint numOfPlatforms_;
		std::vector<opencl_helpers::Platform*> platforms_;
	};

	namespace memory
	{
		/*! \brief Display a demonstration of OpenCL buffered Memory */
		void demonstrateMemBuffer();
	}

}


#endif // GPGPU_OPENCL_OPENCL_DEMO_