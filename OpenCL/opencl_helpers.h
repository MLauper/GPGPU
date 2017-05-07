#ifndef GPGPU_OPENCL_OPENCL_HELPERS_
#define GPGPU_OPENCL_OPENCL_HELPERS_
#include <CL/cl.h>
#include <string>
#include <vector>

/*! \file opencl_helpers.h
 *	\brief Contains helper functions for OpenCL.
 *	
 *	This file contains helper functions to reduce the complexity
 *	of the OpenCL API and make the demonstrations cleaner and 
 *	reduce noise. 
 */

/*! \brief Holds all OpenCL Helper functions
 *
 * This namespace contains all OpenCL Helper functions. They are
 * ordered based on their usecase. The basic functions can be called
 * directly. You will receive an appropriate object in return on which
 * you will be able to invoke advanced OpenCL functions.
 */
namespace opencl_helpers
{
	/*! \brief Holds an OpenCL Error
	 *
	 * This struct holds an OpenCL Error Code and a human readable
	 * Error message.
	 */
	struct opencl_error;
	enum class opencl_device_type;
	class Platform;
	class Device;

	class Platform
	{
	public:
		explicit Platform(cl_platform_id platform_id);
		cl_platform_id getId();
		char* getClPlatformProfile();
		char* getClPlatformVersion();
		char* getClPlatformName();
		char* getClPlatformVendor();
		char* getClPlatformExtensions();
		std::vector<opencl_helpers::Device*> getDevices();

	private:
		void gatherPlatformInfo();
		void gatherDevices();
		cl_platform_id platform_id_;
		char* clPlatformProfile_;
		char* clPlatformVersion_;
		char* clPlatformName_;
		char* clPlatformVendor_;
		char* clPlatformExtensions_;
		std::vector<opencl_helpers::Device*> devices_;
	};

	class Device
	{
	public:
		explicit Device(cl_device_id device_id);
		opencl_device_type device_type;
		cl_device_id getId();
	private:
		cl_device_id device_id_;
	};

	enum class opencl_device_type
	{
		CPU = CL_DEVICE_TYPE_CPU,
		GPU = CL_DEVICE_TYPE_GPU,
		ACCELERATOR = CL_DEVICE_TYPE_ACCELERATOR,
		CUSTOM = CL_DEVICE_TYPE_CUSTOM,
		DEFAULT = CL_DEVICE_TYPE_DEFAULT,
		ALL = CL_DEVICE_TYPE_ALL
	};

	// Platform
	std::vector<opencl_helpers::Platform*> getPlatforms();
	cl_uint getNumOfPlatforms();
	char* getPlatformProfile(cl_platform_id platform_id);
	char* getPlatformVersion(cl_platform_id platform_id);
	char* getPlatformName(cl_platform_id platform_id);
	char* getPlatformVendor(cl_platform_id platform_id);
	char* getPlatformExtensions(cl_platform_id platform_id);

	// Devices
	cl_uint getNumOfDevices(cl_platform_id platform_id);
	cl_uint getNumOfDevices(cl_platform_id platform_id, cl_device_type device_type);
	std::vector<opencl_helpers::Device*> getDevices(cl_platform_id platform_id);
	std::vector<opencl_helpers::Device*> getDevices(cl_platform_id platform_id, cl_device_type device_type);
	
	// Test
	int getUniversalAnswer();

	namespace runtime
	{
		namespace platform
		{
			opencl_error* clGetPlatformIDs(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms);
			opencl_error* clGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
		}
		namespace device
		{
			opencl_error* clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id* devices, cl_uint* num_devices);
			opencl_error* clGetDeviceInfo(cl_device_id device, cl_device_info  param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
		}
	}

	struct opencl_error
	{
		explicit opencl_error(cl_int errorCode)
			: errorCode(errorCode)
		{
		}

		explicit opencl_error(): errorCode(0)
		{
		}

		cl_int errorCode;
		std::string errorString;

		void setError(cl_int errorCode);
	};
}


#endif // GPGPU_OPENCL_OPENCL_HELPERS_
