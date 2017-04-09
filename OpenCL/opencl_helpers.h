#ifndef GPGPU_OPENCL_OPENCL_HELPERS_
#define GPGPU_OPENCL_OPENCL_HELPERS_
#include <CL/cl.h>
#include <string>
#include <vector>

namespace opencl_helpers
{
	struct opencl_error;

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

	private:
		void gatherPlatformInfo();
		cl_platform_id platform_id_;
		char* clPlatformProfile_;
		char* clPlatformVersion_;
		char* clPlatformName_;
		char* clPlatformVendor_;
		char* clPlatformExtensions_;
	};

	std::vector<opencl_helpers::Platform*> getPlatforms();
	cl_uint getNumOfPlatforms();
	char* getPlatformProfile(cl_platform_id platform_id);
	char* getPlatformVersion(cl_platform_id platform_id);
	char* getPlatformName(cl_platform_id platform_id);
	char* getPlatformVendor(cl_platform_id platform_id);
	char* getPlatformExtensions(cl_platform_id platform_id);
	int getUniversalAnswer();

	namespace runtime
	{
		namespace platform
		{
			opencl_error* clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms);
			opencl_error* clGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
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