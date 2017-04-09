#ifndef GPGPU_OPENCL_OPENCL_HELPERS_
#define GPGPU_OPENCL_OPENCL_HELPERS_
#include <CL/cl.h>
#include <string>

int answerToEverything();

namespace opencl_helpers
{
	class Loader
	{
	public:
		static int giveItToMe();
	};

	struct opencl_error
	{
		cl_int errorCode;
		std::string errorString;

		void setError(cl_int errorCode);
	};

	void getPlatforms();

	namespace runtime
	{
		namespace plattform
		{
			opencl_error* clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms);
		}
	}
}


#endif // GPGPU_OPENCL_OPENCL_HELPERS_