#ifndef GPGPU_CUDA_CUDA_DEMO_
#define GPGPU_CUDA_CUDA_DEMO_
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include "cuda_helpers.h"

namespace cuda_demo
{
	namespace device_memory
	{
		void demonstrateDeviceMemory();
		namespace linear_memory
		{
			void demonstrateLinearDeviceMemory();
			void demonstrateSharedDeviceMemory();
		}
	}

}


#endif // GPGPU_CUDA_CUDA_DEMO_