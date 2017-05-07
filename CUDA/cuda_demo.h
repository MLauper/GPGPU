#ifndef GPGPU_CUDA_CUDA_DEMO_
#define GPGPU_CUDA_CUDA_DEMO_
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include "cuda_helpers.h"


/*! \file cuda_demo.h
 *	\brief Provides functions to to interactively demonstrate CUDA.
 *	
 *	This header file provides all function for the namespace cuda_demo.
 *	Under cuda_demo, there are several other namespaces that provide
 *	appropriate demos for different CUDA functionality, such as memory
 *	management, multi-device support, etc.
 */

/*! \brief Contains all CUDA Demos
 *
 * This is the root namespace for all CUDA demos. It contains all other 
 * demos as child namespaces.
 */
namespace cuda_demo
{
	/*! \brief CUDA device memory demonstrations.
	 *
	 * This namespace contains demonstrations to show the capabilities of CUDA
	 * device memory. This includes copy from and to device memory and other 
	 * memory management techniques.
	 */
	namespace device_memory
	{
		/*! \brief Runs all CUDA Device Memory demonstrations.
		 *
		 * This functions runs all CUDA Device Memory demonstrations, this means
		 * printing appropriate header message and invoke all functions defined in
		 * subsidiary namespaces.
		 */
		void demonstrateDeviceMemory();

		/*! \brief CUDA Linear Memory demonstrations
		 * 
		 * This namespace contains all Demos for CUDA Linear Memory, which means
		 * Memory on the GPU itself.
		 */
		namespace linear_memory
		{
			/*! \brief Demonstrates the usage of CUDA Global Memory.
			 *
			 * This function demonstrates the usage of the CUDA Global Memory by 
			 * copy data from the host memory to the device memory, execute a 
			 * kernel and copy the result back to the host.
			 */
			void demonstrateLinearDeviceMemory();

			/*! \brief Demonstrate the usage of CUDA Shared Memory
			 *
			 * Thsi function demonstrates the usage of CUDA Shared Memory, which is 
			 * located on the Streaming Mulitprocessor on the GPU. It copies data
			 * from the host memory to the global memory of the gpu. Afterwards
			 * this data is copied to the shared memory of the device itself.
			 * The result of an arbitrary computation is written to shared memory
			 * and copied back to global memory and back to host memory.
			 */
			void demonstrateSharedDeviceMemory();
		}
	}

}


#endif // GPGPU_CUDA_CUDA_DEMO_