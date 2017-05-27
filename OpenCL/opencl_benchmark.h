#ifndef GPGPU_OPENCL_OPENCL_BENCHMARK_
#define GPGPU_OPENCL_OPENCL_BENCHMARK_
#include <CL/cl.h>
#include <string>
#include "opencl_helpers.h"
#include "benchmark/benchmark.h"

/*! \file opencl_benchmark.h
*	\brief Provides benchmarks to measure OpenCL performance.
*
*	This header file provides all function for the namespace opencl_benchmark.
*	Under opencl_benchmark, there are several functions that are used to
*	measure the performance of OpenCL.
*/

/*! \brief Contains benchmarks for OpenCL
 *
 * This namepsace contains all demonstrations for OpenCL demonstratoins.
 * The subsidiary namespaces are organized based on the purpose of the 
 * demonstration
 */
namespace opencl_benchmark
{
	void bench();

}


#endif // GPGPU_OPENCL_OPENCL_BENCHMARK_