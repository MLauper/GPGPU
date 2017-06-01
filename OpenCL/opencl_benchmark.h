#ifndef GPGPU_OPENCL_OPENCL_BENCHMARK_
#define GPGPU_OPENCL_OPENCL_BENCHMARK_
#include <CL/cl.h>
#include <string>
#include "opencl_helpers.h"
#include "benchmark/benchmark.h"

/*! \file opencl_benchmark.h
*	\brief Provides benchmarks to measure OpenCL performance.
*
* This header file cotains all functions to benchmark OpenCL
* performance.
*/

/*! \brief Array of random float numbers*/
extern float randomFloats[];
/*! \brief Array of random integer numbers*/
extern int randomIntegers[];

/*! \brief Executes full initialization of the device and executes a Kernel
 *
 * This Benchmark initializes an OpenCL Context and executes a 
 * minimalistic Kernel, including minimal data transfer between 
 * the host and the device.
 * The context is released after every execution.
 */
static void BM_OpenCLBasicLatencyTest(benchmark::State& state);

/*! \brief Executes a converged kernel
 *
 * This Benchmark executes an OpenCL kernel which always executes
 * converged and has no branch divergences in it.
 */
static void BM_OpenCLConvergedExecution(benchmark::State& state);

/*! \brief Executes a diverged kernel 
 *
 * This Benchmark executes an OpenCL kernel which always executes
 * diverged, that means multiple branches are taken at execution 
 * time.
 * 
 * NOTE: This benchmark may result in the same result as the converged
 * kernel. It seems that this is due to multiple instruction units
 * on modern GPUs.
 */
static void BM_OpenCLDivergedExecution(benchmark::State& state);

/*! \brief Calculate FLOPS based on predefined random floats
 *
 * This benchmark executes an OpenCL kernel which calculates
 * the number of floating point operations per second (FLOPS)
 * based on a predefined array of 64k float numbers.
 * 
 * NOTE: This benchmark doesn't result in meaningful numbers
 * because 64k are not enought data to lead to a meaningful result.
 */
static void BM_OpenCLFLOPS_RandomData(benchmark::State& state);

/*! \brief Calculate FLOPS based on generated random data
 *
 * This benchmark executes an OpenCL kernel which calculates
 * the number of floating point operations per second (FLOPS)
 * based on generated random float numbers.
 */
static void BM_OpenCLFLOPS_GeneratedData(benchmark::State& state);

/*! \brief Calculates IntOPS on predefined integers
 * 
 * This benchmark executes an OpenCL kernel which calculates
 * the number of integer operations per second (flops)
 * based on a predefined array of 64k int numbers.
 * 
 * NOTE: This benchmark doesn't result in meaningful numbers
 * because 64k are not enought data to lead to a meaningful result.
 */
static void BM_OpenCLIntOPS_RandomData(benchmark::State& state);

/*! \brief  Calculate IntOPS based on generated random data
 *
 * This benchmark executes an OpenCL kernel which calculates
 * the number of integer operations per second (IntOPS)
 * based on generated random integer numbers.
 */
static void BM_OpenCLIntOPS_GeneratedData(benchmark::State& state);

/*! \brief Calculate Float2OPS based on random data
 *
 * This benchmark executes an OpenCL kernel which calculates
 * the number of float2 operations per second (Float2OPS)
 * based on a predefined array of 64k float numbers.
 * 
 * NOTE: This benchmark doesn't result in meaningful numbers
 * because 64k are not enought data to lead to a meaningful result.
 */
static void BM_OpenCLFloat2OPS_RandomData(benchmark::State& state);

/*! \brief Calculate Float2OPS based on generated data
 *
 * This benchmark executes an OpenCL kernel which calculates
 * the number of float2 operations per second (Float2OPS)
 * based on generated random float2 vectors.
 */
static void BM_OpenCLFloat2OPS_GeneratedData(benchmark::State& state);

/*! \brief Calculate FLoat16OPS based on generated data
 *
 * This benchmark executes an OpenCL kernel which calculates
 * the number of float16 (custom OpenCL data type, which represents
 * a vector of floats with size 16) operations per second 
 * (Float16Ops) based on generated random float16 vectors.
 */
static void BM_OpenCLFloat16OPS_GeneratedData(benchmark::State& state);

/*! \brief Measure bandwith from host to device
 *
 * This benchmark measures the bandwidth from the host to 
 * the device. The measurement is based on a full data copy
 * from host to the device and is measured using different 
 * chunk sizes.
 */
static void BM_OpenCLBandwidthHostToDevice(benchmark::State& state);

/*! \brief Measure bandwidth for memory copy operation on the device
 *
 * This benchmark measures the throughput to copy data from 
 * device memory to device memory. 
 */
static void BM_OpenCLBandwidthDeviceToDevice(benchmark::State& state);

/*! \brief Measure bandwidth from device to host
 *
 * This benchmark measures the bandwidth from the device to 
 * the host. THe measurement is based on a full data copy 
 * from device to the host and is measured using different 
 * chunk sizes.
 */
static void BM_OpenCLBandwidthDeviceToHost(benchmark::State& state);

/*! \brief Execute kernels and data
 *
 * This benchmark executes a given number of kernels to manipulate
 * a given batch of data. The goal of this benchmark is to 
 * demonstrate the overhead of kernels, i.e. is it more efficient
 * to start a fixed set of kernels for any input size or should
 * you execute one kernel per input element.
 */
static void BM_OpenCLKernelCreation(benchmark::State& state);

/*! \brief Measure the effect of bad memory coalescence
 *
 * This benchmark measures the performance of kernels that access
 * device memory in a non continuous manor. The device will
 * have to load much more memory than the kernel acutally needs.
 */
static void BM_OpenCLBadMemoryCoalescence(benchmark::State& state);

/*! \brief  Measure the effect of good memory coalescence
 *
 * This benchmark measures the performance of kernels that access
 * device memory in a continuous manor. Therefore the device has 
 * to load only memory, that the kernel acutally uses.
 */
static void BM_OpenCLGoodMemoryCoalescence(benchmark::State& state);

/*! \brief  Interactively select a benchmark device
 *
 * This function lists all available OpenCL devices and let
 * you choose the device for benchmarking. You can automate
 * the device selection by defining autoSelectDevice.
 */
void selectBenchmarkDevice();

/*! \brief The device on which all benchmarks are executed*/
cl::Device benchmarkingDevice;

/*! \brief Predefined benchmarking device
 *
 * This variable defines the id of the device to be used for 
 * benchmarking.
 * 
 * Note: This id has nothing to do with the OpenCL device id.
 * You will have to select a number 0 < i < n, n is the number
 * of available OpenCL devices, to automatically select a device.
 * Set -1 to select a device interactively at runtime.
 */
int autoSelectDevice = 0;

#endif // GPGPU_OPENCL_OPENCL_BENCHMARK_