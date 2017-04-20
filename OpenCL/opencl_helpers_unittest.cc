#include "opencl_helpers.h"
#include "gtest/gtest.h"
#include "opencl_demo.h"

TEST(OpenCl_Platform_clGetPlatformIDs, num_platforms)
{
	cl_uint num_platforms;
	opencl_helpers::opencl_error* err;
	err = opencl_helpers::runtime::platform::clGetPlatformIDs(0, NULL, &num_platforms);
	
	// Expect at least one available OpenCL Platform in the system
	EXPECT_EQ(err->errorCode, CL_SUCCESS);
	EXPECT_GT(num_platforms, 0);

	// Free memory
	delete err;

	return;
}

TEST(OpenCl_Platform_clGetPlatformIDs, platforms)
{
	// Retrieve number of platforms to allocate space
	cl_uint num_platforms;
	opencl_helpers::opencl_error* err;
	err = opencl_helpers::runtime::platform::clGetPlatformIDs(0, NULL, &num_platforms);

	// Retrieve platform IDs
	std::vector<cl_platform_id> platforms(num_platforms);
	err = opencl_helpers::runtime::platform::clGetPlatformIDs(num_platforms, &platforms[0], NULL);

	// Expect no error
	EXPECT_EQ(err->errorCode, CL_SUCCESS);

	// Free memory
	delete err;

	return;
}

TEST(OpenCl_Device_clGetDeviceIDs, devices)
{
	// Retrieve Platforms
	auto platforms = opencl_helpers::getPlatforms();
	EXPECT_GT(platforms.size(), 0);
	auto platform = platforms[0];

	// Retrieve Number of Devices of Platform
	cl_uint num_devices;
	opencl_helpers::opencl_error* err;
	err = opencl_helpers::runtime::device::clGetDeviceIDs(platform->getId(), CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	EXPECT_GT(num_devices, 0);

	// Retrieve Device IDs
	std::vector<cl_device_id> devices(num_devices);
	err = opencl_helpers::runtime::device::clGetDeviceIDs(platform->getId(), CL_DEVICE_TYPE_ACCELERATOR, num_devices, &devices[0], NULL);
	EXPECT_EQ(devices.size(), static_cast<size_t>(num_devices));
}

TEST(OpenCl_Platform_clGetPlatformInfo, queryName)
{
	// Get Testing Platform
	auto platforms = opencl_helpers::getPlatforms();
	EXPECT_GT(platforms.size(), 0);
	auto platform = platforms[0];

	// Query possible values
	opencl_helpers::opencl_error* err;
	size_t ret_size = 0;
	err = opencl_helpers::runtime::platform::clGetPlatformInfo(platform->getId(), CL_PLATFORM_NAME, 0, NULL, &ret_size);
	EXPECT_EQ(err->errorCode, CL_SUCCESS);
}

class OpenCl_Device_clGetDeviceInfoTest : public ::testing::TestWithParam<cl_device_info>
{};
TEST_P(OpenCl_Device_clGetDeviceInfoTest, queryOpenCL12)
{
	// Get Testing Platform
	auto platforms = opencl_helpers::getPlatforms();
	EXPECT_GT(platforms.size(), static_cast<size_t>(0));
	auto platform = platforms[0];

	// Get Testing Device
	auto devices = opencl_helpers::getDevices(platform->getId());
	EXPECT_GT(devices.size(), static_cast<size_t>(0));
	auto device = devices[0];

	// Query Device Infos
	size_t param_size;
	opencl_helpers::opencl_error* err;
	err = opencl_helpers::runtime::device::clGetDeviceInfo(device->getId(), GetParam(), 0, NULL, &param_size);
	auto param_value = malloc(param_size);
	err = opencl_helpers::runtime::device::clGetDeviceInfo(device->getId(), GetParam(), param_size, param_value, NULL);
	EXPECT_GT(sizeof(param_value), 0);
	free(param_value);
}
const cl_device_info deviceInfosCl12[] = { CL_DEVICE_ADDRESS_BITS, CL_DEVICE_AVAILABLE, CL_DEVICE_BUILT_IN_KERNELS, CL_DEVICE_COMPILER_AVAILABLE,
	CL_DEVICE_DOUBLE_FP_CONFIG, CL_DEVICE_ENDIAN_LITTLE, CL_DEVICE_ERROR_CORRECTION_SUPPORT, CL_DEVICE_EXECUTION_CAPABILITIES,
	CL_DEVICE_EXTENSIONS, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
	CL_DEVICE_GLOBAL_MEM_SIZE, CL_DEVICE_HOST_UNIFIED_MEMORY, CL_DEVICE_IMAGE_SUPPORT, CL_DEVICE_IMAGE2D_MAX_HEIGHT,
	CL_DEVICE_IMAGE2D_MAX_WIDTH, CL_DEVICE_IMAGE3D_MAX_DEPTH, CL_DEVICE_IMAGE3D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_WIDTH,
	CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, CL_DEVICE_LINKER_AVAILABLE, CL_DEVICE_LOCAL_MEM_SIZE,
	CL_DEVICE_LOCAL_MEM_TYPE, CL_DEVICE_MAX_CLOCK_FREQUENCY, CL_DEVICE_MAX_COMPUTE_UNITS, CL_DEVICE_MAX_CONSTANT_ARGS,
	CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, CL_DEVICE_MAX_MEM_ALLOC_SIZE, CL_DEVICE_MAX_PARAMETER_SIZE, CL_DEVICE_MAX_READ_IMAGE_ARGS,
	CL_DEVICE_MAX_SAMPLERS, CL_DEVICE_MAX_WORK_GROUP_SIZE, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, CL_DEVICE_MAX_WORK_ITEM_SIZES,
	CL_DEVICE_MAX_WRITE_IMAGE_ARGS, CL_DEVICE_MEM_BASE_ADDR_ALIGN, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, CL_DEVICE_NAME,
	CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG,
	CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, CL_DEVICE_OPENCL_C_VERSION,
	CL_DEVICE_PARENT_DEVICE, CL_DEVICE_PARTITION_MAX_SUB_DEVICES, CL_DEVICE_PARTITION_PROPERTIES, CL_DEVICE_PARTITION_AFFINITY_DOMAIN,
	CL_DEVICE_PARTITION_TYPE, CL_DEVICE_PLATFORM, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
	CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
	CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, CL_DEVICE_PRINTF_BUFFER_SIZE,
	CL_DEVICE_PREFERRED_INTEROP_USER_SYNC, CL_DEVICE_PROFILE, CL_DEVICE_PROFILING_TIMER_RESOLUTION, CL_DEVICE_QUEUE_PROPERTIES,
	CL_DEVICE_REFERENCE_COUNT, CL_DEVICE_SINGLE_FP_CONFIG, CL_DEVICE_TYPE, CL_DEVICE_VENDOR, CL_DEVICE_VENDOR_ID, CL_DEVICE_VERSION,
	CL_DRIVER_VERSION
	// ONLY OpenCl 1.2: CL_DEVICE_HALF_FP_CONFIG
};
INSTANTIATE_TEST_CASE_P(OpenCl_Device_clGetDeviceInfo, OpenCl_Device_clGetDeviceInfoTest, ::testing::ValuesIn(deviceInfosCl12));

TEST(OpenCl_Platform_clGetPlatformInfo, queryPlatformVersion)
{
	// Get Testing Platform
	auto platforms = opencl_helpers::getPlatforms();
	EXPECT_GT(platforms.size(), 0);
	auto platform = platforms[0];

	// Query possible values
	opencl_helpers::opencl_error* err;
	size_t ret_size = 0;
	err = opencl_helpers::runtime::platform::clGetPlatformInfo(platform->getId(), CL_PLATFORM_VERSION, 0, NULL, &ret_size);
	EXPECT_EQ(err->errorCode, CL_SUCCESS);
}

TEST(OpenCl_Platform_clGetPlatformInfo, queryPlatformProfile)
{
	// Get Testing Platform
	auto platforms = opencl_helpers::getPlatforms();
	EXPECT_GT(platforms.size(), 0);
	auto platform = platforms[0];

	// Query possible values
	opencl_helpers::opencl_error* err;
	size_t ret_size = 0;
	err = opencl_helpers::runtime::platform::clGetPlatformInfo(platform->getId(), CL_PLATFORM_PROFILE, 0, NULL, &ret_size);
	EXPECT_EQ(err->errorCode, CL_SUCCESS);
}

TEST(OpenCl_Platform_clGetPlatformInfo, queryVendor)
{
	// Get Testing Platform
	auto platforms = opencl_helpers::getPlatforms();
	EXPECT_GT(platforms.size(), 0);
	auto platform = platforms[0];

	// Query possible values
	opencl_helpers::opencl_error* err;
	size_t ret_size = 0;
	err = opencl_helpers::runtime::platform::clGetPlatformInfo(platform->getId(), CL_PLATFORM_VENDOR, 0, NULL, &ret_size);
	EXPECT_EQ(err->errorCode, CL_SUCCESS);
}

TEST(OpenCl_Platform_clGetPlatformInfo, queryPlatformExtensions)
{
	// Get Testing Platform
	auto platforms = opencl_helpers::getPlatforms();
	EXPECT_GT(platforms.size(), 0);
	auto platform = platforms[0];

	// Query possible values
	opencl_helpers::opencl_error* err;
	size_t ret_size = 0;
	err = opencl_helpers::runtime::platform::clGetPlatformInfo(platform->getId(), CL_PLATFORM_EXTENSIONS, 0, NULL, &ret_size);
	EXPECT_EQ(err->errorCode, CL_SUCCESS);
}

// The Capability CL_PLATFORM_HOST_TIMER_RESOLUTION is an OpenCL 2 feature
//TEST(OpenCl_Platform_clGetPlatformInfo, queryHostTimerResolution)
//{
//	// Get Testing Platform
//	auto platforms = opencl_helpers::getPlatforms();
//	EXPECT_GT(platforms.size(), 0);
//	auto platform = platforms[0];
//
//	// Query possible values
//	opencl_helpers::opencl_error* err;
//	size_t ret_size = 0;
//	err = opencl_helpers::runtime::platform::clGetPlatformInfo(platform->getId(), CL_PLATFORM_HOST_TIMER_RESOLUTION, 0, NULL, &ret_size);
//	EXPECT_EQ(err->errorCode, CL_SUCCESS);
//}

// The capability CL_PLATFORM_ICD_SUFFIX_KHR is an OpenCL 2 feature
//TEST(OpenCl_Platform_clGetPlatformInfo, queryName)
//{
//	// Get Testing Platform
//	auto platforms = opencl_helpers::getPlatforms();
//	EXPECT_GT(platforms.size(), 0);
//	auto platform = platforms[0];
//
//	// Query possible values
//	opencl_helpers::opencl_error* err;
//	size_t ret_size = 0;
//	err = opencl_helpers::runtime::platform::clGetPlatformInfo(platform->getId(), CL_PLATFORM_ICD_SUFFIX_KHR, 0, NULL, &ret_size);
//	EXPECT_EQ(err->errorCode, CL_SUCCESS);
//}

TEST(OpenCl_Helpers_GetNumOfPlatforms, basic)
{
	// Retrieve number of platforms via helper function
	auto numOfPlatforms = opencl_helpers::getNumOfPlatforms();

	// Expect at least one available OpenCL Platform in the system
	EXPECT_GT(numOfPlatforms, 0);

	return;
}

TEST(OpenCl_Helpers_GetNumOfDevices, withDeviceType)
{
	// Retrieve platforms
	auto platforms = opencl_helpers::getPlatforms();
	auto platform = platforms[0];

	// Retrieve Devices
	auto numOfDevices = opencl_helpers::getNumOfDevices(platform->getId(), CL_DEVICE_TYPE_ALL);
	EXPECT_GT(numOfDevices, 0);
}

TEST(OpenCl_Helpers_GetNumOfDevices, withoutDeviceType)
{
	// Retrieve platforms
	auto platforms = opencl_helpers::getPlatforms();
	auto platform = platforms[0];

	// Retrieve Devices
	auto numOfDevices = opencl_helpers::getNumOfDevices(platform->getId());
	EXPECT_GT(numOfDevices, 0);
}

TEST(OpenCl_Helpers_GetPlatforms, basic)
{
	// Retrieve platforms
	auto numOfPlatforms = opencl_helpers::getNumOfPlatforms();
	auto platforms = opencl_helpers::getPlatforms();

	// /Expect to match the number of platforms and returned platform objects
	EXPECT_EQ(numOfPlatforms, platforms.size());
}

TEST(OpenCl_Helpers_GetDeivces, withDeviceType)
{
	// Retrieve platform for testing
	auto platforms = opencl_helpers::getPlatforms();
	auto platform = platforms[0];

	// Retrieve Devices
	auto devices = opencl_helpers::getDevices(platform->getId(), CL_DEVICE_TYPE_ALL);

	EXPECT_GT(devices.size(), static_cast<size_t>(0));
}

TEST(OpenCl_Helpers_GetDeivces, withoutDeviceType)
{
	// Retrieve platform for testing
	auto platforms = opencl_helpers::getPlatforms();
	auto platform = platforms[0];

	// Retrieve Devices
	auto devices = opencl_helpers::getDevices(platform->getId());

	EXPECT_GT(devices.size(), static_cast<size_t>(0));
}

TEST(OpenCl_Helpers_GetPlatformProfile, basic)
{
	auto platforms = opencl_helpers::getPlatforms();
	auto platform = platforms[0];

	char* platformProfile = opencl_helpers::getPlatformProfile(platform->getId());

	// Expect a Platform Profile to be returned
	EXPECT_GT(strlen(platformProfile), 0);
}

TEST(OpenCl_Helpers_GetPlatformVersion, basic)
{
	auto platforms = opencl_helpers::getPlatforms();
	auto platform = platforms[0];

	char* platformVersion = opencl_helpers::getPlatformVersion(platform->getId());

	// Expect a Platform Version to be returned
	EXPECT_GT(strlen(platformVersion), 0);
}

TEST(OpenCl_Helpers_GetPlatformName, basic)
{
	auto platforms = opencl_helpers::getPlatforms();
	auto platform = platforms[0];

	char* platformName = opencl_helpers::getPlatformName(platform->getId());

	// Expect a Platform Version to be returned
	EXPECT_GT(strlen(platformName), 0);
}

TEST(OpenCl_Helpers_GetPlatformVendor, basic)
{
	auto platforms = opencl_helpers::getPlatforms();
	auto platform = platforms[0];

	char* platformVendor = opencl_helpers::getPlatformVendor(platform->getId());

	// Expect a Platform Version to be returned
	EXPECT_GT(strlen(platformVendor), 0);
}

TEST(OpenCl_Helpers_GetPlatformExtensions, basic)
{
	auto platforms = opencl_helpers::getPlatforms();
	auto platform = platforms[0];

	char* platformExtensions = opencl_helpers::getPlatformExtensions(platform->getId());

	// Expect a Platform Version to be returned
	EXPECT_GT(strlen(platformExtensions), 0);
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

