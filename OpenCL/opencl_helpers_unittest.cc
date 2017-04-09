#include "opencl_helpers.h"
#include "gtest/gtest.h"

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
	cl_uint numOfPlatforms = opencl_helpers::getNumOfPlatforms();

	// Expect at least one available OpenCL Platform in the system
	EXPECT_GT(numOfPlatforms, 0);

	return;
}

TEST(OpenCl_Helpers_GetPlatforms, basic)
{
	// Retrieve platforms
	auto numOfPlatforms = opencl_helpers::getNumOfPlatforms();
	auto platforms = opencl_helpers::getPlatforms();

	// /Expect to match the number of platforms and returned platform objects
	EXPECT_EQ(numOfPlatforms, platforms.size());
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

