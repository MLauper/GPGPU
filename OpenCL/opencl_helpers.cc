#include "opencl_helpers.h"
#include "CL/cl.h"
#include "opencl_demo.h"

void opencl_helpers::opencl_error::setError(cl_int errorCode)
{
	if(errorCode == NULL){ errorCode = CL_SUCCESS; }

	this->errorCode = errorCode;
	switch (errorCode)
	{
	case CL_SUCCESS:
		this->errorString = "CL_SUCCESS";
		break;
	case CL_DEVICE_NOT_FOUND:
		this->errorString = "CL_DEVICE_NOT_FOUND";
		break;
	case CL_DEVICE_NOT_AVAILABLE:
		this->errorString = "CL_DEVICE_NOT_AVAILABLE";
		break;
	case CL_COMPILER_NOT_AVAILABLE:
		this->errorString = "CL_COMPILER_NOT_AVAILABLE";
		break;
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		this->errorString = "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		break;
	case CL_OUT_OF_RESOURCES:
		this->errorString = "CL_OUT_OF_RESOURCES";
		break;
	case CL_OUT_OF_HOST_MEMORY:
		this->errorString = "CL_OUT_OF_HOST_MEMORY";
		break;
	case CL_PROFILING_INFO_NOT_AVAILABLE:
		this->errorString = "CL_PROFILING_INFO_NOT_AVAILABLE";
		break;
	case CL_MEM_COPY_OVERLAP:
		this->errorString = "CL_MEM_COPY_OVERLAP";
		break;
	case CL_IMAGE_FORMAT_MISMATCH:
		this->errorString = "CL_IMAGE_FORMAT_MISMATCH";
		break;
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:
		this->errorString = "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		break;
	case CL_BUILD_PROGRAM_FAILURE:
		this->errorString = "CL_BUILD_PROGRAM_FAILURE";
		break;
	case CL_MAP_FAILURE:
		this->errorString = "CL_MAP_FAILURE";
		break;
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:
		this->errorString = "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		break;
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
		this->errorString = "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		break;
	case CL_COMPILE_PROGRAM_FAILURE:
		this->errorString = "CL_COMPILE_PROGRAM_FAILURE";
		break;
	case CL_LINKER_NOT_AVAILABLE:
		this->errorString = "CL_LINKER_NOT_AVAILABLE";
		break;
	case CL_LINK_PROGRAM_FAILURE:
		this->errorString = "CL_LINK_PROGRAM_FAILURE";
		break;
	case CL_DEVICE_PARTITION_FAILED:
		this->errorString = "CL_DEVICE_PARTITION_FAILED";
		break;
	case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
		this->errorString = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
		break;
	case CL_INVALID_VALUE:
		this->errorString = "CL_INVALID_VALUE";
		break;
	case CL_INVALID_DEVICE_TYPE:
		this->errorString = "CL_INVALID_DEVICE_TYPE";
		break;
	case CL_INVALID_PLATFORM:
		this->errorString = "CL_INVALID_PLATFORM";
		break;
	case CL_INVALID_DEVICE:
		this->errorString = "CL_INVALID_DEVICE";
		break;
	case CL_INVALID_CONTEXT:
		this->errorString = "CL_INVALID_CONTEXT";
		break;
	case CL_INVALID_QUEUE_PROPERTIES:
		this->errorString = "CL_INVALID_QUEUE_PROPERTIES";
		break;
	case CL_INVALID_COMMAND_QUEUE:
		this->errorString = "CL_INVALID_COMMAND_QUEUE";
		break;
	case CL_INVALID_HOST_PTR:
		this->errorString = "CL_INVALID_HOST_PTR";
		break;
	case CL_INVALID_MEM_OBJECT:
		this->errorString = "CL_INVALID_MEM_OBJECT";
		break;
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
		this->errorString = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		break;
	case CL_INVALID_IMAGE_SIZE:
		this->errorString = "CL_INVALID_IMAGE_SIZE";
		break;
	case CL_INVALID_SAMPLER:
		this->errorString = "CL_INVALID_SAMPLER";
		break;
	case CL_INVALID_BINARY:
		this->errorString = "CL_INVALID_BINARY";
		break;
	case CL_INVALID_BUILD_OPTIONS:
		this->errorString = "CL_INVALID_BUILD_OPTIONS";
		break;
	case CL_INVALID_PROGRAM:
		this->errorString = "CL_INVALID_PROGRAM";
		break;
	case CL_INVALID_PROGRAM_EXECUTABLE:
		this->errorString = "CL_INVALID_PROGRAM_EXECUTABLE";
		break;
	case CL_INVALID_KERNEL_NAME:
		this->errorString = "CL_INVALID_KERNEL_NAME";
		break;
	case CL_INVALID_KERNEL_DEFINITION:
		this->errorString = "CL_INVALID_KERNEL_DEFINITION";
		break;
	case CL_INVALID_KERNEL:
		this->errorString = "CL_INVALID_KERNEL";
		break;
	case CL_INVALID_ARG_INDEX:
		this->errorString = "CL_INVALID_ARG_INDEX";
		break;
	case CL_INVALID_ARG_VALUE:
		this->errorString = "CL_INVALID_ARG_VALUE";
		break;
	case CL_INVALID_ARG_SIZE:
		this->errorString = "CL_INVALID_ARG_SIZE";
		break;
	case CL_INVALID_KERNEL_ARGS:
		this->errorString = "CL_INVALID_KERNEL_ARGS";
		break;
	case CL_INVALID_WORK_DIMENSION:
		this->errorString = "CL_INVALID_WORK_DIMENSION";
		break;
	case CL_INVALID_WORK_GROUP_SIZE:
		this->errorString = "CL_INVALID_WORK_GROUP_SIZE";
		break;
	case CL_INVALID_WORK_ITEM_SIZE:
		this->errorString = "CL_INVALID_WORK_ITEM_SIZE";
		break;
	case CL_INVALID_GLOBAL_OFFSET:
		this->errorString = "CL_INVALID_GLOBAL_OFFSET";
		break;
	case CL_INVALID_EVENT_WAIT_LIST:
		this->errorString = "CL_INVALID_EVENT_WAIT_LIST";
		break;
	case CL_INVALID_EVENT:
		this->errorString = "CL_INVALID_EVENT";
		break;
	case CL_INVALID_OPERATION:
		this->errorString = "CL_INVALID_OPERATION";
		break;
	case CL_INVALID_GL_OBJECT:
		this->errorString = "CL_INVALID_GL_OBJECT";
		break;
	case CL_INVALID_BUFFER_SIZE:
		this->errorString = "CL_INVALID_BUFFER_SIZE";
		break;
	case CL_INVALID_MIP_LEVEL:
		this->errorString = "CL_INVALID_MIP_LEVEL";
		break;
	case CL_INVALID_GLOBAL_WORK_SIZE:
		this->errorString = "CL_INVALID_GLOBAL_WORK_SIZE";
		break;
	case CL_INVALID_PROPERTY:
		this->errorString = "CL_INVALID_PROPERTY";
		break;
	case CL_INVALID_IMAGE_DESCRIPTOR:
		this->errorString = "CL_INVALID_IMAGE_DESCRIPTOR";
		break;
	case CL_INVALID_COMPILER_OPTIONS:
		this->errorString = "CL_INVALID_COMPILER_OPTIONS";
		break;
	case CL_INVALID_LINKER_OPTIONS:
		this->errorString = "CL_INVALID_LINKER_OPTIONS";
		break;
	case CL_INVALID_DEVICE_PARTITION_COUNT:
		this->errorString = "CL_INVALID_DEVICE_PARTITION_COUNT";
		break;
	// The following two error codes are not available in OpenCL 1.2. 
	//case CL_INVALID_PIPE_SIZE:
	//	this->errorString = "CL_INVALID_PIPE_SIZE";
	//	break;
	//case CL_INVALID_DEVICE_QUEUE:
	//	this->errorString = "CL_INVALID_DEVICE_QUEUE";
	default:
		this->errorString = "UNKNOWN ERROR";
	}
}

opencl_helpers::Platform::Platform(cl_platform_id platform_id): platform_id_(platform_id)
{
	this->gatherPlatformInfo();
	this->gatherDevices();
}

cl_platform_id opencl_helpers::Platform::getId()
{
	return platform_id_;
}

char* opencl_helpers::Platform::getClPlatformProfile()
{
	return clPlatformProfile_;
}

char* opencl_helpers::Platform::getClPlatformVersion()
{
	return clPlatformVersion_;
}

char* opencl_helpers::Platform::getClPlatformName()
{
	return clPlatformName_;
}

char* opencl_helpers::Platform::getClPlatformVendor()
{
	return clPlatformVendor_;
}

char* opencl_helpers::Platform::getClPlatformExtensions()
{
	return clPlatformExtensions_;
}

std::vector<opencl_helpers::Device*> opencl_helpers::Platform::getDevices()
{
	return this->devices_;
}

void opencl_helpers::Platform::gatherPlatformInfo()
{
	this->clPlatformProfile_ = opencl_helpers::getPlatformProfile(this->platform_id_);
	this->clPlatformVersion_ = opencl_helpers::getPlatformVersion(this->platform_id_);
	this->clPlatformName_ = opencl_helpers::getPlatformName(this->platform_id_);
	this->clPlatformVendor_ = opencl_helpers::getPlatformVendor(this->platform_id_);
	this->clPlatformExtensions_ = opencl_helpers::getPlatformExtensions(this->platform_id_);
}

void opencl_helpers::Platform::gatherDevices()
{
	this->devices_ = opencl_helpers::getDevices(this->platform_id_);
}

opencl_helpers::Device::Device(cl_device_id device_id): device_id_(device_id)
{
}

cl_device_id opencl_helpers::Device::getId()
{
	return this->device_id_;
}

/**
 * \brief 
 * \return Returns a Vector of Pointers to Platforms
 */
std::vector<opencl_helpers::Platform*> opencl_helpers::getPlatforms()
{
	// Retrieve number of platforms to allocate space
	cl_uint num_platforms = opencl_helpers::getNumOfPlatforms();
	
	// Retrieve platform IDs
	opencl_helpers::opencl_error* err;
	std::vector<cl_platform_id> platform_ids(num_platforms);
	err = opencl_helpers::runtime::platform::clGetPlatformIDs(num_platforms, &platform_ids[0], NULL);

	// Create Platform objects
	std::vector<opencl_helpers::Platform*> platforms;
	for (auto it = platform_ids.begin(); it != platform_ids.end(); ++it)
	{
		platforms.push_back(new opencl_helpers::Platform(*it));
	}

	// Return Platforms
	return platforms;
}

std::vector<opencl_helpers::Device*> opencl_helpers::getDevices(cl_platform_id platform_id, cl_device_type device_type)
{
	// Retrieve Number of Devices of Platform
	cl_uint num_devices;
	opencl_helpers::opencl_error* err;
	err = opencl_helpers::runtime::device::clGetDeviceIDs(platform_id, device_type, 0, NULL, &num_devices);

	// Retrieve Device IDs
	std::vector<cl_device_id> devices_ids(num_devices);
	err = opencl_helpers::runtime::device::clGetDeviceIDs(platform_id, device_type, num_devices, &devices_ids[0], NULL);

	// Create Device objects
	std::vector<opencl_helpers::Device*> devices;
	for (auto it = devices_ids.begin(); it != devices_ids.end(); ++it)
	{
		devices.push_back(new opencl_helpers::Device(*it));
	}

	return devices;
}

std::vector<opencl_helpers::Device*> opencl_helpers::getDevices(cl_platform_id platform_id)
{
	// Retrieve Device Type CL_DEVICE_TYPE_ALL without explicityl specifying any device type
	return opencl_helpers::getDevices(platform_id, CL_DEVICE_TYPE_ALL);
}

cl_uint opencl_helpers::getNumOfPlatforms()
{
	cl_uint num_platforms;
	opencl_helpers::opencl_error* err;
	err = opencl_helpers::runtime::platform::clGetPlatformIDs(0, NULL, &num_platforms);

	return num_platforms;
}

cl_uint opencl_helpers::getNumOfDevices(cl_platform_id platform_id, cl_device_type device_type)
{
	cl_uint num_devices;
	opencl_helpers::opencl_error* err;
	err = opencl_helpers::runtime::device::clGetDeviceIDs(platform_id, device_type, 0, NULL, &num_devices);
	
	return num_devices;
}

cl_uint opencl_helpers::getNumOfDevices(cl_platform_id platform_id)
{
	cl_uint num_devices;
	num_devices = opencl_helpers::getNumOfDevices(platform_id, CL_DEVICE_TYPE_ALL);

	return num_devices;
}

char* opencl_helpers::getPlatformProfile(cl_platform_id platform_id)
{
	opencl_helpers::opencl_error* err;
	size_t value_size_ret;
	
	// Get size of returned string
	err = opencl_helpers::runtime::platform::clGetPlatformInfo(platform_id, CL_PLATFORM_PROFILE, 0, NULL, &value_size_ret);

	// Retrieve value
	char* platform_profile = new char[value_size_ret];
	err = opencl_helpers::runtime::platform::clGetPlatformInfo(platform_id, CL_PLATFORM_PROFILE, value_size_ret, &platform_profile[0], NULL);

	return platform_profile;
}

char* opencl_helpers::getPlatformVersion(cl_platform_id platform_id)
{
	opencl_helpers::opencl_error* err;
	size_t value_size_ret;

	// Get size of returned string
	err = opencl_helpers::runtime::platform::clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, 0, NULL, &value_size_ret);

	// Retrieve value
	char* platform_version = new char[value_size_ret];
	err = opencl_helpers::runtime::platform::clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, value_size_ret, &platform_version[0], NULL);

	return platform_version;
}

char* opencl_helpers::getPlatformName(cl_platform_id platform_id)
{
	opencl_helpers::opencl_error* err;
	size_t value_size_ret;

	// Get size of returned string
	err = opencl_helpers::runtime::platform::clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 0, NULL, &value_size_ret);

	// Retrieve value
	char* platform_name = new char[value_size_ret];
	err = opencl_helpers::runtime::platform::clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, value_size_ret, &platform_name[0], NULL);

	return platform_name;
}

char* opencl_helpers::getPlatformVendor(cl_platform_id platform_id)
{
	opencl_helpers::opencl_error* err;
	size_t value_size_ret;

	// Get size of returned string
	err = opencl_helpers::runtime::platform::clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, 0, NULL, &value_size_ret);

	// Retrieve value
	char* platform_vendor = new char[value_size_ret];
	err = opencl_helpers::runtime::platform::clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, value_size_ret, &platform_vendor[0], NULL);

	return platform_vendor;
}

char* opencl_helpers::getPlatformExtensions(cl_platform_id platform_id)
{
	opencl_helpers::opencl_error* err;
	size_t value_size_ret;

	// Get size of returned string
	err = opencl_helpers::runtime::platform::clGetPlatformInfo(platform_id, CL_PLATFORM_EXTENSIONS, 0, NULL, &value_size_ret);

	// Retrieve value
	char* platform_extensions = new char[value_size_ret];
	err = opencl_helpers::runtime::platform::clGetPlatformInfo(platform_id, CL_PLATFORM_EXTENSIONS, value_size_ret, &platform_extensions[0], NULL);

	return platform_extensions;
}

int opencl_helpers::getUniversalAnswer()
{
	return 42;
}

opencl_helpers::opencl_error* opencl_helpers::runtime::platform::clGetPlatformIDs(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms)
{
	auto err = ::clGetPlatformIDs(num_entries, platforms, num_platforms);

	return new opencl_helpers::opencl_error(err);
}

opencl_helpers::opencl_error* opencl_helpers::runtime::platform::clGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret)
{
	auto err = ::clGetPlatformInfo(platform, param_name, param_value_size, param_value, param_value_size_ret);

	return new opencl_helpers::opencl_error(err);
}

opencl_helpers::opencl_error* opencl_helpers::runtime::device::clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id* devices, cl_uint* num_devices)
{
	auto err = ::clGetDeviceIDs(platform, device_type, num_entries, devices, num_devices);

	return new opencl_helpers::opencl_error(err);
}

opencl_helpers::opencl_error* opencl_helpers::runtime::device::clGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret)
{
	auto err = ::clGetDeviceInfo(device, param_name, param_value_size, param_value, param_value_size_ret);

	return new opencl_helpers::opencl_error(err);
}