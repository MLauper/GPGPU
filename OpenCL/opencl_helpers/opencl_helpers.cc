#include "opencl_helpers.h"
#include "CL/cl.h"

int answerToEverything(){
	return 42;
}

int opencl_helpers::Loader::giveItToMe()
{
	return 41;
}

void opencl_helpers::opencl_error::setError(cl_int errorCode)
{
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
	case CL_INVALID_PIPE_SIZE:
		this->errorString = "CL_INVALID_PIPE_SIZE";
		break;
	case CL_INVALID_DEVICE_QUEUE:
		this->errorString = "CL_INVALID_DEVICE_QUEUE";
	default:
		this->errorString = "UNKNOWN ERROR";
	}
}


opencl_helpers::opencl_error* opencl_helpers::runtime::plattform::clGetPlatformIDs(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms)
{
	cl_int err = ::clGetPlatformIDs(num_entries, platforms, num_platforms);
	opencl_error* errObj = NULL;

	if (err != NULL)
	{
		errObj = new opencl_error;
		errObj->setError(err);
	}

	return errObj;
}
