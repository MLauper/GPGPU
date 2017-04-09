include_directories (
	${OpenCL_INCLUDE_DIR}
)

include_directories ("${PROJECT_SOURCE_DIR}/OpenCL/opencl_helpers")
add_subdirectory (opencl_helpers)
