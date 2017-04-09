# Setup OpenCL
include(FindOpenCL)
if (NOT OpenCL_FOUND)
	message( FATAL_ERROR "OpenCL not found. Please install OpenCL or disable OpenCL in this project." )
endif ()
if (NOT OpenCL_VERSION_MAJOR EQUAL 2)
	message( WARNING "OpenCL is not in version 2. This is not a tested scenario." )
endif()

include_directories ("${OpenCL_INCLUDE_DIRS}")
include_directories ("${PROJECT_SOURCE_DIR}/OpenCL")

# opencl_helpers
add_library(
	opencl_helpers 
	opencl_helpers.cc
	opencl_helpers.h
)
target_link_libraries(
	opencl_helpers
	${OpenCL_LIBRARIES}
)

# opnecl_demo
add_executable(
	opencl_demo
	opencl_demo.cc
	opencl_demo.h
)
target_link_libraries (
	opencl_demo
	opencl_helpers
	${OpenCL_LIBRARIES}
)

if (USE_GTEST)
	add_executable(
		opencl_helpers_unittest
		opencl_helpers_unittest.cc
	)

	add_dependencies(
		opencl_helpers_unittest
		opencl_helpers
		gtest
	)

	target_link_libraries(
		opencl_helpers_unittest
		opencl_helpers
		gtest_main
		${OpenCL_LIBRARIES}
	)
	
	add_test(NAME opencl_helpers COMMAND opencl_helpers_unittest)
endif()