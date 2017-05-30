# Setup OpenCL
include(FindOpenCL)
if (NOT OpenCL_FOUND)
	message( FATAL_ERROR "OpenCL not found. Please install OpenCL or disable OpenCL in this project." )
endif ()
if (NOT OpenCL_VERSION_MAJOR EQUAL 2)
	message( "OpenCL is not in version 2." )
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

if(USE_OPENCL_BENCHMARK)
	add_executable(
		opencl_benchmark
		opencl_benchmark.cc
		opencl_benchmark.h
		random_data.cc
	)

	add_dependencies(
		opencl_benchmark
		opencl_helpers
		benchmark
	)

	target_link_libraries(
		opencl_benchmark
		opencl_helpers
		benchmark
		${OpenCL_LIBRARIES}
	)
	
endif()