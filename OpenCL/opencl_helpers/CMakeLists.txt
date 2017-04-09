
add_library(
	opencl_helpers 
	opencl_helpers.cc
	opencl_helpers.h
)
target_link_libraries(
	opencl_helpers
	${OpenCL_LIBRARIES}
)
include_directories (
	${OpenCL_INCLUDE_DIR}
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
	)
	
	add_test(NAME opencl_helpers COMMAND opencl_helpers_unittest)
endif()


#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()