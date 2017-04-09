# Setup project
cmake_minimum_required (VERSION 3.8)
project (GPGPU)

# Versioning (using sematic versioning)
set (GLOBAL_VERSION_MAJOR 0)
set (GLOBAL_VERSION_MINOR 1)
set (GLOBAL_VERSION_PATCH 0)

# Options
option (USE_CUDA "Include CUDA code" ON) 
option (USE_OPENCL "Include CUDA code" ON) 
option (USE_GTEST "Run Google Test Framework Tests" ON)

# Include child directories
#if (USE_CUDA)
#	include_directories ("${PROJECT_SOURCE_DIR}/CUDA")
#	include_directories ("${PROJECT_SOURCE_DIR}/CUDA/cuda_helpers")
#	
#	# Cuda helpers
#	add_subdirectory (CUDA/cuda_helpers)
#	add_library(
#		cuda_helpers 
#		cuda_helpers.cc
#		cuda_helpers.h
#	)
#	
#	if (USE_GTEST)
#		add_executable(
#			cuda_helpers_unittest
#			cuda_helpers_unittest.cc
#		)
#		
#		add_dependencies(
#			cuda_helpers_unittest
#			cuda_helpers
#			gtest
#		)
#		
#		target_link_libraries(
#			cuda_helpers_unittest
#			cuda_helpers
#			gtest_main
#		)
#		
#		add_test(NAME cuda_helpers COMMAND cuda_helpers_unittest)
#	endif()
#
#	target_include_directories (cuda_helpers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
#
#endif ()

if (USE_OPENCL)
	add_subdirectory(OpenCL)
endif ()


if (USE_GTEST)
	configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
	execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
	  RESULT_VARIABLE result
	  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
	if(result)
	  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
	endif()
	execute_process(COMMAND ${CMAKE_COMMAND} --build .
	  RESULT_VARIABLE result
	  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
	if(result)
	  message(FATAL_ERROR "Build step for googletest failed: ${result}")
	endif()

	# Prevent overriding the parent project's compiler/linker
	# settings on Windows
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

	# Add googletest directly to our build. This defines
	# the gtest and gtest_main targets.
	add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
					 ${CMAKE_BINARY_DIR}/googletest-build)

	# The gtest/gtest_main targets carry header search path
	# dependencies automatically when using CMake 2.8.11 or
	# later. Otherwise we have to add them here ourselves.
	if (CMAKE_VERSION VERSION_LESS 2.8.11)
	  include_directories("${gtest_SOURCE_DIR}/include")
	endif()

	## Now simply link against gtest or gtest_main as needed. Eg
	#add_executable(example example.cpp)
	#target_link_libraries(example gtest_main)
	#add_test(NAME example_test COMMAND example)
endif()

## Print all CMake Properties
#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()
