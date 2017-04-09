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

## Dynamically configure library location
#configure_file (
#	"${PROJECT_SOURCE_DIR}/<file.in>"
#	"${PROJECT_BINARY_DIR}/<file>"
#)

# Include binary directory to make use of the dynamical configured header files
include_directories("$PROJECT_BINARY_DIR}")

# Include child directories
if (USE_CUDA)
	include_directories ("${PROJECT_SOURCE_DIR}/CUDA")
	add_subdirectory (CUDA)
endif ()

if (USE_OPENCL)
	include_directories ("${PROJECT_SOURCE_DIR}/OpenCL")
	add_subdirectory (OpenCL)
	
	# Try to find OpenCL and verify
	include(FindOpenCL)
	if (NOT OpenCL_FOUND)
		message( FATAL_ERROR "OpenCL not found. Please install OpenCL or disable OpenCL in this project." )
	endif ()
	if (NOT OpenCL_VERSION_MAJOR EQUAL 2)
		message( WARNING "OpenCL is not in version 2. This is not a tested scenario." )
	endif()
	
endif ()

# Add Executables
add_executable(
	GPGPU
	GPGPU.cc
	GPGPU.h
)
target_link_libraries (
	GPGPU
	LINK_PUBLIC cuda_helpers
)

if (USE_GTEST)
	#enable_testing()
    #
	#set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)
    #
	#include(ExternalProject)
	#ExternalProject_Add(
	#	gtest
	#	GIT_REPOSITORY https://github.com/google/googletest.git
	#	GIT_TAG release-1.8.0
	#	CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
	#	PREFIX gtest
	#)
	#
	#include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
	#link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)
	#add_test(AllTestsInGPGPU GPGPU)
	#
	#include_directories ("${PROJECT_BINARY_DIR}/gtest/src/gtest/googletest/include")
    #
	#add_test(
	#	NAME	cuda_helpers_unittest 
	#	COMMAND	cuda_helpers_unittest
	#)
	
	# Download and unpack googletest at configure time
	
	
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
