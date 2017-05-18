# Setup project
cmake_minimum_required (VERSION 3.8)
project (GPGPU)

# Versioning (using sematic versioning)
set (GLOBAL_VERSION_MAJOR 0)
set (GLOBAL_VERSION_MINOR 1)
set (GLOBAL_VERSION_PATCH 0)

# Options
option (USE_CUDA "Include CUDA code" ON) 
option (USE_OPENCL "Include OpenCL code" ON) 
option (USE_OPENGL "Include OpenGL code" OFF)
option (USE_GTEST "Run Google Test Framework Tests" ON)

if (USE_CUDA)
	add_subdirectory(CUDA)
endif ()

if (USE_OPENCL)
	add_subdirectory(OpenCL)
endif ()

if (USE_OPENGL)
	add_subdirectory(OpenGL)
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

# Do not build, because of a bug in the 2.0.0 release of glew on Windows
#if (USE_OPENGL)
#	# Build GLEW Code
#	file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/glew-build)
#	execute_process(COMMAND ${CMAKE_COMMAND} ${CMAKE_SOURCE_DIR}/OpenGL/external/glew/build/cmake
#	  RESULT_VARIABLE result
#	  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/glew-build )
#	if(result)
#	  message(FATAL_ERROR "Build step for glew failed: ${result}")
#	endif()
#	execute_process(COMMAND ${CMAKE_COMMAND} --build .
#	  RESULT_VARIABLE result
#	  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/glew-build )
#	if(result)
#	  message(FATAL_ERROR "Build step for glew failed: ${result}")
#	endif()
#	
#	add_subdirectory(${CMAKE_BINARY_DIR}/glew-build)
#endif()
## Print all CMake Properties
#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()
