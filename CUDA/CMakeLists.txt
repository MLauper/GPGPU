# Setup OpenCL
include(FindCUDA)
if (NOT CUDA_FOUND)
	message( FATAL_ERROR "CUDA not found. Please install CUDA or disable CUDA in this project." )
endif ()
if (NOT CUDA_VERSION_MAJOR EQUAL 8)
	message( WARNING "CUDA is not in version 8. This is not a tested scenario." )
endif()

include_directories ("${CUDA_INCLUDE_DIRS}")
include_directories ("${PROJECT_SOURCE_DIR}/CUDA")

# cuda_helpers
cuda_add_library(
	cuda_helpers 
	cuda_helpers.cu
	cuda_helpers.h
)
target_link_libraries(
	cuda_helpers
	${CUDA_LIBRARIES}
)

# cuda_demo
cuda_add_executable(
	cuda_demo
	cuda_demo.cu
	cuda_demo.h
)
target_link_libraries (
	cuda_demo
	cuda_helpers
	${CUDA_LIBRARIES}
)

if (USE_GTEST)
	cuda_add_executable(
		cuda_helpers_unittest
		cuda_helpers_unittest.cu
	)

	add_dependencies(
		cuda_helpers_unittest
		cuda_helpers
		gtest
	)

	target_link_libraries(
		cuda_helpers_unittest
		cuda_helpers
		gtest_main
		${CUDA_LIBRARIES}
	)
	
	add_test(NAME cuda_helpers COMMAND cuda_helpers_unittest)
endif()