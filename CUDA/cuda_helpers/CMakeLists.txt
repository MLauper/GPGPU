
add_library(
	cuda_helpers 
	cuda_helpers.cc
	cuda_helpers.h
)

if (USE_GTEST)
	add_executable(
		cuda_helpers_unittest
		cuda_helpers_unittest.cc
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
	)
	
	add_test(NAME cuda_helpers COMMAND cuda_helpers_unittest)
endif()

target_include_directories (cuda_helpers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})