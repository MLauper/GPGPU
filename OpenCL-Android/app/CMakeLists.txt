cmake_minimum_required(VERSION 3.4.1)

# Setup OpenCL
include_directories ( include )
link_directories ( imported-lib/${ANDROID_ABI}/ )
find_library( OpenCL imported-lib/${ANDROID_ABI}/ )

# Setup OpenCL Helpers Library
include_directories ( ${CMAKE_CURRENT_SOURCE_DIR}/../../OpenCL )
add_library(
	opencl_helpers
	${CMAKE_CURRENT_SOURCE_DIR}/../../OpenCL/opencl_helpers.cc
	${CMAKE_CURRENT_SOURCE_DIR}/../../OpenCL/opencl_helpers.h
)
target_link_libraries(
    opencl_helpers
    OpenCL
)

# Setup Log Library
add_library( native-lib SHARED src/main/cpp/native-lib.cpp )
find_library( log-lib log )

# Link Wrapper Library to OpenCL and OpenCL Helpers
target_link_libraries( native-lib OpenCL opencl_helpers ${log-lib} )