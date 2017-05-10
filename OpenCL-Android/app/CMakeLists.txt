cmake_minimum_required(VERSION 3.4.1)

# Setup OpenCL
include_directories ( include )
link_directories ( imported-lib/${ANDROID_ABI}/ )
find_library( OpenCL imported-lib/${ANDROID_ABI}/ )

# Setup Log Library
add_library( native-lib SHARED src/main/cpp/native-lib.cpp )
find_library( log-lib log )

# Set Link Libraries
target_link_libraries( native-lib OpenCL ${log-lib} )