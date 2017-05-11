# Setup OpenGL
cmake_minimum_required(VERSION 3.8)

include_directories ( include  )

# find GLSL shaders files (let them show up in the IDEs)
file(GLOB GLSL
    ./glsl/*.frag
    ./glsl/*.vert
    ./glsl/*.geom
    ./glsl/*.tess
    ./glsl/*.tecs
)
source_group("glsl" FILES ${GLSL})

# find packages and libs
find_package(OpenGL REQUIRED)
find_library(GLEW libs/)

add_executable(
	opengl_demo
	opengl_demo.cc
	# opengl_demo.hpp
	${GLSL}
)
target_link_libraries(
	opengl_demo
	glew
)
