# Setup OpenGL
cmake_minimum_required(VERSION 3.8)



set(CMAKE_PREFIX_PATH ./external/glew)

#GLEW libraries
include(FindGLEW)
if (NOT GLEW_FOUND)
	message( FATAL_ERROR "GLEW not found." )
endif ()

include_directories ( ${GLEW_INCLUDE_DIRS}  )

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

add_executable(
	opengl_demo
	opengl_demo.cc
	# opengl_demo.hpp
	${GLSL}
)


target_link_libraries(
	opengl_demo
	${GLEW_LIBRARIES}
)
