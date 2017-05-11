#ifdef _MSC_VER
#pragma warning( disable: 4312 ) // ignore visual studio warnings for FLTK 64-bit type casts
#endif 

#include <iostream>
#include <string>
#include <GL/glew.h>

int main(int argc, char *argv[])
///////////////////////////////////////////////////////////////////////////////////////////////////
{
	// Print Version
	std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << "\n";

	return 0;
}
