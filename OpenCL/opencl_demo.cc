#include "opencl_demo.h"
#include "opencl_helpers.h"
#include <iostream>


int main(int argc, char **argv) {
	
	opencl_demo::PlatformDemo* platform = new opencl_demo::PlatformDemo();

	platform->printNumberOfAvailablePlatforms();
	platform->printPlatforms();
	platform->printPlatformInfo();
	
	// Free resources
	delete platform;

	return 0;
}

opencl_demo::PlatformDemo::PlatformDemo()
{
	this->numOfPlatforms_ = opencl_helpers::getNumOfPlatforms();
	this->platforms_ = opencl_helpers::getPlatforms();
}

void opencl_demo::PlatformDemo::printNumberOfAvailablePlatforms()
{
	std::cout << "Number of available OpenCL Platforms on this System: " << numOfPlatforms_ << "\n\n";
}

void opencl_demo::PlatformDemo::printPlatforms()
{
	for(auto it = this->platforms_.begin(); it != this->platforms_.end(); ++it)
	{
		std::cout << "Platform ID: " << (*it)->getId() << "\n";
	}
	std::cout << "\n";
}

void opencl_demo::PlatformDemo::printPlatformInfo()
{
	for (auto it = this->platforms_.begin(); it != this->platforms_.end(); ++it)
	{
		std::cout << "Platform Info for: " << (*it)->getId() << "\n";
		std::cout << "\t" << "Platform Profile: " << (*it)->getClPlatformProfile() << "\n";
		std::cout << "\t" << "Platform Version: " << (*it)->getClPlatformVersion() << "\n";
		std::cout << "\t" << "Platform Name: " << (*it)->getClPlatformName() << "\n";
		std::cout << "\t" << "Platform Vendor: " << (*it)->getClPlatformVendor() << "\n";
		std::cout << "\t" << "Platform Extensions: " << (*it)->getClPlatformExtensions() << "\n";

		std::cout << "\n";
	}
	std::cout << "\n";
}
