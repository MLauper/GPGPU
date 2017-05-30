#include "cuda_benchmark.h"
#include <iostream>
#include <ratio>
#include <chrono>

int main(int argc, char** argv)
{
	::benchmark::Initialize(&argc, argv);
	::benchmark::RunSpecifiedBenchmarks();
}
