$desc = Read-Host "Provide system descriptor"
iex "..\..\GPGPU_Build\CUDA\Release\cuda_benchmark.exe --benchmark_out=${desc}_cuda.csv --benchmark_format=csv"
iex "..\..\GPGPU_Build\OpenCL\Release\opencl_benchmark.exe --benchmark_out=${desc}_opencl.csv --benchmark_format=csv"