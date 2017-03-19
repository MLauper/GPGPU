### Simple Map


This is a simple CUDA example for a reductoin.

Each value in a list will be cubed on the GPU with CUDA.

To compire and runthis project, run the following commands on Windows:
```
mkdir cuda_build
cd cuda_build
cmake ..
cmake --build .
Debug\kernel.exe
```