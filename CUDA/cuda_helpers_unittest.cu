#include "cuda_helpers.h"
#include "gtest/gtest.h"

#define EXPECT_CUDA_SUCCES (val) (EXPECT_EQ(cudaSuccess, val))

/*! \file cuda_helpers_unittest.cu
 *	\brief Test the proper functionality of CUDA and CUDA Helpers.
 *	
 *	This file contains tests based on the Google Test framework to 
 *	test if CUDA works as expected and if the CUDA Helper functions
 *	properly work.
 */

/*! \brief Test if a CUDA kernel can be scheduled
 *
 *	fefwefew
 */
TEST(executeKernel, dummy)
{
	dummyKernel <<< 1, 1 >>>();
	EXPECT_EQ(cudaSuccess, cudaGetLastError());
}

/*! \brief Test if a single value can be copied to and from the GPU*/
TEST(LinearMemory, singleValueCopy)
{
	auto h_a = 1;
	auto h_b = 1;
	auto h_c = 0;

	int *d_a, *d_b, *d_c;

	EXPECT_EQ(cudaSuccess, cudaMalloc(&d_a, sizeof(int)));
	EXPECT_EQ(cudaSuccess, cudaMalloc(&d_b, sizeof(int)));
	EXPECT_EQ(cudaSuccess, cudaMalloc(&d_c, sizeof(int)));

	EXPECT_EQ(cudaSuccess, cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice));
	EXPECT_EQ(cudaSuccess, cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice));

	addInt << <1, 1 >> >(d_a, d_b, d_c);
	EXPECT_EQ(cudaSuccess, cudaGetLastError());

	EXPECT_EQ(cudaSuccess, cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost));
	EXPECT_EQ(2, h_c);

	EXPECT_EQ(cudaSuccess, cudaFree(d_a));
	EXPECT_EQ(cudaSuccess, cudaFree(d_b));
	EXPECT_EQ(cudaSuccess, cudaFree(d_c));
}

/*! \brief Test if shared memory can be used.*/
TEST(LinearMemory, sharedMemory)
{
	auto h_a = 1;
	auto h_b = 1;
	auto h_c = 0;

	int *d_a, *d_b, *d_c;

	EXPECT_EQ(cudaSuccess, cudaMalloc(&d_a, sizeof(int)));
	EXPECT_EQ(cudaSuccess, cudaMalloc(&d_b, sizeof(int)));
	EXPECT_EQ(cudaSuccess, cudaMalloc(&d_c, sizeof(int)));

	EXPECT_EQ(cudaSuccess, cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice));
	EXPECT_EQ(cudaSuccess, cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice));

	addIntSharedMemory << <1, 1 >> >(d_a, d_b, d_c);
	EXPECT_EQ(cudaSuccess, cudaGetLastError());

	EXPECT_EQ(cudaSuccess, cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost));
	EXPECT_EQ(2, h_c);

	EXPECT_EQ(cudaSuccess, cudaFree(d_a));
	EXPECT_EQ(cudaSuccess, cudaFree(d_b));
	EXPECT_EQ(cudaSuccess, cudaFree(d_c));
}

/*! \brief All tests are run when the Test Executable is run.*/
int main(int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
