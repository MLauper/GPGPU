#include "cuda_helpers.h"
#include "gtest/gtest.h"

TEST(answerToEverythingTest, Right)
{
	EXPECT_EQ(42, cuda_helpers::answerToEverything());
}

TEST(executeKernel, dummy)
{
	dummyKernel <<< 1, 1 >>>();
	EXPECT_EQ(cudaSuccess, cudaGetLastError());
}

TEST(MemoryCopy, linear)
{
	int h_a = 1;
	int h_b = 1;
	int h_c = 0;

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

int main(int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
