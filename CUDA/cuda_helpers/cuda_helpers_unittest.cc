#include "cuda_helpers.h"
#include "gtest/gtest.h"

TEST(answerToEverythingTest, Right) {
	EXPECT_EQ(42, cuda_helpers::answerToEverything());
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}