#include "opencl_helpers.h"
#include "gtest/gtest.h"

TEST(answerToEverythingTest, Right) {
	EXPECT_EQ(42, answerToEverything());
}

TEST(nameSpaces, DoIt)
{
	opencl_helpers::Loader* myLoader = new opencl_helpers::Loader();
	auto i = opencl_helpers::Loader::giveItToMe();
	auto j = myLoader->giveItToMe();
	EXPECT_EQ(41, i);
	EXPECT_EQ(41, j);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

