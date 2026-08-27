#include <gtest/gtest.h>

#include "engine/npy_parser.hpp"

TEST(NpyParser, LoadsCorrectShape) {
    Tensor input = load_npy("golden/golden_input.npy");
    EXPECT_EQ(input.shape, (std::vector<int>{1, 1, 28, 28}));
}

TEST(NpyParser, IdenticalTensorIsCloseToItself) {
    Tensor a = load_npy("golden/stack.0.npy");
    Tensor b = load_npy("golden/stack.0.npy");
    EXPECT_TRUE(check_accuracy(a, b));
}

TEST(NpyParser, DifferentValuesAreNotClose) {
    Tensor linear_output = load_npy("golden/stack.0.npy");
    Tensor relu_output = load_npy("golden/stack.1.npy");
    EXPECT_FALSE(check_accuracy(linear_output, relu_output));
}

TEST(NpyParser, MismatchedShapesAreNotClose) {
    Tensor input = load_npy("golden/golden_input.npy");
    Tensor activation = load_npy("golden/stack.0.npy");
    EXPECT_FALSE(check_accuracy(input, activation));
}
