#include <gtest/gtest.h>

#include "engine/engine.hpp"

TEST(Smoke, GTestIsWired) {
    EXPECT_EQ(1 + 1, 2);
}

TEST(Smoke, EngineLinksAgainstJson) {
    EXPECT_EQ(engine::version(), "ml_inference_engine 0.0.1");
}
