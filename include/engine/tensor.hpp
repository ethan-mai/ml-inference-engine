#pragma once

#include <vector>

struct Tensor {
    std::vector<float> data;
    std::vector<int> shape; 
};

