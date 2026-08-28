#pragma once

#include <string>
#include <vector>

struct Node {
    std::string op;
    std::vector<std::string> inputs;
    std::string output;
};