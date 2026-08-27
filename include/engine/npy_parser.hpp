#pragma once

#include <string>

#include "tensor.hpp"

Tensor load_npy(const std::string& path);

bool check_accuracy(const Tensor& a, const Tensor& b, float max_tol = 1e-4f, float min_tol = 1e-6f);
