#include "engine/npy_parser.hpp"

#include <cmath>

#include <npy.hpp>

Tensor load_npy(const std::string& path) {
    npy::npy_data<float> npy_data = npy::read_npy<float>(path);

    Tensor tensor;

    tensor.data = std::move(npy_data.data);
    tensor.shape.assign(npy_data.shape.begin(), npy_data.shape.end());

    return tensor;
}

bool check_accuracy(const Tensor& a, const Tensor& b, float max_tol, float min_tol) {
    if (a.shape != b.shape) {
        return false;
    }

    for (size_t i = 0; i < a.data.size(); ++i) {
        float diff = std::fabs(a.data[i] - b.data[i]);
        if (diff > min_tol + max_tol * std::fabs(b.data[i])) {
            return false;
        }
    }

    return true;
}
