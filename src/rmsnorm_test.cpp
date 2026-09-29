#include "tensor_ops.h"
#include <iostream>

int main() {
    float x[2] = {3, 4};
    float weight[2] = {1, 1};
    float output[2];

    rmsnorm(x, weight, output, 2, 0.00001f);

    std::cout << "Output: " << output[0] << " " << output[1] << std::endl;
    return 0;
}