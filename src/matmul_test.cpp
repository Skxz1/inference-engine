#include "tensor_ops.h"
#include <iostream>

int main() {
    // A is 2x3
    float A[6] = {
        1, 2, 3,
        4, 5, 6
    };

    // B is 3x4
    float B[12] = {
        7,  8,  9,  10,
        11, 12, 13, 14,
        15, 16, 17, 18
    };

    // C will be 2x4
    float C[8];

    matmul(A, B, C, 2, 3, 4);

    std::cout << "Result:" << std::endl;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << C[i * 4 + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}