// tesnor_ops.h

// The cor numerical building blocks of the transformer forward pass:
// matrix multiply, RMSNorm, softmax, SwiGLU, attention.

#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>

void matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;

            for (int k = 0; k < K; k++) {
                // multiply-and-accumulate goes here
                sum = sum + A[i*K + k] * B[k*N + j];
            }

            // store sum into C goes here
            C[i*N + j] = sum;

        }
    }
}