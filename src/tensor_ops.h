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

void rmsnorm(const float* x, const float* weight, float* output, int d, float epsilon) {
    // Step 1-4: compute the single RMS value for this vector
    float sum_of_squares = 0;
    for (int i = 0; i < d; i++){
        sum_of_squares = sum_of_squares + x[i] * x[i];
    }

    float mean_square = sum_of_squares / d;
    float rms = sqrt(mean_square + epsilon);

    // Step 5-6: rescale each element and apply the learned weight
    for(int i = 0; i < d; i++){
        output[i] = (x[i]/ rms) * weight[i];
    }

}