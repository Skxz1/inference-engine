// assert_close.h
//
// Checks whether two lists of numbers are close enough to count as the
// same, instead of exactly equal, because floating point maths from two
// different implementations rarely produces identical results even when
// both are correct.

#pragma once

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

void assert_close(const std::vector<float>& actual, const std::vector<float>& expected, float atol, float rtol) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("Size mismatch");
    }

    for (size_t i = 0; i < actual.size(); i++) {
        float diff = std::abs(actual[i] - expected[i]);
        float allowed = atol + rtol * std::abs(expected[i]);
        if (diff > allowed) {
            throw std::runtime_error("Mismatch at index " + std::to_string(i));
        }
    }
}