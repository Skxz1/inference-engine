#include "gguf_reader.h"
#include <iostream>
#include <cstring>

int main() {
    // Build one Q8_0 block by hand: 2 bytes scale (F16 for 2.0) + 32 int8 values.
    unsigned char block[34];

    uint16_t scale_f16 = 0x4000;  // 2.0 in F16
    std::memcpy(block, &scale_f16, 2);

    int8_t values[32] = {1, 2, 3, 4, 5, 6, 7, 8};  // rest default to 0
    for (int i = 8; i < 32; i++) values[i] = 0;
    std::memcpy(block + 2, values, 32);

    // Fake a MappedFile pointing directly at our hand-built block.
    MappedFile fake_mapped;
    fake_mapped.data = block;
    fake_mapped.size = 34;

    std::vector<float> result = read_tensor_data(fake_mapped, 0, 8, 32);

    std::cout << "Dequantized: ";
    for (int i = 0; i < 10; i++) std::cout << result[i] << " ";
    std::cout << std::endl;

    return 0;
}