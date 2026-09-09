#include "gguf_reader.h"

int main(int argc, char* argv[]){

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path-to-gguf>" << std::endl;
        return 1;
    }

    MappedFile result = map_file(argv[1]);

    inspect_header(result);
    size_t metadata_end = inspect_metadata(result, 23);
    size_t tensor_index_end = inspect_tensors(result, 201, metadata_end);

    size_t alignment = 32;
    size_t tensor_data_start = ((tensor_index_end + alignment - 1) / alignment) * alignment;

    uint64_t output_norm_offset = 1169063936;
    std::vector<float> output_norm_weights = read_tensor_data(result, tensor_data_start + output_norm_offset, 0, 2048);

    std::cout << "First 5 values of output_norm.weight: ";
    for (int i = 0; i < 5; i++) {
        std::cout << output_norm_weights[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Tensor data starts at: " << tensor_data_start << std::endl;

    float test = f16_to_f32(0x3C00);
    std::cout << "F16 0x3C00 should be 1.0, got: " << test << std::endl;

    float test2 = f16_to_f32(0x4000);
    std::cout << "F16 0x4000 should be 2.0, got: " << test2 << std::endl;

    uint64_t attn_k_offset = 1159036928;
    uint64_t attn_k_elements = 2048 * 256;
    std::vector<float> attn_k_weights = read_tensor_data(result, tensor_data_start + attn_k_offset, 8, attn_k_elements);

    std::cout << "First 5 values of blk.9.attn_k.weight: ";
    for (int i = 0; i < 5; i++) {
        std::cout << attn_k_weights[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}