#include "gguf_reader.h"
#include "tensor_ops.h"
#include "../tests/npy_loader.h"
#include "../tests/assert_close.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path-to-gguf>" << std::endl;
        return 1;
    }

    MappedFile mapped = map_file(argv[1]);
    MetadataResult metadata_result = inspect_metadata(mapped, 23);
    size_t tensor_index_end = inspect_tensors(mapped, 201, metadata_result.end_pos);
    size_t alignment = 32;
    size_t tensor_data_start = ((tensor_index_end + alignment - 1) / alignment) * alignment;

    uint64_t token_embd_offset = 69632000;
    uint64_t attn_norm_offset = 139264000;

    // Token 450 ("The"), a normal word, not the special BOS token.
    std::vector<float> token_450_embedding = read_tensor_data(
        mapped, tensor_data_start + token_embd_offset, 8, 2048, 450, 2048);

    // Layer 0's attn_norm weight, F32, no row-skip needed.
    std::vector<float> attn_norm_weight = read_tensor_data(
        mapped, tensor_data_start + attn_norm_offset, 0, 2048);
        

    // Run our own rmsnorm
    std::vector<float> our_output(2048);
    rmsnorm(token_450_embedding.data(), attn_norm_weight.data(), our_output.data(), 2048, 0.00001f);

    // Reference: layer_0_rmsnorm_output.npy is shape (1, 6, 2048), one
    // slice per sequence position. Token 450 sits at position 1 in our
    // prompt (token IDs were [1, 450, 7483, 310, 3444, 338]), so we want
    // the SECOND 2048-float slice, not the first.
    NpyArray reference = load_npy("reference_tensors/layer_0_rmsnorm_output.npy");
    std::vector<float> expected(reference.data.begin() + 2048, reference.data.begin() + 4096);

    std::cout << "First 5 of our output: ";
    for (int i = 0; i < 5; i++) std::cout << our_output[i] << " ";
    std::cout << std::endl;

    std::cout << "First 5 of expected: ";
    for (int i = 0; i < 5; i++) std::cout << expected[i] << " ";
    std::cout << std::endl;

    float max_abs_diff = 0;
    float max_rel_diff = 0;
    for (int i = 0; i < 2048; i++) {
        float abs_diff = std::abs(our_output[i] - expected[i]);
        float rel_diff = abs_diff / (std::abs(expected[i]) + 1e-8f);
        if (abs_diff > max_abs_diff) max_abs_diff = abs_diff;
        if (rel_diff > max_rel_diff) max_rel_diff = rel_diff;
    }
    std::cout << "Max absolute diff: " << max_abs_diff << std::endl;
    std::cout << "Max relative diff: " << max_rel_diff << std::endl;
    std::cout << "Index 70: ours=" << our_output[70] << " expected=" << expected[70] << std::endl;

    assert_close(our_output, expected, 5e-3f, 5e-2f);

    std::cout << "RMSNorm matches PyTorch reference!" << std::endl;
    return 0;
}