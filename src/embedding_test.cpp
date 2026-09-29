#include "gguf_reader.h"
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

    // Token ID 450 ("The"), a normal, common word, not a special token.
    std::vector<float> our_embedding = read_tensor_data(
        mapped, tensor_data_start + token_embd_offset, 8, 2048, 450, 2048);

    NpyArray reference = load_npy("reference_tensors/embed_token_450.npy");

    std::cout << "First 5 of our embedding: ";
    for (int i = 0; i < 5; i++) std::cout << our_embedding[i] << " ";
    std::cout << std::endl;

    std::cout << "First 5 of expected: ";
    for (int i = 0; i < 5; i++) std::cout << reference.data[i] << " ";
    std::cout << std::endl;

    assert_close(our_embedding, reference.data, 1e-3f, 1e-2f);

    std::cout << "Embedding lookup matches PyTorch reference!" << std::endl;
    return 0;
}