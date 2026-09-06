#include "gguf_reader.h"

int main(){

    MappedFile result = map_file("models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf");

    inspect_header(result);
    size_t metadata_end = inspect_metadata(result, 23);
    size_t tensor_index_end = inspect_tensors(result, 201, metadata_end);

    size_t alignment = 32;
    size_t tensor_data_start = ((tensor_index_end + alignment - 1) / alignment) * alignment;

    std::cout << "Tensor data starts at: " << tensor_data_start << std::endl;

    return 0;
}