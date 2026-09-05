#include "gguf_reader.h"

int main(){

    MappedFile result = map_file("models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf");

    inspect_header(result);
    size_t metadata_end = inspect_metadata(result, 23);

    return 0;
}