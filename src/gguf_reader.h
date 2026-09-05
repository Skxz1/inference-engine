// gguf_reader.h
//
//  This file reads GGUF model files, we use mmap rather than ifstream because we do not need to open the whole file which would be very large
//  We are just inspecting parts of the header 
//  Distinction: Still will load the file into the disk but will only retrieve data into memory for what you need

#pragma once

#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

struct MappedFile {
    void* data;
    size_t size;
};

MappedFile map_file(const std::string& path) {
    // open the file (O_RDONLY), throw if it fails
    int fd = open(path.c_str(), O_RDONLY);

    if (fd == -1){
        throw std::runtime_error("Failed to open file: " + path);
    }
    
    // fstat to get file size
    struct stat st;
    fstat(fd, &st);

    off_t file_size = st.st_size;
    
    // mmap the whole file, throw if MAP_FAILED
    void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

    if (mapped == MAP_FAILED){
        throw std::runtime_error("Failed to mmap file: " + path);
    }
    
    // close the file descriptor (safe to do right after mmap, the mapping stays valid)
    close(fd);
    
    // return a MappedFile with the pointer and size
    return MappedFile{mapped, static_cast<size_t>(file_size)};
}

void inspect_header(const MappedFile& mapped) {
    char* bytes = reinterpret_cast<char*>(mapped.data);
    
    uint32_t magic;
    std::memcpy(&magic, bytes, 4);
    
    // next: check magic, then read version, tensor_count, metadata_kv_count
    if (magic != 0x46554747){
        throw std::runtime_error("Invalid Magic Number! Failed");
    }

    uint32_t version;
    std::memcpy(&version, bytes + 4, 4);

    if (version < 3){
        throw std::runtime_error("Version Number is unsupported! Failed");
    }

    uint64_t tensor_count;
    std::memcpy(&tensor_count, bytes + 8, 8);

    uint64_t metadata_kv_count;
    std::memcpy(&metadata_kv_count, bytes + 16, 8);

    std::cout << "Magic: 0x" << std::hex << magic << std::dec << std::endl;
    std::cout << "Version: " << version << std::endl;
    std::cout << "Tensor count: " << tensor_count << std::endl;
    std::cout << "Metadata KV count: " << metadata_kv_count << std::endl;

}


struct GgufString {
    std::string value;
    uint64_t bytes_consumed;
};

GgufString read_gguf_string(const char* bytes) {
    // read an 8-byte length from bytes
    uint64_t length;
    std::memcpy(&length, bytes, 8);     // length holds copy of the first 8 bytes from location bytes
    
    // build a std::string from the next `length` bytes, starting after the 8-byte prefix
    std::string value(bytes + 8, length);
    
    // return a GgufString with the string and total bytes consumed (8 + length)
    return GgufString{value, 8 + length};
}

size_t inspect_metadata(const MappedFile& mapped, uint64_t metadata_kv_count) {
    char* bytes = reinterpret_cast<char*>(mapped.data);
    size_t pos = 24;

    for (uint64_t i = 0; i < metadata_kv_count; i++){
        
        GgufString key = read_gguf_string(bytes + pos);
        std::cout << "First metadata key: " << key.value << std::endl;
        
        pos += key.bytes_consumed;
        
        uint32_t value_type;
        std::memcpy(&value_type, bytes + pos, 4);
        std::cout << "Value type: " << value_type << std::endl;
        
        pos += 4;  // advance past the type tag we just read
        std::string value_str;
        

        if (value_type == 8) {
            GgufString value = read_gguf_string(bytes + pos);
            value_str = value.value;
            pos += value.bytes_consumed;
        } else if (value_type == 4) {
            uint32_t num;
            std::memcpy(&num, bytes + pos, 4);
            value_str = std::to_string(num);
            pos += 4;
        }else if (value_type == 6){
            float num;
            std::memcpy(&num, bytes + pos, 4);
            value_str = std::to_string(num);
            pos += 4;

        } else if (value_type == 9){
            uint32_t element_type;
            std::memcpy(&element_type, bytes + pos, 4);
            pos += 4;

            uint64_t array_len;
            std::memcpy(&array_len, bytes + pos, 8);
            pos += 8;

            value_str = "Array of " + std::to_string(array_len) + "Elements, type " + std::to_string(element_type);

            // Actual loop through and skip past all array_len elements
            if (element_type == 8){
                for (uint64_t j = 0; j < array_len; j++){
                    GgufString element = read_gguf_string(bytes + pos);
                    pos += element.bytes_consumed;
                }
            }
            else if (element_type == 4 || element_type == 6 || element_type == 5){
                pos += 4 * array_len;
            }
            else {
                throw std::runtime_error("Unhandled array element type: " + std::to_string(element_type));
            }

        }else {
            throw std::runtime_error("Unhandled metadata value type: " + std::to_string(value_type));
        }

        std::cout << "Value: " << value_str << std::endl;
    }
    return pos;
}

void inspect_tensors(const MappedFile& mapped, uint64_t tensor_count, size_t start_pos){
    char * bytes = reinterpret_cast<char*>(mapped.data);
    size_t pos = start_pos;

    GgufString name = read_gguf_string(bytes + pos);
    pos += name.bytes_consumed;

    uint32_t n_dimensions;
    std::memcpy(&n_dimensions, bytes + pos, 4);
    pos += 4;

    std::cout << "Tensor Name: " << name.value << std::endl;
    std::cout << "Number of Tensor Dimension: " << n_dimensions << std::endl; 

    std::vector<uint64_t> dimensions;
    for (uint32_t i = 0; i < n_dimensions; i ++){
        uint64_t dim;
        std::memcpy(&dim, bytes + pos, 8);
        dimensions.push_back(dim);
        pos += 8;
    }
    std::cout << "Dimensions: ";
    for (uint64_t d : dimensions){
        std::cout << d << " ";
    }
    std::cout << std::endl;

    uint32_t type;
    std::memcpy(&type, bytes + pos, 4);
    pos += 4;

    uint64_t offset;
    std::memcpy(&offset, bytes + pos, 8);
    pos += 8;

    std::cout << "dimension type: " << type << std::endl; 
    std::cout << "Dimension Offset: " << offset << std::endl; 

}