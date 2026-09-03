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

}