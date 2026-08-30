// npy_loader.h
//
// Minimal reader for numpy .npy files, restricted to what dump_reference.py
// actually produces: float32, C-contiguous arrays. Not a general-purpose
// .npy parser -- the header dictionary is scanned for the two fields we
// need (dtype, shape) with plain string search rather than a real parser,
// since we control how these files were written.

#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Holds the result of loading one .npy file: the raw float data in a flat
// buffer, plus the shape that tells you how to interpret it.
// The data is flat (not nested) because that's simplest to work with in
// C++; the shape tells the caller how to index into it.
struct NpyArray {
    std::vector<float> data;
    std::vector<int64_t> shape;

    // return type helper
    int64_t num_elements() const{
        int64_t total = 1;
        for (int64_t num : shape){
            total *= num;
        }
        return total;
    }
};

NpyArray load_npy(const std::string& path) {
    // open the file
    std::ifstream file(path, std::ios::binary);
    
    // check it opened, throw if not
    if (!file){
        throw std::runtime_error("File has not been opened. Error");
    }
    
    // read 6 bytes into a buffer called magic
    char magic[6];
    file.read(magic, 6);
    
    // check magic == "\x93NUMPY", throw if not
    if (std::string(magic, 6) != "\x93NUMPY"){
        throw std::runtime_error("Magic String Incorrect. Error");
    }

    // read version_major (1 byte) and version_minor (1 byte)
    uint8_t version_major = 0;
    file.read(reinterpret_cast<char*>(&version_major), 1);

    uint8_t version_minor = 0;
    file.read(reinterpret_cast<char*>(&version_minor), 1);
    
    // declare header_len as uint32_t, will hold the final value either way
    uint32_t header_len = 0;
    
    // if version_major == 1: read a uint16_t and assign it into header_len
    if (version_major == 1){
        uint16_t header_len_16 = 0;
        file.read(reinterpret_cast<char*>(&header_len_16), 2);
        header_len = header_len_16;
    }

    // else: read directly into header_len (4 bytes)
    else{
        file.read(reinterpret_cast<char*>(&header_len), 4);
    }

    // declare header as a string of length header_len, filled with spaces
    std::string header(header_len, ' ');
    
    // read header_len bytes into header
    file.read(header.data(), header_len);
    
    // if header does not contain "'<f4'", throw
    if (header.find("'<f4'") == std::string::npos){
        throw std::runtime_error("Header Incorrect. Error");
    }

    // find "'shape': (" in header, throw if not found
    size_t shape_start = header.find("'shape': (");
    if (shape_start == std::string::npos) {
        throw std::runtime_error("'shape': ( not found. Error");
    }
    
    // move shape_start past that substring
    shape_start += std::string("'shape': (").size();
    
    // find the closing ')' from shape_start onward
    size_t shape_end = header.find(')', shape_start);

    // extract the substring between them into shape_str
    std::string shape_str = header.substr(shape_start, shape_end - shape_start);

    // declare an empty vector<int64_t> called shape
    std::vector<int64_t> shape;
    
    // loop through shape_str splitting on commas, push_back each number
    size_t pos = 0;
    while (pos < shape_str.size()) {
        size_t comma = shape_str.find(',', pos);
        std::string token = shape_str.substr(pos, comma - pos);
        if (!token.empty()) {
            shape.push_back(std::stoll(token));
        }
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }

    // declare NpyArray result
    NpyArray result;

    // assign shape into result.shape
    result.shape = shape;
    
    // get expected element count by calling result.num_elements()
    int64_t expected_count = result.num_elements();
    
    // resize result.data to that count
    result.data.resize(expected_count);
    
    // read (expected_count * sizeof(float)) bytes into result.data.data(), cast to char*
    file.read(reinterpret_cast<char*>(result.data.data()), expected_count * sizeof(float));

        if (!file) {
        throw std::runtime_error("Ran out of data while reading floats -- file may be truncated: " + path);
    }

    return result;

}
