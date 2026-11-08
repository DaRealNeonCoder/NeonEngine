#pragma once

// Streaming reader for .safetensors checkpoints (one or more files in a
// directory). Only the JSON header is held in memory; tensor bytes are read on
// demand, so a 13 GB checkpoint never has to fit in RAM at once.
//
// File format: u64 little-endian header length, JSON header
// {"name": {"dtype": "BF16", "shape": [...], "data_offsets": [begin, end]}, ...},
// then the raw data; offsets are relative to the end of the header.

#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace lve::ml {

struct SafeTensorInfo {
    std::string dtype;            // "BF16", "F32", "F16", "U8", "I32", ...
    std::vector<uint64_t> shape;
    uint64_t begin = 0;           // absolute byte offset in the file
    uint64_t bytes = 0;
    size_t file = 0;              // index into SafeTensors::files
    uint64_t numel() const;
};

class SafeTensors {
public:
    // Opens every *.safetensors file in `directory`.
    explicit SafeTensors(const std::string& directory);
    ~SafeTensors();

    SafeTensors(const SafeTensors&) = delete;
    SafeTensors& operator=(const SafeTensors&) = delete;

    bool has(const std::string& name) const { return tensors.count(name) != 0; }
    const SafeTensorInfo& info(const std::string& name) const;
    const std::map<std::string, SafeTensorInfo>& all() const { return tensors; }

    // Raw bytes of a whole tensor.
    void read(const std::string& name, void* dst, uint64_t bytes) const;
    std::vector<uint8_t> readBytes(const std::string& name) const;
    // Reads a byte range [offset, offset + bytes) inside a tensor.
    void readRange(const std::string& name, uint64_t offset, void* dst, uint64_t bytes) const;

    // Tensor converted to fp32 (supports F32, BF16, F16).
    std::vector<float> readF32(const std::string& name) const;

private:
    std::vector<std::FILE*> files;
    std::vector<std::string> paths;
    std::map<std::string, SafeTensorInfo> tensors;
};

// bf16 / fp16 helpers shared by the loaders.
float bf16ToF32(uint16_t bits);
uint16_t f32ToF16(float v);
float f16ToF32(uint16_t bits);

}  // namespace lve::ml
