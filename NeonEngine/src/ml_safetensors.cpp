#include "ml_safetensors.hpp"

#include "ml_json.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace lve::ml {

namespace {

bool seekTo(std::FILE* f, uint64_t offset) {
#ifdef _WIN32
    return _fseeki64(f, static_cast<long long>(offset), SEEK_SET) == 0;
#else
    return fseeko(f, static_cast<off_t>(offset), SEEK_SET) == 0;
#endif
}

}  // namespace

uint64_t SafeTensorInfo::numel() const {
    uint64_t n = 1;
    for (uint64_t d : shape) n *= d;
    return n;
}

float bf16ToF32(uint16_t bits) {
    uint32_t u = uint32_t(bits) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

float f16ToF32(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) {
            u = sign;
        } else {  // subnormal: renormalise
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3FFu;
            u = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000u | (mant << 13);
    } else {
        u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

uint16_t f32ToF16(float v) {
    // Round to nearest even; overflow -> inf.
    uint32_t u;
    std::memcpy(&u, &v, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = int32_t((u >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = u & 0x7FFFFFu;
    if (((u >> 23) & 0xFFu) == 0xFFu) return uint16_t(sign | 0x7C00u | (mant ? 0x200u : 0u));
    if (exp >= 31) return uint16_t(sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return uint16_t(sign);
        mant |= 0x800000u;
        uint32_t shift = uint32_t(14 - exp);
        uint32_t half = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1u);
        uint32_t mid = 1u << (shift - 1u);
        if (rem > mid || (rem == mid && (half & 1u))) ++half;
        return uint16_t(sign | half);
    }
    uint32_t half = sign | (uint32_t(exp) << 10) | (mant >> 13);
    uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) ++half;  // may carry into the exponent (correct)
    return uint16_t(half);
}

SafeTensors::SafeTensors(const std::string& directory) {
    namespace fs = std::filesystem;
    std::vector<fs::path> found;
    if (fs::is_regular_file(directory)) {
        found.push_back(directory);
    } else {
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file() && entry.path().extension() == ".safetensors") found.push_back(entry.path());
        }
    }
    std::sort(found.begin(), found.end());
    if (found.empty()) throw std::runtime_error("SafeTensors: no .safetensors files in " + directory);

    for (const auto& path : found) {
        std::FILE* f = std::fopen(path.string().c_str(), "rb");
        if (!f) throw std::runtime_error("SafeTensors: cannot open " + path.string());
        files.push_back(f);
        paths.push_back(path.string());

        uint64_t headerLen = 0;
        if (std::fread(&headerLen, 8, 1, f) != 1 || headerLen > (1ull << 30))
            throw std::runtime_error("SafeTensors: bad header in " + path.string());
        std::string header(headerLen, '\0');
        if (std::fread(header.data(), 1, headerLen, f) != headerLen)
            throw std::runtime_error("SafeTensors: truncated header in " + path.string());
        const uint64_t dataStart = 8 + headerLen;

        JsonValue root = JsonValue::parse(header);
        for (const auto& [name, v] : root.object) {
            if (name == "__metadata__") continue;
            SafeTensorInfo info;
            info.dtype = v["dtype"].asString();
            for (const auto& d : v["shape"].array) info.shape.push_back(uint64_t(d.asInt()));
            const auto& offs = v["data_offsets"];
            uint64_t b = uint64_t(offs[0].asInt());
            uint64_t e = uint64_t(offs[1].asInt());
            info.begin = dataStart + b;
            info.bytes = e - b;
            info.file = files.size() - 1;
            tensors[name] = std::move(info);
        }
    }
}

SafeTensors::~SafeTensors() {
    for (std::FILE* f : files) std::fclose(f);
}

const SafeTensorInfo& SafeTensors::info(const std::string& name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) throw std::runtime_error("SafeTensors: tensor not found: " + name);
    return it->second;
}

void SafeTensors::readRange(const std::string& name, uint64_t offset, void* dst, uint64_t bytes) const {
    const SafeTensorInfo& t = info(name);
    if (offset + bytes > t.bytes) throw std::runtime_error("SafeTensors: read past end of " + name);
    std::FILE* f = files[t.file];
    if (!seekTo(f, t.begin + offset)) throw std::runtime_error("SafeTensors: seek failed for " + name);
    uint8_t* out = static_cast<uint8_t*>(dst);
    uint64_t done = 0;
    while (done < bytes) {
        size_t chunk = size_t(std::min<uint64_t>(bytes - done, 64ull << 20));
        size_t got = std::fread(out + done, 1, chunk, f);
        if (got != chunk) throw std::runtime_error("SafeTensors: short read for " + name);
        done += got;
    }
}

void SafeTensors::read(const std::string& name, void* dst, uint64_t bytes) const {
    const SafeTensorInfo& t = info(name);
    if (bytes != t.bytes)
        throw std::runtime_error("SafeTensors: size mismatch for " + name + " (" + std::to_string(bytes) +
                                 " vs " + std::to_string(t.bytes) + ")");
    readRange(name, 0, dst, bytes);
}

std::vector<uint8_t> SafeTensors::readBytes(const std::string& name) const {
    std::vector<uint8_t> out(size_t(info(name).bytes));
    read(name, out.data(), out.size());
    return out;
}

std::vector<float> SafeTensors::readF32(const std::string& name) const {
    const SafeTensorInfo& t = info(name);
    std::vector<uint8_t> raw = readBytes(name);
    std::vector<float> out(size_t(t.numel()));
    if (t.dtype == "F32") {
        std::memcpy(out.data(), raw.data(), out.size() * 4);
    } else if (t.dtype == "BF16" || t.dtype == "F16") {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(raw.data());
        const bool bf = t.dtype == "BF16";
        for (size_t i = 0; i < out.size(); ++i) out[i] = bf ? bf16ToF32(src[i]) : f16ToF32(src[i]);
    } else {
        throw std::runtime_error("SafeTensors: cannot convert " + t.dtype + " to f32 for " + name);
    }
    return out;
}

}  // namespace lve::ml
