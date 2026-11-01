#include "digit_recognizer_train.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace lve {

namespace {

constexpr size_t SRC_SIZE = 28;
constexpr size_t DST_SIZE = 27;
constexpr size_t IN = DST_SIZE * DST_SIZE;  // 729
constexpr size_t HID = 128;
constexpr size_t OUT = 10;

// Reads a whole file into memory.
bool readFile(const std::filesystem::path& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamsize size = f.tellg();
    f.seekg(0);
    out.resize(static_cast<size_t>(size));
    return static_cast<bool>(f.read(reinterpret_cast<char*>(out.data()), size));
}

uint32_t readBigEndian32(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

// Parses an IDX3 image file and resizes each 28x28 image to 27x27 with bilinear sampling.
bool readImages(const std::filesystem::path& path, std::vector<float>& out, size_t& count) {
    std::vector<uint8_t> raw;
    if (!readFile(path, raw) || raw.size() < 16) return false;
    if (readBigEndian32(raw.data()) != 2051) return false;

    const size_t n = readBigEndian32(raw.data() + 4);
    const size_t rows = readBigEndian32(raw.data() + 8);
    const size_t cols = readBigEndian32(raw.data() + 12);
    if (rows != SRC_SIZE || cols != SRC_SIZE || raw.size() < 16 + n * rows * cols) return false;

    // Source coordinate for each destination pixel (same mapping as align_corners=False).
    std::array<size_t, DST_SIZE> lo{};
    std::array<size_t, DST_SIZE> hi{};
    std::array<float, DST_SIZE> frac{};
    for (size_t d = 0; d < DST_SIZE; ++d) {
        float src = (float(d) + 0.5f) * float(SRC_SIZE) / float(DST_SIZE) - 0.5f;
        src = std::clamp(src, 0.0f, float(SRC_SIZE - 1));
        lo[d] = size_t(std::floor(src));
        hi[d] = std::min(lo[d] + 1, SRC_SIZE - 1);
        frac[d] = src - float(lo[d]);
    }

    out.resize(n * IN);
    const uint8_t* pixels = raw.data() + 16;
    for (size_t k = 0; k < n; ++k) {
        const uint8_t* img = pixels + k * SRC_SIZE * SRC_SIZE;
        float* dst = out.data() + k * IN;
        for (size_t y = 0; y < DST_SIZE; ++y) {
            for (size_t x = 0; x < DST_SIZE; ++x) {
                const float a = img[lo[y] * SRC_SIZE + lo[x]];
                const float b = img[lo[y] * SRC_SIZE + hi[x]];
                const float c = img[hi[y] * SRC_SIZE + lo[x]];
                const float d = img[hi[y] * SRC_SIZE + hi[x]];
                const float top = a + (b - a) * frac[x];
                const float bot = c + (d - c) * frac[x];
                dst[y * DST_SIZE + x] = (top + (bot - top) * frac[y]) / 255.0f;
            }
        }
    }
    count = n;
    return true;
}

// Parses an IDX1 label file.
bool readLabels(const std::filesystem::path& path, std::vector<uint8_t>& out) {
    std::vector<uint8_t> raw;
    if (!readFile(path, raw) || raw.size() < 8) return false;
    if (readBigEndian32(raw.data()) != 2049) return false;
    const size_t n = readBigEndian32(raw.data() + 4);
    if (raw.size() < 8 + n) return false;
    out.assign(raw.begin() + 8, raw.begin() + 8 + n);
    return true;
}

// Round-to-nearest-even float -> IEEE binary16 bits.
uint16_t floatToHalf(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t expF = (x >> 23) & 0xFFu;
    uint32_t mant = x & 0x7FFFFFu;

    if (expF == 0xFFu) return uint16_t(sign | 0x7C00u | (mant ? 0x200u : 0u));  // inf / nan

    const int32_t exp = int32_t(expF) - 127 + 15;
    if (exp >= 31) return uint16_t(sign | 0x7C00u);  // overflow -> inf
    if (exp <= 0) {                                   // subnormal or zero
        if (exp < -10) return uint16_t(sign);
        mant |= 0x800000u;
        const uint32_t shift = uint32_t(14 - exp);
        uint32_t half = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half & 1u))) ++half;
        return uint16_t(sign | half);
    }
    uint32_t half = (uint32_t(exp) << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) ++half;  // carry into exponent is correct
    return uint16_t(sign | half);
}

bool writeBinary(const std::filesystem::path& path, const void* data, size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(f);
}

}  // namespace

bool loadMnistSplit(const std::string& dir, MnistSplit& out) {
    namespace fs = std::filesystem;
    const fs::path base(dir);
    size_t trainN = 0;
    size_t testN = 0;
    if (!readImages(base / "train-images-idx3-ubyte", out.trainX, trainN)) return false;
    if (!readLabels(base / "train-labels-idx1-ubyte", out.trainY)) return false;
    if (!readImages(base / "t10k-images-idx3-ubyte", out.testX, testN)) return false;
    if (!readLabels(base / "t10k-labels-idx1-ubyte", out.testY)) return false;
    return out.trainY.size() == trainN && out.testY.size() == testN;
}

bool saveWeights(const std::string& dir, const MlpParams& params) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) return false;

    const fs::path base(dir);
    std::vector<uint16_t> w1h(params.w1.size()), w2h(params.w2.size());
    for (size_t i = 0; i < w1h.size(); ++i) w1h[i] = floatToHalf(params.w1[i]);
    for (size_t i = 0; i < w2h.size(); ++i) w2h[i] = floatToHalf(params.w2[i]);

    return writeBinary(base / "weight1_f16.bin", w1h.data(), w1h.size() * sizeof(uint16_t)) &&
           writeBinary(base / "bias1_f32.bin", params.b1.data(), params.b1.size() * sizeof(float)) &&
           writeBinary(base / "weight2_f16.bin", w2h.data(), w2h.size() * sizeof(uint16_t)) &&
           writeBinary(base / "bias2_f32.bin", params.b2.data(), params.b2.size() * sizeof(float));
}

}  // namespace lve
