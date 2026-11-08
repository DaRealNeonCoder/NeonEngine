#pragma once

// Minimal JSON reader for model files (config.json, safetensors headers).
// Parses the whole document into a tree; no writer, no streaming.

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace lve::ml {

class JsonValue {
public:
    enum class Kind { Null, Bool, Number, String, Array, Object };

    Kind kind = Kind::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<JsonValue> array;
    std::map<std::string, JsonValue> object;

    static JsonValue parse(const std::string& text);

    bool isObject() const { return kind == Kind::Object; }
    bool has(const std::string& key) const { return kind == Kind::Object && object.count(key) != 0; }
    const JsonValue& operator[](const std::string& key) const;
    const JsonValue& operator[](size_t index) const;
    size_t size() const { return kind == Kind::Array ? array.size() : object.size(); }

    double asNumber() const;
    int64_t asInt() const { return static_cast<int64_t>(asNumber()); }
    const std::string& asString() const;

    // Value of `key` as a number, or `fallback` when missing.
    double numberOr(const std::string& key, double fallback) const;
};

}  // namespace lve::ml
