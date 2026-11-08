#include "ml_json.hpp"

#include <cctype>
#include <cstdlib>

namespace lve::ml {

namespace {

struct Parser {
    const std::string& s;
    size_t i = 0;

    [[noreturn]] void fail(const char* what) const {
        throw std::runtime_error(std::string("JSON parse error: ") + what + " at offset " + std::to_string(i));
    }

    void ws() {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    }

    bool consume(char c) {
        ws();
        if (i < s.size() && s[i] == c) {
            ++i;
            return true;
        }
        return false;
    }

    void expect(char c) {
        if (!consume(c)) fail("unexpected character");
    }

    static void appendUtf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    uint32_t hex4() {
        if (i + 4 > s.size()) fail("short \\u escape");
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            char c = s[i++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
            else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
            else fail("bad \\u escape");
        }
        return v;
    }

    std::string str() {
        ws();
        if (i >= s.size() || s[i] != '"') fail("expected string");
        ++i;
        std::string out;
        while (true) {
            if (i >= s.size()) fail("unterminated string");
            char c = s[i++];
            if (c == '"') break;
            if (c != '\\') {
                out += c;
                continue;
            }
            if (i >= s.size()) fail("bad escape");
            char e = s[i++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && i + 1 < s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                        i += 2;
                        uint32_t lo = hex4();
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    appendUtf8(out, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
        return out;
    }

    JsonValue value() {
        ws();
        if (i >= s.size()) fail("unexpected end");
        JsonValue v;
        char c = s[i];
        if (c == '{') {
            ++i;
            v.kind = JsonValue::Kind::Object;
            if (consume('}')) return v;
            do {
                std::string key = str();
                expect(':');
                v.object[key] = value();
            } while (consume(','));
            expect('}');
        } else if (c == '[') {
            ++i;
            v.kind = JsonValue::Kind::Array;
            if (consume(']')) return v;
            do {
                v.array.push_back(value());
            } while (consume(','));
            expect(']');
        } else if (c == '"') {
            v.kind = JsonValue::Kind::String;
            v.string = str();
        } else if (s.compare(i, 4, "true") == 0) {
            i += 4;
            v.kind = JsonValue::Kind::Bool;
            v.boolean = true;
        } else if (s.compare(i, 5, "false") == 0) {
            i += 5;
            v.kind = JsonValue::Kind::Bool;
        } else if (s.compare(i, 4, "null") == 0) {
            i += 4;
        } else {
            const char* start = s.c_str() + i;
            char* end = nullptr;
            v.number = std::strtod(start, &end);
            if (end == start) fail("bad value");
            i += static_cast<size_t>(end - start);
            v.kind = JsonValue::Kind::Number;
        }
        return v;
    }
};

}  // namespace

JsonValue JsonValue::parse(const std::string& text) {
    Parser p{text};
    JsonValue v = p.value();
    p.ws();
    if (p.i != text.size()) p.fail("trailing characters");
    return v;
}

const JsonValue& JsonValue::operator[](const std::string& key) const {
    auto it = object.find(key);
    if (kind != Kind::Object || it == object.end()) throw std::runtime_error("JSON: missing key '" + key + "'");
    return it->second;
}

const JsonValue& JsonValue::operator[](size_t index) const {
    if (kind != Kind::Array || index >= array.size()) throw std::runtime_error("JSON: bad array index");
    return array[index];
}

double JsonValue::asNumber() const {
    if (kind == Kind::Bool) return boolean ? 1.0 : 0.0;
    if (kind != Kind::Number) throw std::runtime_error("JSON: value is not a number");
    return number;
}

const std::string& JsonValue::asString() const {
    if (kind != Kind::String) throw std::runtime_error("JSON: value is not a string");
    return string;
}

double JsonValue::numberOr(const std::string& key, double fallback) const {
    return has(key) ? (*this)[key].asNumber() : fallback;
}

}  // namespace lve::ml
