#pragma once

#include "lve_device.hpp"
#include "lve_window.hpp"

namespace lve {

// Serve mode must call this before constructing DigitRecognizerApp: the engine prints
// device enumeration to stdout, and this moves those prints to stderr so only protocol bytes reach stdout.
void beginServeProtocol();

class DigitRecognizerApp {
public:
    static constexpr int WIDTH = 320;
    static constexpr int HEIGHT = 240;

    DigitRecognizerApp() = default;
    void run();
    // Long-lived inference process driven over stdin/stdout (see digit_recognizer_app.cpp).
    void serve();

private:
    LveWindow lveWindow{ WIDTH, HEIGHT, "Digit Recognizer (headless)" };
    LveDevice lveDevice{ lveWindow };
};

}  // namespace lve