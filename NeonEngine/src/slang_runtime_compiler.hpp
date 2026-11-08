#pragma once

#include <slang/slang.h>
#include <slang/slang-com-ptr.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace lve {

// Compiles a Slang compute entry point to SPIR-V in memory, so shaders do not
// need to be precompiled with slangc before the app runs.
class SlangRuntimeCompiler {
public:
    SlangRuntimeCompiler();

    // sourcePath:  the .slang file containing the entry point
    // includeDir:  directory searched for `import`ed Slang modules
    // entryName:   a [shader("compute")] function in sourcePath
    std::vector<uint32_t> compileEntryPoint(
        const std::string& sourcePath,
        const std::string& includeDir,
        const std::string& entryName);

    // Same, with preprocessor defines (-D name=value).
    std::vector<uint32_t> compileEntryPoint(
        const std::string& sourcePath,
        const std::string& includeDir,
        const std::string& entryName,
        const std::vector<std::pair<std::string, std::string>>& defines);

private:
    Slang::ComPtr<slang::IGlobalSession> globalSession;
};

}  // namespace lve
