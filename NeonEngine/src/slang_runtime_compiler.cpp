#include "slang_runtime_compiler.hpp"

#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace lve {

namespace {

std::string readText(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("SlangRuntimeCompiler: cannot open " + path);
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

std::string diagText(slang::IBlob* diag) {
    if (!diag) return {};
    return std::string(static_cast<const char*>(diag->getBufferPointer()), diag->getBufferSize());
}

void check(SlangResult result, slang::IBlob* diag, const char* step) {
    if (SLANG_FAILED(result)) {
        throw std::runtime_error(std::string("SlangRuntimeCompiler: ") + step + " failed:\n" + diagText(diag));
    }
}

slang::CompilerOptionEntry intOption(slang::CompilerOptionName name, int32_t value) {
    slang::CompilerOptionEntry entry{};
    entry.name = name;
    entry.value.kind = slang::CompilerOptionValueKind::Int;
    entry.value.intValue0 = value;
    return entry;
}

}  // namespace

SlangRuntimeCompiler::SlangRuntimeCompiler() {
    check(slang::createGlobalSession(globalSession.writeRef()), nullptr, "createGlobalSession");
}

std::vector<uint32_t> SlangRuntimeCompiler::compileEntryPoint(
    const std::string& sourcePath,
    const std::string& includeDir,
    const std::string& entryName) {

    // Same settings as the slangc command line in shaders/ML/tests/compile_digit_recognizer.ps1
    slang::TargetDesc target{};
    target.structureSize = sizeof(slang::TargetDesc);
    target.format = SLANG_SPIRV;
    target.profile = globalSession->findProfile("spirv_1_5");

    slang::CompilerOptionEntry options[] = {
        intOption(slang::CompilerOptionName::EmitSpirvDirectly, 1),
        intOption(slang::CompilerOptionName::VulkanUseEntryPointName, 1),  // -fvk-use-entrypoint-name
        intOption(slang::CompilerOptionName::VulkanEmitReflection, 1),     // -fspv-reflect
        intOption(slang::CompilerOptionName::GLSLForceScalarLayout, 1),    // -fvk-use-scalar-layout (required by tensor.slang)
        intOption(slang::CompilerOptionName::Optimization, SLANG_OPTIMIZATION_LEVEL_MAXIMAL),  // -O3
    };

    const char* searchPaths[] = { includeDir.c_str() };

    slang::SessionDesc sessionDesc{};
    sessionDesc.structureSize = sizeof(slang::SessionDesc);
    sessionDesc.targets = &target;
    sessionDesc.targetCount = 1;
    sessionDesc.searchPaths = searchPaths;
    sessionDesc.searchPathCount = 1;
    sessionDesc.compilerOptionEntries = options;
    sessionDesc.compilerOptionEntryCount = static_cast<uint32_t>(std::size(options));

    // A session is created per call: each call loads the module under the same name,
    // and a fresh session avoids collisions between entry points from the same file.
    Slang::ComPtr<slang::ISession> session;
    check(globalSession->createSession(sessionDesc, session.writeRef()), nullptr, "createSession");

    const std::string source = readText(sourcePath);
    Slang::ComPtr<slang::IBlob> diag;
    slang::IModule* module = session->loadModuleFromSourceString(
        "digit_recognizer", sourcePath.c_str(), source.c_str(), diag.writeRef());
    if (!module) {
        throw std::runtime_error("SlangRuntimeCompiler: failed to load " + sourcePath + ":\n" + diagText(diag));
    }

    slang::IEntryPoint* entry = nullptr;
    check(module->findEntryPointByName(entryName.c_str(), &entry), diag.get(), "findEntryPointByName");

    slang::IComponentType* parts[] = { module, entry };
    Slang::ComPtr<slang::IComponentType> composed;
    check(session->createCompositeComponentType(parts, 2, composed.writeRef(), diag.writeRef()),
        diag.get(), "createCompositeComponentType");

    Slang::ComPtr<slang::IComponentType> linked;
    check(composed->link(linked.writeRef(), diag.writeRef()), diag.get(), "link");

    Slang::ComPtr<slang::IBlob> code;
    check(linked->getEntryPointCode(0, 0, code.writeRef(), diag.writeRef()),
        diag.get(), "getEntryPointCode");

    std::vector<uint32_t> words(code->getBufferSize() / sizeof(uint32_t));
    std::memcpy(words.data(), code->getBufferPointer(), words.size() * sizeof(uint32_t));
    return words;
}

}  // namespace lve
