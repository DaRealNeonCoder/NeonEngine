#pragma once
#include "lve_buffer.hpp"
#include "lve_device.hpp"
// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm.hpp>
// std
#include <memory>
#include <vector>
#include <string>

namespace lve {

    struct LveMaterial {
        glm::vec4 albedo{1.f, 1.f, 1.f, 1.f};
        glm::vec4 emission{0.f, 0.f, 0.f, 0.f};
        std::string albedoMap{};
        uint32_t type{ 0 };
    };
    class LveModel {
    public:
        struct Vertex {
            glm::vec3 position{};
            glm::vec3 color{};
            glm::vec3 normal{};
            glm::vec2 uv{};
            static std::vector<VkVertexInputBindingDescription> getBindingDescriptions();
            static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions();
            bool operator==(const Vertex& other) const {
                return position == other.position && color == other.color && normal == other.normal &&
                    uv == other.uv;
            }
        };
        struct Builder {
            std::vector<Vertex> vertices{};
            std::vector<uint32_t> indices{};
            void loadModel(const std::string& filepath);
        };
        LveModel(LveDevice& device, const LveModel::Builder& builder);
        ~LveModel();
        LveModel(const LveModel&) = delete;
        LveModel& operator=(const LveModel&) = delete;
        static std::unique_ptr<LveModel> createModelFromFile(
            LveDevice& device, const std::string& filepath);
        void bind(VkCommandBuffer commandBuffer);
        void draw(VkCommandBuffer commandBuffer);
        const std::vector<Vertex>& getVertices() const { return vertices; }
        const std::vector<uint32_t>& getIndices() const { return indices; }
   
    private:
        void createVertexBuffers(const std::vector<Vertex>& vertices);
        void createIndexBuffers(const std::vector<uint32_t>& indices);
        LveDevice& lveDevice;
        std::unique_ptr<LveBuffer> vertexBuffer;
        uint32_t vertexCount;
        bool hasIndexBuffer = false;
        std::unique_ptr<LveBuffer> indexBuffer;
        uint32_t indexCount;
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
    };

}  // namespace lve