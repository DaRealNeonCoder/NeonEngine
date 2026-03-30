#pragma once
#include "lve_device.hpp"
#include "lve_pipeline.hpp"
#include "lve_game_object.hpp"
#include "lve_frame_info.hpp"
#include <memory>
#include <vector>
#include <vulkan.h>
#include <glm.hpp>

namespace lve {

    struct SimplePushConstantData {
        glm::mat4 modelMatrix{1.f};
        glm::mat4 normalMatrix{1.f};
    };

    struct GBufferRenderTargets {
        VkImage positionImage;
        VkImageView positionView;
        VkImage normalImage;
        VkImageView normalView;
        VkImage barycentricImage;
        VkImageView barycentricView;
        VkImage motionImage;
        VkImageView motionView;


        VkImage historyColourImage;
        VkImageView historyColorView;
        VkImage historyLengthImage;
        VkImageView historyLengthView;

        VkImage historyColourImage2;
        VkImageView historyColorView2;
        // In your GBufferRenderTargets struct, add:
        VkImage        depthImage;
        VkImageView    depthView;
    };

    class RayTracingRast {
    public:
        bool pingPong = true;

        RayTracingRast(LveDevice& device, VkRenderPass pass,
            VkDescriptorSetLayout globalSetLayout, VkDescriptorSetLayout computeSetLayout, VkDescriptorSetLayout computeSetLayout2, VkExtent2D _swapChainExtents);
        ~RayTracingRast();

        RayTracingRast(const RayTracingRast&) = delete;
        RayTracingRast& operator=(const RayTracingRast&) = delete;

        void renderGameObjects(RayFrameInfo& frameInfo);
        GBufferRenderTargets targets;
        void createGBufferImages(LveDevice& device, GBufferRenderTargets& gBuffers);
        VkExtent2D swapChainExtents;
        void renderCompute(
            VkCommandBuffer commandBuffer,
            VkDescriptorSet descriptorSet,
            VkDescriptorSet descriptorSet2,
            uint32_t groupsX,
            uint32_t groupsY);
        void renderAtrous(
            VkCommandBuffer  commandBuffer,
            VkDescriptorSet  setA,          // input=storageImage,  output=writeImage
            VkDescriptorSet  setB,          // input=writeImage,    output=storageImage
            uint32_t         groupsX,
            uint32_t         groupsY,
            uint32_t         numIterations);
        void barrierStorageToCompute(
            VkCommandBuffer cmd,
            VkImage& storageImage);
        void barrierGBufferToCompute(
            VkCommandBuffer cmd,
            GBufferRenderTargets& gBuffers); 
        VkSampler gBufferSampler;
    private:
        LveDevice& lveDevice;
        VkPipelineLayout pipelineLayout{};
        std::unique_ptr<LvePipeline> lvePipeline;

        VkPipelineLayout computePipelineLayout{};
        VkPipeline computePipeline{};
        VkShaderModule computeShaderModule{};

        VkPipelineLayout atrousPipelineLayout{};
        VkPipeline atrousPipeline{};
        VkShaderModule atrousShaderModule{};

        void createPipelineLayout(VkDescriptorSetLayout globalSetLayout);
        void createPipeline(VkRenderPass pass);

        void CreateComputePipelineLayout(VkDescriptorSetLayout setLayout);
        void CreateComputePipeline(); 
        void CreateAtrousPipelineLayout(VkDescriptorSetLayout setLayout);
        void CreateAtrousPipeline();


        void CreateSampler();
        

    };

} // namespace lve