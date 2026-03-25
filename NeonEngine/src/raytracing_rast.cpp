#pragma once 

#include "raytracing_rast.hpp"
#include "lve_pipeline.hpp"
#include "lve_swap_chain.hpp"
#include "lve_descriptors.hpp"

#include <iostream>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace lve {

    RayTracingRast::RayTracingRast(LveDevice& device, VkRenderPass pass,
        VkDescriptorSetLayout globalSetLayout, VkDescriptorSetLayout computeSetLayout, VkDescriptorSetLayout computeSetLayout2, VkExtent2D _swapChainExtents)
        : lveDevice{ device }, swapChainExtents{ _swapChainExtents } {

        createGBufferImages(device, targets);
        createPipelineLayout(globalSetLayout);
        createPipeline(pass);
        CreateComputePipelineLayout(computeSetLayout);
        CreateComputePipeline();
        CreateSampler();

        CreateAtrousPipelineLayout(computeSetLayout);
        CreateAtrousPipeline();
    }

    RayTracingRast::~RayTracingRast() {
        vkDestroyPipelineLayout(lveDevice.device(), pipelineLayout, nullptr);
    }

    void RayTracingRast::createPipelineLayout(VkDescriptorSetLayout globalSetLayout) {
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(SimplePushConstantData);

        std::vector<VkDescriptorSetLayout> descriptorSetLayouts{ globalSetLayout };

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
        pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

        if (vkCreatePipelineLayout(lveDevice.device(), &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }
    }

    void RayTracingRast::createPipeline(VkRenderPass pass) {
        assert(pipelineLayout != nullptr && "Cannot create pipeline before pipeline layout");

        PipelineConfigInfo pipelineConfig{};
        LvePipeline::defaultPipelineConfigInfo(pipelineConfig);

        pipelineConfig.renderPass = pass;
        pipelineConfig.pipelineLayout = pipelineLayout;

        // 6 colour attachments: position, normal, barycentric, motion, historyColour, historyLength
        static VkPipelineColorBlendAttachmentState blendAttachments[7];
        for (int i = 0; i < 7; i++) {
            blendAttachments[i].colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            blendAttachments[i].blendEnable = VK_FALSE;
        }
        pipelineConfig.colorBlendInfo.attachmentCount = 7;
        pipelineConfig.colorBlendInfo.pAttachments = blendAttachments;

        lvePipeline = std::make_unique<LvePipeline>(
            lveDevice,
            "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\NeonEngine\\shaders\\raytracing_rast.vert.spv",
            "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\NeonEngine\\shaders\\raytracing_rast.frag.spv",
            pipelineConfig
        );
    }

    void RayTracingRast::renderGameObjects(RayFrameInfo& frameInfo) {
        VkCommandBuffer cmd = frameInfo.commandBuffer;

        lvePipeline->bind(cmd);

        vkCmdBindDescriptorSets(
            cmd,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelineLayout,
            0, 1,
            &frameInfo.globalDescriptorSet,
            0, nullptr
        );

        for (auto& kv : frameInfo.gameObjects) {
            auto& obj = kv.second;
            if (!obj.model) continue;

            SimplePushConstantData push{};
            push.modelMatrix = obj.transform.mat4();
            push.normalMatrix = obj.transform.normalMatrix();

            vkCmdPushConstants(
                cmd,
                pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(SimplePushConstantData),
                &push
            );

            obj.model->bind(cmd);
            obj.model->draw(cmd);
        }
    }
    void RayTracingRast::createGBufferImages(LveDevice& device, GBufferRenderTargets& gBuffers) {
        VkFormat formats[7] = {
    VK_FORMAT_R32G32B32A32_SFLOAT, // position
    VK_FORMAT_R32G32B32A32_SFLOAT, // normal
    VK_FORMAT_R32G32B32A32_SFLOAT, // barycentric
    VK_FORMAT_R16G16_SFLOAT,       // motion
    VK_FORMAT_R16G16B16A16_SFLOAT, // historyColour
    VK_FORMAT_R16G16B16A16_SFLOAT, // historyLength
    VK_FORMAT_R16G16B16A16_SFLOAT, // historyColourImage2
        };

        VkImageUsageFlags usageFlags[7] = {
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, // position
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, // normal
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, // barycentric
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, // motion
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT, // historyColour
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT, // historyLength
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT, // historyColourImage2
        };

        VkImage* images[7] = {
            &gBuffers.positionImage,
            &gBuffers.normalImage,
            &gBuffers.barycentricImage,
            &gBuffers.motionImage,
            &gBuffers.historyColourImage,
            &gBuffers.historyLengthImage,
            &gBuffers.historyColourImage2,

        };

        VkImageView* views[7] = {
            &gBuffers.positionView,
            &gBuffers.normalView,
            &gBuffers.barycentricView,
            &gBuffers.motionView,
            &gBuffers.historyColorView,
            &gBuffers.historyLengthView,
            &gBuffers.historyColorView2,

        };

        for (int i = 0; i < 7; i++) {
            VkImageCreateInfo imageInfo{};
            imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.extent.width = swapChainExtents.width;
            imageInfo.extent.height = swapChainExtents.height;
            imageInfo.extent.depth = 1;
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.format = formats[i];
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imageInfo.usage = usageFlags[i];  // <-- was hardcoded
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VkDeviceMemory imageMemory;
            device.createImageWithInfo(imageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, *images[i], imageMemory);

            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = *images[i];
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = formats[i];
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.baseMipLevel = 0;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = 1;

            if (vkCreateImageView(device.device(), &viewInfo, nullptr, views[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create G-buffer image view!");
            }
        }

        // ---- Depth attachment ----
        VkImageCreateInfo depthInfo{};
        depthInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        depthInfo.imageType = VK_IMAGE_TYPE_2D;
        depthInfo.extent = { swapChainExtents.width, swapChainExtents.height, 1 };
        depthInfo.mipLevels = 1;
        depthInfo.arrayLayers = 1;
        depthInfo.format = VK_FORMAT_D32_SFLOAT;
        depthInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        depthInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        depthInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        depthInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkDeviceMemory depthMemory;
        device.createImageWithInfo(depthInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            gBuffers.depthImage, depthMemory);

        VkImageViewCreateInfo depthViewInfo{};
        depthViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        depthViewInfo.image = gBuffers.depthImage;
        depthViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        depthViewInfo.format = VK_FORMAT_D32_SFLOAT;
        depthViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        depthViewInfo.subresourceRange.baseMipLevel = 0;
        depthViewInfo.subresourceRange.levelCount = 1;
        depthViewInfo.subresourceRange.baseArrayLayer = 0;
        depthViewInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device.device(), &depthViewInfo, nullptr, &gBuffers.depthView) != VK_SUCCESS) {
            throw std::runtime_error("failed to create G-buffer depth image view!");
        }
    }
    void RayTracingRast::CreateSampler() {
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_NEAREST;
        samplerInfo.minFilter = VK_FILTER_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.maxAnisotropy = 1.0f;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

        if (vkCreateSampler(lveDevice.device(), &samplerInfo, nullptr, &gBufferSampler) != VK_SUCCESS) {
            throw std::runtime_error("failed to create gbuffer sampler");
        }
    }

    void RayTracingRast::CreateComputePipelineLayout(VkDescriptorSetLayout setLayout) {
        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &setLayout;

        if (vkCreatePipelineLayout(lveDevice.device(), &layoutInfo, nullptr, &computePipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline layout");
        }
    }

    void RayTracingRast::CreateComputePipeline() {
        computeShaderModule = LvePipeline::loadShaderModule(
            "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\NeonEngine\\shaders\\raytracing_pp.comp.spv",
            lveDevice.device());

        VkPipelineShaderStageCreateInfo stageInfo{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = computeShaderModule;
        stageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = computePipelineLayout;

        if (vkCreateComputePipelines(lveDevice.device(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline");
        }
    }


    void RayTracingRast::CreateAtrousPipelineLayout(VkDescriptorSetLayout setLayout) {
        // Push constant for the wavelet step level (iteration 0..N-1)
        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(uint32_t);   // int stepLevel

        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &setLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushRange;

        if (vkCreatePipelineLayout(lveDevice.device(), &layoutInfo, nullptr, &atrousPipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create ATrous pipeline layout");
        }
    }

    void RayTracingRast::CreateAtrousPipeline() {
        atrousShaderModule = LvePipeline::loadShaderModule(
            "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\NeonEngine\\shaders\\raytracing_atrous.comp.spv",
            lveDevice.device());

        VkPipelineShaderStageCreateInfo stageInfo{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = atrousShaderModule;
        stageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = atrousPipelineLayout;

        if (vkCreateComputePipelines(lveDevice.device(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &atrousPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create ATrous pipeline");
        }
    }



    bool pingPong = false;  
    void RayTracingRast::renderCompute(
        VkCommandBuffer commandBuffer,
        VkDescriptorSet descriptorSet,
        VkDescriptorSet descriptorSet2,
        uint32_t groupsX,
        uint32_t groupsY)
    {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        VkDescriptorSet currentSet = pingPong ? descriptorSet : descriptorSet2;

        pingPong = !pingPong;

        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            computePipelineLayout,
            0, 1,
            &currentSet,
            0, nullptr);



        vkCmdDispatch(commandBuffer, groupsX, groupsY, 1);
    }

    void RayTracingRast::renderAtrous(
        VkCommandBuffer  commandBuffer,
        VkDescriptorSet  setA,
        VkDescriptorSet  setB,
        uint32_t         groupsX,
        uint32_t         groupsY,
        uint32_t         numIterations)
    {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, atrousPipeline);

        for (uint32_t i = 0; i < numIterations; ++i) {
            uint32_t stepLevel = i;

            VkDescriptorSet currentSet = (i % 2 == 0) ? setA : setB;

            vkCmdBindDescriptorSets(commandBuffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                atrousPipelineLayout,
                0, 1, &currentSet,
                0, nullptr);

            vkCmdPushConstants(commandBuffer,
                atrousPipelineLayout,
                VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(uint32_t), &stepLevel);

            vkCmdDispatch(commandBuffer, groupsX, groupsY, 1);

            // Inter-iteration barrier
            VkMemoryBarrier memBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
            memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &memBarrier, 0, nullptr, 0, nullptr);
        }

        if (numIterations % 2 == 0) {
            uint32_t fixupStep = 0;
            vkCmdBindDescriptorSets(commandBuffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                atrousPipelineLayout,
                0, 1, &setB,
                0, nullptr);
            vkCmdPushConstants(commandBuffer,
                atrousPipelineLayout,
                VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(uint32_t), &fixupStep);
            vkCmdDispatch(commandBuffer, groupsX, groupsY, 1);

            // Final barrier after fixup
            VkMemoryBarrier memBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
            memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &memBarrier, 0, nullptr, 0, nullptr);
        }
    }

    void RayTracingRast::barrierStorageToCompute(VkCommandBuffer cmd, VkImage& storageImage) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = storageImage;
        barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );
    }
    void RayTracingRast::barrierGBufferToCompute(
        VkCommandBuffer cmd,
        GBufferRenderTargets& gBuffers)
    {
        VkImageMemoryBarrier barriers[7]{};

        VkImage images[7] = {
            gBuffers.positionImage,
            gBuffers.normalImage,
            gBuffers.barycentricImage,
            gBuffers.motionImage,
            gBuffers.historyColourImage,
            gBuffers.historyLengthImage,
            gBuffers.historyColourImage2,
        };

        for (int i = 0; i < 7; i++) {
            barriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].image = images[i];

            barriers[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barriers[i].subresourceRange.baseMipLevel = 0;
            barriers[i].subresourceRange.levelCount = 1;
            barriers[i].subresourceRange.baseArrayLayer = 0;
            barriers[i].subresourceRange.layerCount = 1;

            barriers[i].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        }

        for (int i = 0; i < 4; i++) {
            barriers[i].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barriers[i].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        }

        barriers[4].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barriers[4].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barriers[4].dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT;

        barriers[5].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barriers[5].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barriers[5].dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT;

        barriers[6].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barriers[6].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barriers[6].dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT;

        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            7, barriers
        );
    }
} // namespace lve