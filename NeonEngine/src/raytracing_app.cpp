#include "raytracing_app.hpp"

#include "keyboard_movement_controller.hpp"
#include "lve_buffer.hpp"
#include "lve_camera.hpp"
#include "lve_frame_info.hpp"
#include "raytracing_system.hpp"
#include "raytracing_rast.hpp"

// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm.hpp>
#include <gtc/constants.hpp>
#include <vulkan.h>
// std
#include <array>
#include <cassert>
#include <chrono>
#include <stdexcept>
#include <iostream>

namespace lve {

RayTracingApp::RayTracingApp() {
int numFrames = LveSwapChain::MAX_FRAMES_IN_FLIGHT;
int totalSetsNeeded = 2 * numFrames; // withShadow + noShadow per frame

globalPool =
    LveDescriptorPool::Builder(lveDevice)

        .addPoolSize(
                     VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                     LveSwapChain::MAX_FRAMES_IN_FLIGHT)
        .addPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, LveSwapChain::MAX_FRAMES_IN_FLIGHT * 2)
        .addPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, LveSwapChain::MAX_FRAMES_IN_FLIGHT)
        .addPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, LveSwapChain::MAX_FRAMES_IN_FLIGHT * 8)
        .addPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, LveSwapChain::MAX_FRAMES_IN_FLIGHT * 8)

        .setMaxSets(LveSwapChain::MAX_FRAMES_IN_FLIGHT * 36)
        .build();

loadGameObjects();
}
RayTracingApp::~RayTracingApp() {}




void RayTracingApp::run() {
    std::cout << "Pass 1 \n";

    auto currentTime = std::chrono::high_resolution_clock::now();
    char title[128];
    float fpsTimer = 0.0f;
    int frameCount = 0;
    float fps = 0.0f;
    std::vector<float> fpsHistory(31, 0.0f);
    int fpsIndex = 0;

    // ---- UBO buffers (shared by both raster and ray tracing passes) ----
    std::vector<std::unique_ptr<LveBuffer>> uboBuffers(LveSwapChain::MAX_FRAMES_IN_FLIGHT);
    for (int i = 0; i < (int)uboBuffers.size(); i++) {
        uboBuffers[i] = std::make_unique<LveBuffer>(
            lveDevice,
            sizeof(RayUbo),
            1,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        uboBuffers[i]->map();
    }

    // ---- Descriptor set layouts ----
    auto gBufferSetLayout =
        lve::LveDescriptorSetLayout::Builder(lveDevice)
        .addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
        .build();

    //for ze denoiser
    auto computeSetLayout =
        lve::LveDescriptorSetLayout::Builder(lveDevice)
        .addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT)    // position
        .addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT)   // normal
        .addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT)  // barycentric
        .addBinding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT) // motion (cuz u ain't got none)
        .addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_ALL)         // output
        .addBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_ALL)  // historyColour
        .addBinding(6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_ALL)  // historyLength
        .addBinding(7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_ALL)  // historyLength
        .build();

    //for ze denoiser
    auto computeSetLayout2 =
        lve::LveDescriptorSetLayout::Builder(lveDevice)
        .addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT)    // position
        .addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT)   // normal
        .addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT)  // barycentric
        .addBinding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT) // motion (cuz u ain't got none)
        .addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_ALL)         // output
        .addBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_ALL)  // historyColour
        .addBinding(6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_ALL)  // historyLength
        .addBinding(7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_ALL)  // historyLength
        .build();
    auto rayTracingSetLayout =
        lve::LveDescriptorSetLayout::Builder(lveDevice)
        .addBinding(0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
            VK_SHADER_STAGE_MISS_BIT_KHR)
        .addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
        .addBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL)
        .addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL)
        .addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL)
        .addBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL)
        .build();



    std::cout << "Pass 2 \n";

    // ---- Build vertex/index data for TLAS ----
    std::vector<RayTracingVertex> vertices;
    std::vector<uint32_t> indices;

    for (auto& kv : gameObjects) {
        auto& obj = kv.second;
        if (!obj.model) continue;

        uint32_t vertexOffset = static_cast<uint32_t>(vertices.size());
        const auto& modelVertices = obj.model->getVertices();
        const auto& modelIndices = obj.model->getIndices();

        for (const auto& vertex : modelVertices) {
            RayTracingVertex rtVertex;
            glm::vec4 transformedPos = obj.transform.mat4() * glm::vec4(vertex.position, 1.0f);
            rtVertex.pos = transformedPos;
            rtVertex.materialIndex = obj.getId();
            glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(obj.transform.mat4())));
            rtVertex.normal = glm::vec4(glm::normalize(normalMatrix * vertex.normal), 1.0f);
            vertices.push_back(rtVertex);
        }

        for (const auto& index : modelIndices) {
            indices.push_back(index + vertexOffset);
        }
    }
    // ---- Create G-buffer render pass (6 MRTs + depth) ----
    VkRenderPass gBufferRenderPass{ VK_NULL_HANDLE };

    {
        // 6 color + 1 depth = 7 total attachments
        VkAttachmentDescription allAttachments[8]{};

        VkFormat gBufferFormats[7] = {
            VK_FORMAT_R32G32B32A32_SFLOAT, // position
            VK_FORMAT_R32G32B32A32_SFLOAT, // normal
            VK_FORMAT_R32G32B32A32_SFLOAT, // barycentric
            VK_FORMAT_R16G16_SFLOAT,       // motion
            VK_FORMAT_R16G16B16A16_SFLOAT, // historyColour
            VK_FORMAT_R16G16B16A16_SFLOAT,          // historyLength
            VK_FORMAT_R16G16B16A16_SFLOAT, // historyColour
        };

        // Color attachments [0..5]
        for (int i = 0; i < 7; i++) {
            allAttachments[i].format = gBufferFormats[i];
            allAttachments[i].samples = VK_SAMPLE_COUNT_1_BIT;
            allAttachments[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            allAttachments[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            allAttachments[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            allAttachments[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            allAttachments[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            allAttachments[i].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        // historyColour and historyLength are storage images — need GENERAL layout
        allAttachments[4].finalLayout = VK_IMAGE_LAYOUT_GENERAL;
        allAttachments[5].finalLayout = VK_IMAGE_LAYOUT_GENERAL;
        allAttachments[6].finalLayout = VK_IMAGE_LAYOUT_GENERAL;


        allAttachments[4].initialLayout = VK_IMAGE_LAYOUT_GENERAL;
        allAttachments[5].initialLayout = VK_IMAGE_LAYOUT_GENERAL;
        allAttachments[6].initialLayout = VK_IMAGE_LAYOUT_GENERAL;

        allAttachments[4].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        allAttachments[5].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        allAttachments[6].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;

        // Depth attachment [6]
        allAttachments[7].format = VK_FORMAT_D32_SFLOAT;
        allAttachments[7].samples = VK_SAMPLE_COUNT_1_BIT;
        allAttachments[7].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        allAttachments[7].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        allAttachments[7].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        allAttachments[7].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        allAttachments[7].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        allAttachments[7].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorAttachmentRefs[7]{};
        for (int i = 0; i < 7; i++) {
            colorAttachmentRefs[i].attachment = i;
            colorAttachmentRefs[i].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }

        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 7;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 7;
        subpass.pColorAttachments = colorAttachmentRefs;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;

        VkSubpassDependency dependencies[3]{};

        dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass = 0;
        dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[0].srcAccessMask = 0;
        dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        dependencies[1].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[1].dstSubpass = 0;
        dependencies[1].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependencies[1].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependencies[1].srcAccessMask = 0;
        dependencies[1].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        dependencies[2].srcSubpass = 0;
        dependencies[2].dstSubpass = VK_SUBPASS_EXTERNAL;

        dependencies[2].srcStageMask =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        dependencies[2].dstStageMask =
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        dependencies[2].srcAccessMask =
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        dependencies[2].dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT;

        dependencies[2].dependencyFlags = 0;


        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 8;
        renderPassInfo.pAttachments = allAttachments;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 3;
        renderPassInfo.pDependencies = dependencies;

        if (vkCreateRenderPass(lveDevice.device(), &renderPassInfo, nullptr, &gBufferRenderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create G-buffer render pass!");
        }
    }
    // ---- Ray tracing raster system ----
    RayTracingRast rayTracingRast{
        lveDevice,
        gBufferRenderPass,
        gBufferSetLayout->getDescriptorSetLayout(),
        computeSetLayout->getDescriptorSetLayout(),
        computeSetLayout2->getDescriptorSetLayout(),
        lveRenderer.getSwapChainExtents() };

    // Create the 4 G-buffer images via your existing method
    GBufferRenderTargets gBuffers = rayTracingRast.targets;

       VkImageView attachments[8] = {
        gBuffers.positionView,
        gBuffers.normalView,
        gBuffers.barycentricView,
        gBuffers.motionView,
        gBuffers.historyColorView,   // ADD
        gBuffers.historyLengthView,  // ADD
        gBuffers.historyColorView2,   // ADD
        gBuffers.depthView,
    };





    VkFramebufferCreateInfo fbInfo{};
    fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbInfo.renderPass = gBufferRenderPass;
    fbInfo.attachmentCount = 8;
    fbInfo.pAttachments = attachments;
    fbInfo.width = lveRenderer.getSwapChainExtents().width;
    fbInfo.height = lveRenderer.getSwapChainExtents().height;
    fbInfo.layers = 1;

    VkFramebuffer gBufferFramebuffer;
    if (vkCreateFramebuffer(lveDevice.device(), &fbInfo, nullptr, &gBufferFramebuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create G-buffer framebuffer!");
    }

    // ---- Ray tracing system ----
    RayTracingSystem rayTracingSystem{
        lveDevice,
        lveRenderer.getSwapChainImageFormat(),
        rayTracingSetLayout->getDescriptorSetLayout(),
        vertices,
        indices,
        lveRenderer.getSwapChainExtents().width,
        lveRenderer.getSwapChainExtents().height };





    // ---- Descriptor sets ----
    std::vector<VkDescriptorSet> gBufferDescriptorSets(LveSwapChain::MAX_FRAMES_IN_FLIGHT);
    std::vector<VkDescriptorSet> globalDescriptorSets(LveSwapChain::MAX_FRAMES_IN_FLIGHT);
    std::vector<VkDescriptorSet> computeDescriptorSets(LveSwapChain::MAX_FRAMES_IN_FLIGHT);
    std::vector<VkDescriptorSet> computeDescriptorSets2(LveSwapChain::MAX_FRAMES_IN_FLIGHT);




    for (int i = 0; i < LveSwapChain::MAX_FRAMES_IN_FLIGHT; i++) {
        auto uboInfo = uboBuffers[i]->descriptorInfo();
        LveDescriptorWriter(*gBufferSetLayout, *globalPool)
            .writeBuffer(0, &uboInfo)
            .build(gBufferDescriptorSets[i]);

        auto storageImageInfo = rayTracingSystem.getStorageImageDescriptor(i);
        auto tlas = rayTracingSystem.getTLAS();

        VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
        asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        asInfo.accelerationStructureCount = 1;
        asInfo.pAccelerationStructures = &tlas;

        auto materialBufferInfo = rayTracingSystem.getMaterialBufferDescriptor();
        auto vertexBufferInfo = rayTracingSystem.getVertexBufferDescriptor();
        auto indexBufferInfo = rayTracingSystem.getIndexBufferDescriptor();

        LveDescriptorWriter(*rayTracingSetLayout, *globalPool)
            .writeAccelerationStructure(0, &asInfo, globalDescriptorSets[i])
            .writeImage(1, &storageImageInfo)
            .writeBuffer(2, &uboInfo)
            .writeBuffer(3, &materialBufferInfo)
            .writeBuffer(4, &vertexBufferInfo)
            .writeBuffer(5, &indexBufferInfo)
            .build(globalDescriptorSets[i]);



        VkDescriptorImageInfo posInfo{};
        posInfo.sampler = rayTracingRast.gBufferSampler;
        posInfo.imageView = gBuffers.positionView;
        posInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo normalInfo{};
        normalInfo.sampler = rayTracingRast.gBufferSampler;
        normalInfo.imageView = gBuffers.normalView;
        normalInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo baryInfo{};
        baryInfo.sampler = rayTracingRast.gBufferSampler;
        baryInfo.imageView = gBuffers.barycentricView;
        baryInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo motionInfo{};
        motionInfo.sampler = rayTracingRast.gBufferSampler;
        motionInfo.imageView = gBuffers.motionView;
        motionInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;


        VkDescriptorImageInfo historyColourInfo{};
        historyColourInfo.imageView = gBuffers.historyColorView;
        historyColourInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo historyLengthInfo{};
        historyLengthInfo.imageView = gBuffers.historyLengthView;
        historyLengthInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        //why is color and colour mixed?
        VkDescriptorImageInfo historyColourInfo2{};
        historyColourInfo2.imageView = gBuffers.historyColorView2;
        historyColourInfo2.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo outputInfo = rayTracingSystem.getStorageImageDescriptor(i);
        // outputInfo.imageLayout should be VK_IMAGE_LAYOUT_GENERAL

        LveDescriptorWriter(*computeSetLayout, *globalPool)
            .writeImage(0, &posInfo)
            .writeImage(1, &normalInfo)
            .writeImage(2, &baryInfo)
            .writeImage(3, &motionInfo)
            .writeImage(4, &outputInfo)
            .writeImage(5, &historyColourInfo)
            .writeImage(6, &historyLengthInfo)
            .writeImage(7, &historyColourInfo2)

            .build(computeDescriptorSets[i]);

        LveDescriptorWriter(*computeSetLayout2, *globalPool)
            .writeImage(0, &posInfo)
            .writeImage(1, &normalInfo)
            .writeImage(2, &baryInfo)
            .writeImage(3, &motionInfo)
            .writeImage(4, &outputInfo)
            .writeImage(5, &historyColourInfo2)
            .writeImage(6, &historyLengthInfo)
            .writeImage(7, &historyColourInfo)

            .build(computeDescriptorSets2[i]);
    }



    auto recreateComputeDescriptors = [&]() {
        for (int i = 0; i < LveSwapChain::MAX_FRAMES_IN_FLIGHT; i++) {


            VkDescriptorImageInfo posInfo{};
            posInfo.sampler = rayTracingRast.gBufferSampler;
            posInfo.imageView = gBuffers.positionView;
            posInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo normalInfo{};
            normalInfo.sampler = rayTracingRast.gBufferSampler;
            normalInfo.imageView = gBuffers.normalView;
            normalInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo baryInfo{};
            baryInfo.sampler = rayTracingRast.gBufferSampler;
            baryInfo.imageView = gBuffers.barycentricView;
            baryInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo motionInfo{};
            motionInfo.sampler = rayTracingRast.gBufferSampler;
            motionInfo.imageView = gBuffers.motionView;
            motionInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;


            VkDescriptorImageInfo historyColourInfo{};
            historyColourInfo.imageView = gBuffers.historyColorView;
            historyColourInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkDescriptorImageInfo historyLengthInfo{};
            historyLengthInfo.imageView = gBuffers.historyLengthView;
            historyLengthInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkDescriptorImageInfo historyColourInfo2{};
            historyColourInfo2.imageView = gBuffers.historyColorView2;
            historyColourInfo2.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkDescriptorImageInfo outputInfo = rayTracingSystem.getStorageImageDescriptor(i);
            // outputInfo.imageLayout should be VK_IMAGE_LAYOUT_GENERAL

            LveDescriptorWriter(*computeSetLayout, *globalPool)
                .writeImage(0, &posInfo)
                .writeImage(1, &normalInfo)
                .writeImage(2, &baryInfo)
                .writeImage(3, &motionInfo)
                .writeImage(4, &outputInfo)
                .writeImage(5, &historyColourInfo)
                .writeImage(6, &historyLengthInfo)
                .writeImage(7, &historyColourInfo2)

                .build(computeDescriptorSets[i]);


            LveDescriptorWriter(*computeSetLayout, *globalPool)
                .writeImage(0, &posInfo)
                .writeImage(1, &normalInfo)
                .writeImage(2, &baryInfo)
                .writeImage(3, &motionInfo)
                .writeImage(4, &outputInfo)
                .writeImage(5, &historyColourInfo2)
                .writeImage(6, &historyLengthInfo)
                .writeImage(7, &historyColourInfo)

                .build(computeDescriptorSets2[i]);
        }
    };
    
    // ---- Main loop ----
    LveCamera camera{};
    auto viewerObject = LveGameObject::createGameObject();
    viewerObject.transform.translation = glm::vec3(0.0444f, -0.7041f, -1.1978f);
    KeyboardMovementController cameraController{};

    glm::mat4 prevView(0);
    bool Trip = false;
    while (!lveWindow.shouldClose()) {
        glfwPollEvents();

        auto newTime = std::chrono::high_resolution_clock::now();
        float frameTime =
            std::chrono::duration<float, std::chrono::seconds::period>(newTime - currentTime).count();
        currentTime = newTime;

        fpsTimer += frameTime;
        frameCount++;

        if (fpsTimer >= 1.0f) {
            float fps = frameCount / fpsTimer;
            fpsHistory[fpsIndex] = glm::round(fps * 10.f) / 10.f;
            fpsIndex = (fpsIndex + 1) % fpsHistory.size();
            snprintf(title, sizeof(title), "Ray Tracer | FPS: %.1f (%.2f ms)", fps, 1000.0f / fps);
            glfwSetWindowTitle(lveWindow.getGLFWwindow(), title);
            fpsTimer = 0.0f;
            frameCount = 0;
        }

        bool hasMoved = false;
        cameraController.moveInPlaneXZ(lveWindow.getGLFWwindow(), frameTime, viewerObject, hasMoved);
        camera.setViewYXZ(viewerObject.transform.translation, viewerObject.transform.rotation);
        camera.setPerspectiveProjection(glm::radians(50.f), lveRenderer.getAspectRatio(), 0.1f, 100.f);



        if (auto commandBuffer = lveRenderer.beginFrame([&]() {
            rayTracingSystem.handleResize(lveRenderer.getSwapChainExtents().width,
            lveRenderer.getSwapChainExtents().height,
            globalDescriptorSets);
            rayTracingSystem.resetFrameId(); 
            recreateGBuffer(gBufferRenderPass, gBuffers, gBufferFramebuffer, rayTracingRast, lveRenderer.getSwapChainExtents());
            Trip = false;
            recreateComputeDescriptors();
            })) {
            int frameIndex = lveRenderer.getFrameIndex();

            // Update UBO
            RayUbo ubo{};
            ubo.projection = camera.getProjection();
            ubo.inverseProjection = camera.getInverseProjection();
            ubo.view = camera.getView();
            ubo.inverseView = camera.getInverseView();

            ubo.prevView = prevView;

            prevView = ubo.view;

            uboBuffers[frameIndex]->writeToBuffer(&ubo);
            uboBuffers[frameIndex]->flush();

            if (hasMoved) rayTracingSystem.resetFrameId();
            if (!Trip) {
                Trip = true;
                transitionHistoryToGeneral(commandBuffer, gBuffers.historyColourImage);
                transitionHistoryToGeneral(commandBuffer, gBuffers.historyLengthImage);
                transitionHistoryToGeneral(commandBuffer, gBuffers.historyColourImage2);
            }
            // ---- G-buffer raster pass ----
        lveRenderer.beginRenderPass(commandBuffer, gBufferRenderPass, gBufferFramebuffer, lveRenderer.getSwapChainExtents());
        {
            RayFrameInfo gBufferFrameInfo{
                frameIndex, frameTime, commandBuffer, camera,
                gBufferDescriptorSets[frameIndex], gameObjects };
            rayTracingRast.renderGameObjects(gBufferFrameInfo);
        }
        lveRenderer.endRenderPass(commandBuffer);
        // GBuffer images are now in SHADER_READ_ONLY_OPTIMAL (via renderpass finalLayout)

        // ---- Ray tracing pass ----
        {
            FrameInfo frameInfo{
                frameIndex, frameTime, commandBuffer, camera,
                globalDescriptorSets[frameIndex], gameObjects };
            rayTracingSystem.render(frameInfo);
        }
        // ---- Barrier: GBuffer color attachments -> compute shader read ----
        rayTracingRast.barrierGBufferToCompute(commandBuffer, gBuffers);

        // ---- Barrier: storageImage ray tracing write -> compute shader read/write ----
        rayTracingRast.barrierStorageToCompute(commandBuffer, rayTracingSystem.getStorageImage());


        // ---- Compute pass: Temporal Accumulation ----
        rayTracingRast.renderCompute(
            commandBuffer,
            computeDescriptorSets[frameIndex],
            computeDescriptorSets2[frameIndex],
            (lveRenderer.getSwapChainExtents().width + 15) / 16,
            (lveRenderer.getSwapChainExtents().height + 15) / 16);

        // ---- Barrier: temporal wrote storageImage, ATrous will read it ----
        {
            VkMemoryBarrier memBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
            memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &memBarrier, 0, nullptr, 0, nullptr);
        }

        // ---- ATrous Wavelet pass ----
        // 5 iterations. The temporal pass left storageImage as the "current" output.
        // pingPong is already flipped by renderCompute, so we know which set was
        // last used — the OPPOSITE set is the one whose binding 4 = storageImage.
        // We just feed them in the same alternating order so ATrous continues the chain.
        constexpr uint32_t kAtrousIterations = 5;
        rayTracingRast.renderAtrous(
            commandBuffer,
            computeDescriptorSets[frameIndex],   // setA — binding 4 = storageImage
            computeDescriptorSets2[frameIndex],  // setB — binding 4 = historyColour (scratch)
            (lveRenderer.getSwapChainExtents().width + 15) / 16,
            (lveRenderer.getSwapChainExtents().height + 15) / 16,
            kAtrousIterations);

        // ---- Barrier: ATrous done, storageImage ready for copy ----
        {
            VkMemoryBarrier memBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
            memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            memBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 1, &memBarrier, 0, nullptr, 0, nullptr);
        }





        // ---- Copy storageImage to swapchain ----
        rayTracingSystem.copyStorageImageToSwapChain(
            commandBuffer,
            lveRenderer.getCurrentSwapChainImage(),
            lveRenderer.getSwapChainExtents().width,
            lveRenderer.getSwapChainExtents().height,
            frameIndex);
                    lveRenderer.endFrame([&]() {
                        rayTracingSystem.handleResize(lveRenderer.getSwapChainExtents().width,
                        lveRenderer.getSwapChainExtents().height,
                        globalDescriptorSets);
                    rayTracingSystem.resetFrameId();
                    recreateGBuffer(gBufferRenderPass, gBuffers, gBufferFramebuffer, rayTracingRast, lveRenderer.getSwapChainExtents());
                    recreateComputeDescriptors();
                    Trip = false;

                        });
                }
            }

    vkDeviceWaitIdle(lveDevice.device());

    if (gBufferFramebuffer != VK_NULL_HANDLE) vkDestroyFramebuffer(lveDevice.device(), gBufferFramebuffer, nullptr);
    if (gBufferRenderPass != VK_NULL_HANDLE) vkDestroyRenderPass(lveDevice.device(), gBufferRenderPass, nullptr);

    for (size_t i = 0; i < fpsHistory.size(); i++) std::cout << fpsHistory[i] << ", ";
    std::cout << viewerObject.transform.translation.x << " "
        << viewerObject.transform.translation.y << " "
        << viewerObject.transform.translation.z << std::endl;
}



void RayTracingApp::recreateGBuffer(
    VkRenderPass gBufferRenderPass,
    GBufferRenderTargets& gBuffers,
    VkFramebuffer& frameBuffer,
    RayTracingRast& rayTracingRast,
    VkExtent2D extents
) {
    std::cout << "REC CALLED" << std::endl << std::endl;
    // Destroy old framebuffer
    if (frameBuffer != VK_NULL_HANDLE) {
        vkDestroyFramebuffer(lveDevice.device(), frameBuffer, nullptr);
        frameBuffer = VK_NULL_HANDLE;
    }

    // Add the two new views + depth
    vkDestroyImageView(lveDevice.device(), gBuffers.positionView, nullptr);
    vkDestroyImageView(lveDevice.device(), gBuffers.normalView, nullptr);
    vkDestroyImageView(lveDevice.device(), gBuffers.barycentricView, nullptr);
    vkDestroyImageView(lveDevice.device(), gBuffers.motionView, nullptr);
    vkDestroyImageView(lveDevice.device(), gBuffers.historyColorView, nullptr);   
    vkDestroyImageView(lveDevice.device(), gBuffers.historyLengthView, nullptr);  
    vkDestroyImageView(lveDevice.device(), gBuffers.historyColorView2, nullptr);  
    vkDestroyImageView(lveDevice.device(), gBuffers.depthView, nullptr);          

    rayTracingRast.swapChainExtents = extents;
    rayTracingRast.createGBufferImages(lveDevice, gBuffers);

    VkImageView attachments[8] = {
        gBuffers.positionView,
        gBuffers.normalView,
        gBuffers.barycentricView,
        gBuffers.motionView,
        gBuffers.historyColorView,   // ADD
        gBuffers.historyLengthView,  // ADD
        gBuffers.historyColorView2,   // ADD
        gBuffers.depthView,          // ADD
    };

    VkFramebufferCreateInfo fbInfo{};
    fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbInfo.renderPass = gBufferRenderPass;
    fbInfo.attachmentCount = 8;  // was 5
    fbInfo.pAttachments = attachments;
    fbInfo.width = extents.width;
    fbInfo.height = extents.height;
    fbInfo.layers = 1;

    if (vkCreateFramebuffer(
        lveDevice.device(),
        &fbInfo,
        nullptr,
        &frameBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create G-buffer framebuffer!");
    }
}

void RayTracingApp::transitionHistoryToGeneral(VkCommandBuffer cmd, VkImage image) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;

    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;

    barrier.srcAccessMask = 0;
    barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_SHADER_WRITE_BIT;

    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    barrier.image = image;
    barrier.subresourceRange = {
        VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1
    };

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );
}

void RayTracingApp::loadGameObjects() {
  /*
  std::shared_ptr<LveModel> lveModel = LveModel::createModelFromFile(
      lveDevice,
      "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\NeonEngine\\models\\Dragon_lowPoly.obj");
  auto obj = LveGameObject::createGameObject();
  obj.model = lveModel;
  obj.transform.translation = {0.07f, -0.4f, 0.f};
  obj.transform.rotation = {0.f, -70.f + 180, 0.f};
  obj.transform.scale = {1, 1, 1};
  gameObjects.emplace(obj.getId(), std::move(obj));
  */
  /*
  std::shared_ptr<LveModel> lveModel = LveModel::createModelFromFile(
      lveDevice,
      "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\NeonEngine\\models\\Knight.obj");
  auto obj = LveGameObject::createGameObject();
  obj.model = lveModel;
  obj.transform.translation = {0.f, -0.55f, 0.f};
  obj.transform.rotation = {0.f, 0, 0.f};
  obj.transform.scale = {0.23f, 0.23f, 0.23f};
  gameObjects.emplace(obj.getId(), std::move(obj));
  */
  std::shared_ptr<LveModel> lveModel = LveModel::createModelFromFile(
      lveDevice,
      "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\NeonEngine\\models\\car.obj");
  auto obj = LveGameObject::createGameObject();
  obj.model = lveModel;
  obj.transform.translation = {0.f, -0.557f, 0.f};
  obj.transform.rotation = {0.f, glm::radians(-40.f), 0.f};
  obj.transform.scale = { 0.35f, 0.35f , 0.35f };
  gameObjects.emplace(obj.getId(), std::move(obj));
 lveModel = LveModel::createModelFromFile(
      lveDevice,
      "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\NeonEngine\\models\\cube.obj");
  
    
    //ceiling
    auto cube = LveGameObject::createGameObject();
  cube.model = lveModel;
  cube.transform.translation = {0.f, -1.5f, 0.f};
  cube.transform.scale = {0.58f, 0.04f, 0.5f};
  gameObjects.emplace(cube.getId(), std::move(cube));
  //floor
  cube = LveGameObject::createGameObject();
  cube.model = lveModel;
  cube.transform.translation = {0.f, -0.5f, 0.f};
  cube.transform.scale = {0.58f, 0.04f, 0.5f};
  gameObjects.emplace(cube.getId(), std::move(cube));
  //right wall
  cube = LveGameObject::createGameObject();
  cube.model = lveModel;
  cube.transform.translation = {0.54f, -1.f, 0.f};
  cube.transform.scale = {0.04f, 0.46f, 0.5f};
  gameObjects.emplace(cube.getId(), std::move(cube));
  //left wall
    cube = LveGameObject::createGameObject();
  cube.model = lveModel;
  cube.transform.translation = {-0.54f, -1.f, 0.f};
  cube.transform.scale = {0.04f, 0.46f, 0.5f};
  gameObjects.emplace(cube.getId(), std::move(cube));

  //backwall
  cube = LveGameObject::createGameObject();
  cube.model = lveModel;
  cube.transform.translation = {0.f, -1.f, 0.5f};
  cube.transform.scale = {0.5f, 0.5f, 0.04f};
  gameObjects.emplace(cube.getId(), std::move(cube));

}

}  // namespace lve
