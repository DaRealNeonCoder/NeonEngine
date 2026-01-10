#include "raytracing_app.hpp"

#include "keyboard_movement_controller.hpp"
#include "lve_buffer.hpp"
#include "lve_camera.hpp"
#include "lve_frame_info.hpp"
#include "raytracing_system.hpp"

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
        .addPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, LveSwapChain::MAX_FRAMES_IN_FLIGHT)
        .addPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, LveSwapChain::MAX_FRAMES_IN_FLIGHT)
        .setMaxSets(LveSwapChain::MAX_FRAMES_IN_FLIGHT)
        .build();

loadGameObjects();
}
RayTracingApp::~RayTracingApp() {}
void RayTracingApp::run() {
std::cout << "Pass 1 \n";
std::vector<std::unique_ptr<LveBuffer>> uboBuffers(LveSwapChain::MAX_FRAMES_IN_FLIGHT);

for (int i = 0; i < (int)uboBuffers.size(); i++) {
  uboBuffers[i] = std::make_unique<LveBuffer>(
      lveDevice,
      sizeof(GlobalUbo),
      1,
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  uboBuffers[i]->map();
}

auto rayTracingSetLayout =
    lve::LveDescriptorSetLayout::Builder(lveDevice)
        // TLAS
        .addBinding(
            0,
            VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                VK_SHADER_STAGE_MISS_BIT_KHR)

        .addBinding(
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)

        // Uniform buffer
        .addBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL)

        .build();

std::cout << "Pass 2 \n";
std::vector<RayTracingVertex> vertices;
std::vector<uint32_t> indices;

// Iterate through all game objects and collect their geometry
for (auto& kv : gameObjects) {
  auto& obj = kv.second;

  // Skip objects without models
  if (!obj.model) continue;

  // Get the current offset for indices (before adding new vertices)
  uint32_t vertexOffset = static_cast<uint32_t>(vertices.size());

  // Get vertices and indices from the model
  const auto& modelVertices = obj.model->getVertices();
  const auto& modelIndices = obj.model->getIndices();

  // Transform vertices by the game object's transform matrix
  for (const auto& vertex : modelVertices) {
    RayTracingVertex rtVertex;

    // Transform position by object's model matrix
    glm::vec4 transformedPos = obj.transform.mat4() * glm::vec4(vertex.position, 1.0f);
    rtVertex.pos = glm::vec3(transformedPos);

    /*
    // Transform normal (use inverse transpose for normals)
    glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(obj.transform.mat4())));
    rtVertex.normal = glm::normalize(normalMatrix * vertex.normal);

    // Copy other attributes (adjust based on your RayTracingVertex structure)
    rtVertex.color = vertex.color;
    rtVertex.uv = vertex.uv;
    */

    vertices.push_back(rtVertex);
  }

  // Add indices with offset applied
  for (const auto& index : modelIndices) {
    indices.push_back(index + vertexOffset);
  }
}
RayTracingSystem rayTracingSystem{
    lveDevice,
    lveRenderer.getSwapChainImageFormat(),
    rayTracingSetLayout->getDescriptorSetLayout(),
    vertices,
    indices,
    lveRenderer.getSwapChainExtents().width,
    lveRenderer.getSwapChainExtents().height};

std::vector<VkDescriptorSet> globalDescriptorSets(LveSwapChain::MAX_FRAMES_IN_FLIGHT);
for (int i = 0; i < LveSwapChain::MAX_FRAMES_IN_FLIGHT; i++) {
  auto uboInfo = uboBuffers[i]->descriptorInfo();

  // FIXED: Get the storage image descriptor for THIS frame
  VkDescriptorImageInfo storageImageInfo = rayTracingSystem.getStorageImageDescriptor(i);

  VkAccelerationStructureKHR tlas = rayTracingSystem.getTLAS();

  // --- Acceleration structure write ---
  VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
  asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
  asInfo.accelerationStructureCount = 1;
  asInfo.pAccelerationStructures = &tlas;

  LveDescriptorWriter writer(*rayTracingSetLayout, *globalPool);

  writer.writeAccelerationStructure(0, &asInfo, globalDescriptorSets[i])
      .writeImage(1, &storageImageInfo)
      .writeBuffer(2, &uboInfo)
      .build(globalDescriptorSets[i]);
}
std::cout << "Pass 4 \n";

// ... camera setup ...
LveCamera camera{};
auto viewerObject = LveGameObject::createGameObject();
viewerObject.transform.translation.z = -2.5f;
KeyboardMovementController cameraController{};

auto currentTime = std::chrono::high_resolution_clock::now();
std::cout << "Pass 5 \n";

while (!lveWindow.shouldClose()) {
  glfwPollEvents();

  auto newTime = std::chrono::high_resolution_clock::now();
  float frameTime =
      std::chrono::duration<float, std::chrono::seconds::period>(newTime - currentTime).count();
  currentTime = newTime;

  cameraController.moveInPlaneXZ(lveWindow.getGLFWwindow(), frameTime, viewerObject);
  camera.setViewYXZ(viewerObject.transform.translation, viewerObject.transform.rotation);

  float aspect = lveRenderer.getAspectRatio();
  camera.setPerspectiveProjection(glm::radians(50.f), aspect, 0.1f, 100.f);

  
  if (auto commandBuffer = lveRenderer.beginFrame()) {
    
int frameIndex = lveRenderer.getFrameIndex();
    
    FrameInfo frameInfo{
        frameIndex,
        frameTime,
        commandBuffer,
        camera,
        globalDescriptorSets[frameIndex],
        gameObjects};

    // Update UBO
    GlobalUbo ubo{};
    ubo.projection = camera.getProjection();
    ubo.view = camera.getView();
    ubo.inverseView = camera.getInverseView();
    ubo.numLights.x = 1;

    uboBuffers[frameIndex]->writeToBuffer(&ubo);
    uboBuffers[frameIndex]->flush();
    std::cout << "  UBO updated\n";

    std::cout << "  Calling rayTracingSystem.render()...\n";
    rayTracingSystem.render(frameInfo);
    std::cout << "  rayTracingSystem.render() completed\n";

    std::cout << "  Copying to swap chain...\n";
    // FIXED: Pass frameIndex to copyStorageImageToSwapChain

    rayTracingSystem.copyStorageImageToSwapChain(
        commandBuffer,
        lveRenderer.getCurrentSwapChainImage(),
        lveRenderer.getSwapChainExtents().width,
        lveRenderer.getSwapChainExtents().height,
        frameIndex);  // ADDED frameIndex parameter

    std::cout << "  Calling endFrame()...\n";
    lveRenderer.endFrame();

    // Check device status after submit

    std::cout << "=== Frame completed ===\n";
  } 
  
  else {

      std::cout << std::endl << "\nhere we go again\n" << std::endl;
    rayTracingSystem.handleResize(
        lveRenderer.getSwapChainExtents().width,
        lveRenderer.getSwapChainExtents().height, globalDescriptorSets);
  
  }
}

  //should do this via deconstructors, but whatever.

  vkDeviceWaitIdle(lveDevice.device());
}

void RayTracingApp::loadGameObjects() {
  std::shared_ptr<LveModel> lveModel = LveModel::createModelFromFile(
      lveDevice,
      "C:\\Users\\ZyBros\\Downloads\\littleVulkanEngine-tut27\\littleVulkanEngine-tut27\\models\\car.obj");
  auto flatVase = LveGameObject::createGameObject();
  flatVase.model = lveModel;
  flatVase.transform.translation = {0.f, -0.25f, 0.f};
  flatVase.transform.scale = {1, 1, 1};
  gameObjects.emplace(flatVase.getId(), std::move(flatVase));


  /*
  std::vector<glm::vec3> lightColors{
      {1.f, 1.f, 1.f}
      //{.1f, .1f, 1.f},
      //{.1f, 1.f, .1f},
      //{1.f, 1.f, .1f},
      //{.1f, 1.f, 1.f},
      //{1.f, 1.f, 1.f}  //
  };

  for (int i = 0; i < lightColors.size(); i++) {
    auto pointLight = LveGameObject::makePointLight(0.2f);
    pointLight.color = lightColors[i];
    auto rotateLight = glm::rotate(
        glm::mat4(1.f),
        (i * glm::two_pi<float>()) / lightColors.size(),
        {0.f, -1.f, 0.f});
    pointLight.transform.translation = glm::vec3(rotateLight * glm::vec4(-1.f, -1.f, -1.f, 1.f));
    gameObjects.emplace(pointLight.getId(), std::move(pointLight));
  }
  */
}

}  // namespace lve
