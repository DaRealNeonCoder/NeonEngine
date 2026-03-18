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
        .addPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, LveSwapChain::MAX_FRAMES_IN_FLIGHT * 3)
        .setMaxSets(LveSwapChain::MAX_FRAMES_IN_FLIGHT * 3)
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
        .addBinding(
            0,
            VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                VK_SHADER_STAGE_MISS_BIT_KHR)
        .addBinding(
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
        .addBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL)
        .addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL)
        .addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL)
        .addBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL)
        .build();

std::cout << "Pass 2 \n";
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
    rtVertex.normal = glm::vec4(glm::normalize(normalMatrix * vertex.normal), 1);
    vertices.push_back(rtVertex);
  }

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

  VkDescriptorImageInfo storageImageInfo = rayTracingSystem.getStorageImageDescriptor(i);
  VkAccelerationStructureKHR tlas = rayTracingSystem.getTLAS();

  VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
  asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
  asInfo.accelerationStructureCount = 1;
  asInfo.pAccelerationStructures = &tlas;

  LveDescriptorWriter writer(*rayTracingSetLayout, *globalPool);

  auto materialBufferInfo = rayTracingSystem.getMaterialBufferDescriptor();
  auto vertexBufferInfo = rayTracingSystem.getVertexBufferDescriptor();
  auto indexBufferInfo = rayTracingSystem.getIndexBufferDescriptor();

  writer.writeAccelerationStructure(0, &asInfo, globalDescriptorSets[i])
      .writeImage(1, &storageImageInfo)
      .writeBuffer(2, &uboInfo)
      .writeBuffer(3, &materialBufferInfo)
      .writeBuffer(4, &vertexBufferInfo)
      .writeBuffer(5, &indexBufferInfo)
      .build(globalDescriptorSets[i]);
}
std::cout << "Pass 4 \n";

LveCamera camera{};
auto viewerObject = LveGameObject::createGameObject();
viewerObject.transform.translation.z = -2.5f;
KeyboardMovementController cameraController{};

std::cout << "Pass 5 \n";

while (!lveWindow.shouldClose()) {
  glfwPollEvents();

  auto newTime = std::chrono::high_resolution_clock::now();
  float frameTime =
      std::chrono::duration<float, std::chrono::seconds::period>(newTime - currentTime).count();
  currentTime = newTime;

  fpsTimer += frameTime;
  frameCount++;

  if (fpsTimer >= 1.0f) {
    fps = frameCount / fpsTimer;

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

  float aspect = lveRenderer.getAspectRatio();
  camera.setPerspectiveProjection(glm::radians(50.f), aspect, 0.1f, 100.f);

  if (auto commandBuffer = lveRenderer.beginFrame([&]() {
        rayTracingSystem.handleResize(
            lveRenderer.getSwapChainExtents().width,
            lveRenderer.getSwapChainExtents().height,
            globalDescriptorSets);
        rayTracingSystem.resetFrameId();
      })) {
    int frameIndex = lveRenderer.getFrameIndex();

    FrameInfo frameInfo{
        frameIndex,
        frameTime,
        commandBuffer,
        camera,
        globalDescriptorSets[frameIndex],
        gameObjects};

    GlobalUbo ubo{};
    ubo.projection = camera.getProjection();
    ubo.inverseProjection = camera.getInverseProjection();
    ubo.view = camera.getView();
    ubo.inverseView = camera.getInverseView();
    ubo.numLights.x = 1;

    uboBuffers[frameIndex]->writeToBuffer(&ubo);
    uboBuffers[frameIndex]->flush();

    if (hasMoved) {
      rayTracingSystem.resetFrameId();
    }
    rayTracingSystem.render(frameInfo);

    rayTracingSystem.copyStorageImageToSwapChain(
        commandBuffer,
        lveRenderer.getCurrentSwapChainImage(),
        lveRenderer.getSwapChainExtents().width,
        lveRenderer.getSwapChainExtents().height,
        frameIndex);

    lveRenderer.endFrame([&]() {
      rayTracingSystem.handleResize(
          lveRenderer.getSwapChainExtents().width,
          lveRenderer.getSwapChainExtents().height,
          globalDescriptorSets);
      rayTracingSystem.resetFrameId();
    });
  }
}

vkDeviceWaitIdle(lveDevice.device());
}


void RayTracingApp::loadGameObjects() {
  /*
  std::shared_ptr<LveModel> lveModel = LveModel::createModelFromFile(
      lveDevice,
      "C:\\Users\\ZyBros\\Downloads\\littleVulkanEngine-tut27\\littleVulkanEngine-tut27\\models\\Dragon_lowPoly.obj");
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
      "C:\\Users\\ZyBros\\Downloads\\littleVulkanEngine-tut27\\littleVulkanEngine-"
      "tut27\\models\\Knight.obj");
  auto obj = LveGameObject::createGameObject();
  obj.model = lveModel;
  obj.transform.translation = {0.f, -0.55f, 0.f};
  obj.transform.rotation = {0.f, 0, 0.f};
  obj.transform.scale = {0.23f, 0.23f, 0.23f};
  gameObjects.emplace(obj.getId(), std::move(obj));
  */
  std::shared_ptr<LveModel> lveModel = LveModel::createModelFromFile(
      lveDevice,
      "C:\\Users\\ZyBros\\Downloads\\littleVulkanEngine-tut27\\littleVulkanEngine-"
      "tut27\\models\\car.obj");
  auto obj = LveGameObject::createGameObject();
  obj.model = lveModel;
  obj.transform.translation = {0.f, -0.55f, 0.f};
  obj.transform.rotation = {0.f, glm::radians(-40.f), 0.f};
  obj.transform.scale = {0.35f, 0.35f, 0.35f};
  gameObjects.emplace(obj.getId(), std::move(obj));
 lveModel = LveModel::createModelFromFile(
      lveDevice,
      "C:\\Users\\ZyBros\\Downloads\\littleVulkanEngine-tut27\\littleVulkanEngine-"
      "tut27\\models\\cube.obj");
  
    
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
