#include "water_app.hpp"

#include "keyboard_movement_controller.hpp"
#include "lve_buffer.hpp"
#include "lve_camera.hpp"
#include "lve_texture.hpp"
#include "water_physics.hpp"
#include "water_render_system.hpp"
#include "point_light_system.hpp"
#include "simple_render_system.hpp"

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
#include <thread>
namespace lve {

WaterApp::WaterApp() {
int numFrames = LveSwapChain::MAX_FRAMES_IN_FLIGHT;
int totalSetsNeeded = 2 * numFrames; // withShadow + noShadow per frame

globalPool =
    LveDescriptorPool::Builder(lveDevice)
        .setMaxSets(totalSetsNeeded)
        // UBO: needed in both descriptors per frame => 2 * numFrames
        .addPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, totalSetsNeeded)
        // sampler: only in the 'withShadow' set per frame => numFrames
        .addPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, numFrames * 4)
        .build();

loadGameObjects();
}
WaterApp::~WaterApp() {}
void WaterApp::run() {
  std::cout << "Pass 1 \n";
  std::vector<std::unique_ptr<LveBuffer>> uboBuffers(LveSwapChain::MAX_FRAMES_IN_FLIGHT);

  for (int i = 0; i < (int)uboBuffers.size(); i++) {
    uboBuffers[i] = std::make_unique<LveBuffer>(
        lveDevice,
        sizeof(WaterUbo),
        1,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    uboBuffers[i]->map();
  }

  auto globalSetLayout =
      LveDescriptorSetLayout::Builder(lveDevice)
          .addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL_GRAPHICS)  // UBO
          .build();


  std::cout << "Pass 2 \n";

  std::vector<VkDescriptorSet> globalDescriptorSets(LveSwapChain::MAX_FRAMES_IN_FLIGHT);

  for (int i = 0; i < (int)globalDescriptorSets.size(); i++) {
    auto bufferInfo = uboBuffers[i]->descriptorInfo();

    // write the full descriptor for main pass: UBO + shadow image

    LveDescriptorWriter(*globalSetLayout, *globalPool)
        .writeBuffer(0, &bufferInfo)
        .build(globalDescriptorSets[i]);
  }
  std::cout << "Pass 4 \n";

  WaterRenderSystem waterRenderSystem{
      lveDevice,
      lveRenderer.getSwapChainRenderPass(),
      globalSetLayout->getDescriptorSetLayout()};
  
   WaterPhysics waterPhysics{
      gameObjects,
      0.6f,    // h = 2 × particle radius
      1000.f,  // rest density (water)
      0.f,    // viscosity
      6000.0f  // pressure stiffness
  };
  // ... camera setup ...
  LveCamera camera{};
  auto viewerObject = LveGameObject::createGameObject();
  viewerObject.transform.translation.z = -2.5f;
  KeyboardMovementController cameraController{};

  auto currentTime = std::chrono::high_resolution_clock::now();
  std::cout << "Pass 5 \n";
  char title[128];
  float fpsTimer = 0.0f;
  int frameCount = 0;
  float fps = 0.0f;
  std::vector<float> fpsHistory(31, 0.0f);  // fixed size
  int fpsIndex = 0;                         // circular write index

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
    

    fpsTimer += frameTime;
    frameCount++;

    if (fpsTimer >= 1.0f) {
      fps = frameCount / fpsTimer;

      fpsHistory[fpsIndex] = glm::round(fps * 10.f) / 10.f;
      fpsIndex = (fpsIndex + 1) % fpsHistory.size();

      snprintf(title, sizeof(title), "idk man | FPS: %.1f (%.2f ms)", fps, 1000.0f / fps);
      glfwSetWindowTitle(lveWindow.getGLFWwindow(), title);

      fpsTimer = 0.0f;
      frameCount = 0;
    }

    if (auto commandBuffer = lveRenderer.beginFrame()) {
      int frameIndex = lveRenderer.getFrameIndex();
      FrameInfo frameInfo{
          frameIndex,
          frameTime,
          commandBuffer,
          camera,
          globalDescriptorSets[frameIndex],
          gameObjects};

      // ---- UPDATE GLOBAL UBO ----
      WaterUbo ubo{};
      ubo.projection = camera.getProjection();
      ubo.inverseProjection = camera.getProjection();
      ubo.view = camera.getView();
      ubo.inverseView = camera.getInverseView();

      uboBuffers[frameIndex]->writeToBuffer(&ubo);
      uboBuffers[frameIndex]->flush();

      waterPhysics.RunSimulation(0.01f);
      
      lveRenderer.beginSwapChainRenderPass(commandBuffer);
      
      waterRenderSystem.renderGameObjects(frameInfo);
      
      lveRenderer.endSwapChainRenderPass(commandBuffer);

      lveRenderer.endFrame();
    }
  }

 std::cout << std::endl << std::endl;
 
 for (size_t i = 0; i < fpsHistory.size(); i++) {
    std::cout << fpsHistory[i] << ", ";
  }
  vkDeviceWaitIdle(lveDevice.device());
}

void WaterApp::loadGameObjects() {
  
    std::shared_ptr<LveModel> lveModel = LveModel::createModelFromFile(
      lveDevice,
      "C:\\Users\\ZyBros\\Downloads\\littleVulkanEngine-tut27\\littleVulkanEngine-"
      "tut27\\models\\WaterTank.obj");

    auto flatVase = LveGameObject::createGameObject();
    flatVase.model = lveModel;
    flatVase.transform.translation = {-3.1f, -1.3f, 0};
    flatVase.transform.scale = {2,2,2};
    flatVase.color = glm::vec3(0.8f);
    gameObjects.emplace(flatVase.getId(), std::move(flatVase));

   std::shared_ptr<LveModel> cubeModel = LveModel::createModelFromFile(
        lveDevice,
        "C:\\Users\\ZyBros\\Downloads\\littleVulkanEngine-tut27\\littleVulkanEngine-"
        "tut27\\models\\WaterParticle.obj");
    int particleLen = 7;
    float particleScale = 0.1f;
    float particleSpacing = 2.2f * particleScale;

    float spc = 0.002f;  // whatever you actually used when creating the lattice
    float particleVolume = spc * spc * spc;
    float particleMass =
        1000.f * particleVolume;  // gives mass that yields rho ~ restDensity i
    glm::vec3 offset(0,-2.f,0);


    for (size_t x = 0; x < particleLen; x++) {
        for (size_t y = 0; y < particleLen; y++) {
            for (size_t z = 0; z < particleLen; z++) {
      
              LveGameObject particle = LveGameObject::createGameObject();
              particle.model = cubeModel;
              
              particle.transform.translation = {
                  x * particleSpacing,
                  y * particleSpacing,
                  z * particleSpacing};

              particle.transform.translation += offset;

              particle.transform.scale = {particleScale, particleScale, particleScale};
              particle.color = glm::vec3(0,0.4f,1.f);

              gameObjects.emplace(particle.getId(), std::move(particle));
            }
        }
      }
}

}  // namespace lve
