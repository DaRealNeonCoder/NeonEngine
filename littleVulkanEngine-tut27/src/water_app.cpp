#include "water_app.hpp"

#include "keyboard_movement_controller.hpp"
#include "lve_buffer.hpp"
#include "lve_camera.hpp"
#include "lve_texture.hpp"
#include "point_light_system.hpp"
#include "simple_render_system.hpp"
#include "water_physics.hpp"
#include "water_render_system.hpp"

// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <vulkan.h>

#include <glm.hpp>
#include <gtc/constants.hpp>
// std
#include <array>
#include <cassert>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
namespace lve {

WaterApp::WaterApp() {
  int numFrames = LveSwapChain::MAX_FRAMES_IN_FLIGHT;
  int totalSetsNeeded = 2 * numFrames;

  globalPool = LveDescriptorPool::Builder(lveDevice)
                   .setMaxSets(numFrames * 26)
                   // UBO: needed in both descriptors per frame => 2 * numFrames
                   .addPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, totalSetsNeeded)
                   // sampler: only in the 'withShadow' set per frame => numFrames
                   .addPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, numFrames * 4)
                   .addPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, numFrames * 18)
                   .build();

  loadGameObjects();
}
WaterApp::~WaterApp() {}
void WaterApp::run() {
  std::cout << "Pass 1 \n";

  auto currentTime = std::chrono::high_resolution_clock::now();
  char title[128];
  float fpsTimer = 0.0f;
  int frameCount = 0;
  float fps = 0.0f;
  std::vector<float> fpsHistory(31, 0.0f);  // fixed size
  int fpsIndex = 0;                         // circular write index

  std::vector<std::unique_ptr<LveBuffer>> uboBuffers(LveSwapChain::MAX_FRAMES_IN_FLIGHT);
  std::vector<std::unique_ptr<LveBuffer>> phyUboBuffers(LveSwapChain::MAX_FRAMES_IN_FLIGHT);

  int particleLen = 25;
  int particleCount = particleLen * particleLen * particleLen;

    std::vector<glm::vec4> posTemp(particleCount);
  colors.assign(particleCount, glm::vec3(0, 0.4f, 0.8f));  

  float particleScale = 0.2f;
  glm::vec4 offset(0, -2.f, 0, 0);
  float particleSpacing = 2.2f * particleScale;
  for (size_t x = 0; x < particleLen; x++) {
    for (size_t y = 0; y < particleLen; y++) {
      for (size_t z = 0; z < particleLen; z++) {
        posTemp[x + y * particleLen + z * particleLen * particleLen] =
            glm::vec4(x * particleSpacing, y * particleSpacing, z * particleSpacing, 0.1f);
      }
    }
  }
  KeyboardMovementController waterBoxController{};

  for (int i = 0; i < (int)uboBuffers.size(); i++) {
    uboBuffers[i] = std::make_unique<LveBuffer>(
        lveDevice,
        sizeof(WaterUbo),
        1,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    uboBuffers[i]->map();
  }

  for (int i = 0; i < (int)phyUboBuffers.size(); i++) {
    phyUboBuffers[i] = std::make_unique<LveBuffer>(
        lveDevice,
        sizeof(WaterPhysUbo),
        1,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    phyUboBuffers[i]->map();
  }
  auto globalSetLayout =
      LveDescriptorSetLayout::Builder(lveDevice)
          .addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL_GRAPHICS)  // UBO
          .addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL_GRAPHICS)  // pos
          .addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL_GRAPHICS)  // col
          .build();

  auto computeSetLayout =
      LveDescriptorSetLayout::Builder(lveDevice)
          .addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
          .addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
          .addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
          .addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
          .addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
          .addBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
          .addBinding(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
          .addBinding(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
          .addBinding(8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
          .addBinding(9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
          .addBinding(10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
          .build();

  std::cout << "Pass 2 \n";

  std::vector<VkDescriptorSet> globalDescriptorSets(LveSwapChain::MAX_FRAMES_IN_FLIGHT);
  std::vector<VkDescriptorSet> computeDescriptorSets(LveSwapChain::MAX_FRAMES_IN_FLIGHT);


  float scleThing = 1.2f;
  float smoothingRadius = 0.3f;
  float pressureMult = 300.f;

  WaterPhysUbo phyUbo{};

  phyUbo.uGridDim = glm::ivec4(20, 20, 20, 1);

  phyUbo.uCellSize = 0.3f;

  phyUbo.uH = smoothingRadius;
  phyUbo.uH2 = smoothingRadius * smoothingRadius;
  phyUbo.poly6Coeff = 315.0f / (64.0f * glm::pi<float>() * pow(smoothingRadius, 9));
  phyUbo.spikyGradCoeff = -45.f / (glm::pi<float>() * pow(smoothingRadius, 6));
  phyUbo.viscLapCoeff = 45.0f / (glm::pi<float>() * pow(smoothingRadius, 6));

  phyUbo.uMass = 8.0f * scleThing;
  phyUbo.uRestDensity = 1000.f;

  phyUbo.uMu = pressureMult;
  phyUbo.uViscosity = 0.10f;
  phyUbo.uEps = 0.01f * smoothingRadius * smoothingRadius;
  //phyUbo.uDt = 0.002f;
  //dt in push 
  phyUbo.uBoxMin = glm::vec4(-4.928f, -3.08f, -1.848f, 1);
  phyUbo.uBoxMax = glm::vec4(4.928f, 0.55f, 4.928f / 2.f, 1);
  phyUbo.uGravity = glm::vec4(0.f, 9.81f, 0.f, 0.f);

  phyUbo.uDamping = 0.6;
  phyUbo.uNumParticles = particleCount ;
  std::cout << waterParticles.size() << 
     "\n";
  phyUbo.uNumCells = phyUbo.uGridDim.x * phyUbo.uGridDim.y * phyUbo.uGridDim.z;

  for (size_t i = 0; i < phyUboBuffers.size(); i++) {
    phyUboBuffers[i]->writeToBuffer(&phyUbo);
    phyUboBuffers[i]->flush();
  }

  WaterPhysics waterPhysics{
      particleCount, 
      posTemp,
      computeSetLayout->getDescriptorSetLayout(),
      phyUbo,
      gameObjects,
      smoothingRadius,  
      1000.f,           // rest density (water)
      0.3f,             // viscosity
      1000.0f,          // pressure stiffness
      &lveDevice,
  };

  for (int i = 0; i < (int)computeDescriptorSets.size(); i++) {
    auto cellCountInfo = waterPhysics.getCellCountDescInfo();
    auto cellIndicesInfo = waterPhysics.getCellIndiciesDescInfo();
    auto cellStartInfo = waterPhysics.getCellStartDescInfo();
    auto velocitiesInfo = waterPhysics.getVelocitiesDescInfo();
    auto densitiesInfo = waterPhysics.getDensitiesDescInfo();
    auto pressuresInfo = waterPhysics.getPressuresDescInfo();
    auto forcesInfo = waterPhysics.getForcesDescInfo();
    auto uboInfo = phyUboBuffers[i]->descriptorInfo();
    auto outputInfo = waterPhysics.getOutputDescInfo();
    auto cellCursorInfo = waterPhysics.getCellCursorDescInfo();
    auto colorsInfo = waterPhysics.getColorDescInfo();

    LveDescriptorWriter(*computeSetLayout, *globalPool)
        .writeBuffer(0, &uboInfo)         // binding 1: output positions
        .writeBuffer(1, &outputInfo)      // binding 1: output positions
        .writeBuffer(2, &velocitiesInfo)  // binding 1: output positions
        .writeBuffer(3, &densitiesInfo)   // binding 1: output positions
        .writeBuffer(4, &pressuresInfo)   // binding 1: output positions
        .writeBuffer(5, &forcesInfo)      // binding 1: output positions
        .writeBuffer(6, &cellStartInfo)
        .writeBuffer(7, &cellCountInfo)    // binding 0: input positions
        .writeBuffer(8, &cellIndicesInfo)  // binding 0: input positions
        .writeBuffer(9, &cellCursorInfo)  // binding 0: input positions
        .writeBuffer(10, &colorsInfo)   // binding 0: input positions
        .build(computeDescriptorSets[i]);
  }
  std::cout << "Pass 4 \n";
 
  std::shared_ptr<LveModel> model;
  for (auto& kv : gameObjects) {
      auto& obj = kv.second;
      if (obj.model == nullptr) continue;
      model = obj.model;
    break;
  }
  std::cout << "Pass 4.1\n";
  WaterRenderSystem waterRenderSystem{
      lveDevice,
      lveRenderer.getSwapChainRenderPass(),
      globalSetLayout->getDescriptorSetLayout(),
      particleCount,
      model->getVertices()};
  waterRenderSystem.particleVert = model->getVertices().size();
  waterRenderSystem.updateBuffers(posTemp, colors);

  
  for (int i = 0; i < (int)globalDescriptorSets.size(); i++) {
    auto bufferInfo = uboBuffers[i]->descriptorInfo();
    auto bufferInfo2 = waterPhysics.getOutputDescInfo();
    auto bufferInfo3 = waterPhysics.getColorDescInfo();

    LveDescriptorWriter(*globalSetLayout, *globalPool)
        .writeBuffer(0, &bufferInfo)
        .writeBuffer(1, &bufferInfo2)
        .writeBuffer(2, &bufferInfo3)
        .build(globalDescriptorSets[i]);
  }
  std::cout << "Pass 4.2\n";

  // ... camera setup ...
  LveCamera camera{};
  auto viewerObject = LveGameObject::createGameObject();
  viewerObject.transform.translation.z = -2.5f;
  KeyboardMovementController cameraController{};

  std::cout << "Pass 5 \n";



  std::cout << "Pass 6 \n";
  std::this_thread::sleep_for(std::chrono::seconds(3));
  bool hasMoved;
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


    waterBoxController.editBoxDimensions(lveWindow.getGLFWwindow(), frameTime, boxDim, hasMoved);
    if (hasMoved) {
      phyUbo.uBoxMin = boxDim * glm::vec4(-4.928f, -3.08f, -1.848f, 1);
      phyUbo.uBoxMax = boxDim * glm::vec4(4.928f, 0.55f, 4.928f / 2.f, 1);
      phyUboBuffers[0]->writeToBuffer(&phyUbo);
      phyUboBuffers[0]->flush();

      hasMoved = false;
    }
    if (auto commandBuffer = lveRenderer.beginFrame()) {

      int frameIndex = lveRenderer.getFrameIndex();
      WaterFrameInfo frameInfo{
          frameTime,
          commandBuffer,
          camera,
          globalDescriptorSets[frameIndex],
          computeDescriptorSets[0],
          gameObjects};

      WaterUbo ubo{};
      ubo.projection = camera.getProjection();
      ubo.inverseProjection = camera.getProjection();
      ubo.view = camera.getView();
      ubo.inverseView = camera.getInverseView();

      uboBuffers[frameIndex]->writeToBuffer(&ubo);
      uboBuffers[frameIndex]->flush();

      waterPhysics.RunSimulation(0.002f, frameInfo);

      //waterRenderSystem.updateBuffers(waterPhysics.outPositions, colors);
      lveRenderer.beginSwapChainRenderPass(commandBuffer);

      waterRenderSystem.renderGameObjects(frameInfo, waterPhysics);

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
  flatVase.transform.scale = {2, 2, 2};
  flatVase.color = glm::vec3(0.8f);
  //gameObjects.emplace(flatVase.getId(), std::move(flatVase));

  std::shared_ptr<LveModel> cubeModel = LveModel::createModelFromFile(
      lveDevice,
      "C:\\Users\\ZyBros\\Downloads\\littleVulkanEngine-tut27\\littleVulkanEngine-"
      "tut27\\models\\quad.obj");
  int particleLen = 7;

  
  float spc = 0.1f;  // whatever you actually used when creating the lattice
  float particleVolume = spc * spc * spc;
  float particleMass = 1000.f * particleVolume;  // gives mass that yields rho ~ restDensity i
  glm::vec3 offset(0, -2.f, 0);
  float particleScale = 0.2f;

 LveGameObject particle = LveGameObject::createGameObject();
  particle.model = cubeModel;

  particle.transform.translation = {0, 0, 0};

  particle.transform.translation += offset;

  particle.transform.scale = {particleScale, particleScale, particleScale};
  particle.color = glm::vec3(0, 0.4f, 1.f);

  gameObjects.emplace(particle.getId(), std::move(particle));
}

}  // namespace lve