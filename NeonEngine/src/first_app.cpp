#include "first_app.hpp"

#include "keyboard_movement_controller.hpp"
#include "lve_buffer.hpp"
#include "lve_camera.hpp"
#include "lve_texture.hpp"
#include "shadow_system.hpp"
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

FirstApp::FirstApp() {
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
FirstApp::~FirstApp() {}
void FirstApp::run() {
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

  auto globalSetLayoutWithShadow =
      LveDescriptorSetLayout::Builder(lveDevice)
          .addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL_GRAPHICS)  // UBO
          .addBinding(
              1,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_SHADER_STAGE_FRAGMENT_BIT)  // shadow map (sampled)
          .addBinding(
              2,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_SHADER_STAGE_FRAGMENT_BIT)  // texture map (sampled)
          .build();

  // Layout used by shadow pass: only UBO (binding 0). No binding 1!
  auto globalSetLayoutNoShadow =
      LveDescriptorSetLayout::Builder(lveDevice)
          .addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT)  // UBO only
          .build();

  std::cout << "Pass 2 \n";
  ShadowSystem shadowSystem{lveDevice, globalSetLayoutNoShadow->getDescriptorSetLayout()};

  LveTexture text = {};
  LveTexture::AllocatedImage img = text.memoryStuff("C:\\Users\\ZyBros\\Downloads\\texture.png", lveDevice);

  std::vector<VkDescriptorSet> globalDescriptorSetsWithShadow(LveSwapChain::MAX_FRAMES_IN_FLIGHT);
  std::vector<VkDescriptorSet> globalDescriptorSetsNoShadow(LveSwapChain::MAX_FRAMES_IN_FLIGHT);

  for (int i = 0; i < (int)globalDescriptorSetsWithShadow.size(); i++) {
    auto bufferInfo = uboBuffers[i]->descriptorInfo();

    // write the full descriptor for main pass: UBO + shadow image
    VkDescriptorImageInfo shadowImageInfo = shadowSystem.getShadowMapDescriptor(0);
    VkDescriptorImageInfo textureImageInfo = text.getDescriptor(img);

    LveDescriptorWriter(*globalSetLayoutWithShadow, *globalPool)
        .writeBuffer(0, &bufferInfo)
        .writeImage(1, &shadowImageInfo)
        .writeImage(2, &textureImageInfo)
        .build(globalDescriptorSetsWithShadow[i]);

    // write descriptor for shadow pass: only UBO bound to the no-shadow layout
    LveDescriptorWriter(*globalSetLayoutNoShadow, *globalPool)
        .writeBuffer(0, &bufferInfo)
        .build(globalDescriptorSetsNoShadow[i]);
  }
  std::cout << "Pass 4 \n";
  std::cout << "Pass 4 \n";

  // create render systems as before (they'll use global descriptor set binding 0+1)
  SimpleRenderSystem simpleRenderSystem{
      lveDevice,
      lveRenderer.getSwapChainRenderPass(),
      globalSetLayoutWithShadow->getDescriptorSetLayout()};

  PointLightSystem pointLightSystem{
      lveDevice,
      lveRenderer.getSwapChainRenderPass(),
      globalSetLayoutWithShadow->getDescriptorSetLayout()};

  // ... camera setup ...
  LveCamera camera{};
  auto viewerObject = LveGameObject::createGameObject();
  viewerObject.transform.translation.z = -2.5f;
  KeyboardMovementController cameraController{};

  auto currentTime = std::chrono::high_resolution_clock::now();
  std::cout << "Pass 5 \n";
  std::this_thread::sleep_for(std::chrono::seconds(3));

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
          globalDescriptorSetsWithShadow[frameIndex],
          gameObjects};

      // ---- UPDATE GLOBAL UBO ----
      GlobalUbo ubo{};
      ubo.projection = camera.getProjection();
      ubo.view = camera.getView();
      ubo.inverseView = camera.getInverseView();
      ubo.numLights.x = 1;
      /// SHADOW PASS: render using descriptor set WITHOUT the shadow image
      FrameInfo shadowFrameInfo = frameInfo;
      shadowFrameInfo.globalDescriptorSet = globalDescriptorSetsNoShadow[frameIndex];

      shadowSystem.update(shadowFrameInfo, ubo);
      pointLightSystem.update(frameInfo, ubo);
      uboBuffers[frameIndex]->writeToBuffer(&ubo);
      uboBuffers[frameIndex]->flush();


      shadowSystem.render(shadowFrameInfo);  
      // MAIN SCENE PASS: use original frameInfo (with shadow sampler bound)
      lveRenderer.beginSwapChainRenderPass(commandBuffer);
      simpleRenderSystem.renderGameObjects(frameInfo);
      pointLightSystem.render(frameInfo);
      lveRenderer.endSwapChainRenderPass(commandBuffer);

      lveRenderer.endFrame();
    }
  }
  //should do this via deconstructors, but whatever.

  text.destroyImage(lveDevice, img);
  shadowSystem.destroy();
  vkDeviceWaitIdle(lveDevice.device());
}

void FirstApp::loadGameObjects() {
  
    std::shared_ptr<LveModel> lveModel = LveModel::createModelFromFile(
      lveDevice,
      "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\NeonEngine\\models\\cube.obj");

    auto flatVase = LveGameObject::createGameObject();
    flatVase.model = lveModel;
    flatVase.transform.translation = {0.3f, -0.25f, 0.f};
    flatVase.transform.scale = {0.07f, 0.07f, 0.07f};
    gameObjects.emplace(flatVase.getId(), std::move(flatVase));
  lveModel = LveModel::createModelFromFile(
      lveDevice,
      "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\NeonEngine\\models\\smooth_vase.obj");
  auto smoothVase = LveGameObject::createGameObject();
  smoothVase.model = lveModel;
  smoothVase.transform.translation = {-3.f, -3.f, -3.f};
  smoothVase.transform.scale = {1.0f, 1.0f, 1.0f};

  lveModel = LveModel::createModelFromFile(
      lveDevice,
      "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\NeonEngine\\models\\quad.obj");
  auto floor = LveGameObject::createGameObject();
  floor.model = lveModel;
  floor.transform.translation = {0.f, 0.f, 0.f};
  floor.transform.scale = {1.f, 1.f, 1.f};
  gameObjects.emplace(floor.getId(), std::move(floor));

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
}

}  // namespace lve
