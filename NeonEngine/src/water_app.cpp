#include "water_app.hpp"

#include "keyboard_movement_controller.hpp"
#include "lve_buffer.hpp"
#include "lve_camera.hpp"
#include "lve_texture.hpp"
#include "point_light_system.hpp"
#include "simple_render_system.hpp"
#include "water_physics.hpp"
#include "water_render_system.hpp"
#include <random>

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
                   .setMaxSets(numFrames * 28)
                   // UBO: needed in both descriptors per frame => 2 * numFrames
                   .addPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, totalSetsNeeded)
                   // sampler: only in the 'withShadow' set per frame => numFrames
                   .addPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, numFrames * 4)
                   .addPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, numFrames * 20)
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
  std::vector<float> fpsHistory(31, 0.0f);
  int fpsIndex = 0;

  std::vector<std::unique_ptr<LveBuffer>> uboBuffers(LveSwapChain::MAX_FRAMES_IN_FLIGHT);
  std::vector<std::unique_ptr<LveBuffer>> phyUboBuffers(LveSwapChain::MAX_FRAMES_IN_FLIGHT);

  int actualPartLen = 25;
  int particleLen = actualPartLen;
  int particleCount = particleLen * particleLen * particleLen;

  std::vector<glm::vec4> posTemp(particleCount);
  colors.assign(particleCount, glm::vec3(0, 0.4f, 0.8f));

  float particleScale = 0.2f;
  float particleSpacing = 2.2f * particleScale;

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
          .addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL_GRAPHICS)
          .addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL_GRAPHICS)
          .addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL_GRAPHICS)
          .build();
  
  auto computeSetLayout =
    LveDescriptorSetLayout::Builder(lveDevice)
        .addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // partPos
        .addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // partVel
        .addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // gridU
        .addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // gridV
        .addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // gridW
        .addBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // prevGridU
        .addBinding(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // prevGridV
        .addBinding(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // prevGridW
        .addBinding(8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // gridFlags
        .addBinding(9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // colors
        .addBinding(10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // UBO
        .addBinding(11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // debug
        .addBinding(12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // gridDU
        .addBinding(13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // gridDV
        .addBinding(14, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // gridDW
        .addBinding(15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // gridS
        .addBinding(16, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // pressureRead
        .addBinding(17, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // pressureWrite
        .addBinding(18, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // gridS
        .addBinding(19, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // pressureRead
        .addBinding(20, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // pressureWrite
        .addBinding(21, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // gridS
        .addBinding(22, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // pressureRead
        .addBinding(23, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)  // pressureWrite
        .build();

  std::cout << "Pass 2 \n";

  std::vector<VkDescriptorSet> globalDescriptorSets(LveSwapChain::MAX_FRAMES_IN_FLIGHT);
  std::vector<VkDescriptorSet> computeDescriptorSetsPong(LveSwapChain::MAX_FRAMES_IN_FLIGHT);
  std::vector<VkDescriptorSet> computeDescriptorSetsPing(LveSwapChain::MAX_FRAMES_IN_FLIGHT);
  float scleThing = 1.2f;
  float smoothingRadius = 0.1f;
  float pressureMult = 0.1f;

  WaterPhysUbo phyUbo{};

  // ============================================================
  // DEBUG: Print initial configuration
  // ============================================================
  std::cout << "\n========================================" << std::endl;
  std::cout << "INITIAL CONFIGURATION" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Particle count: " << particleCount << std::endl;
  std::cout << "Smoothing radius: " << smoothingRadius << std::endl;
  std::cout << "========================================\n" << std::endl;

  // Define simulation box FIRST
  phyUbo.uBoxMin = glm::vec4(-4.928f, -3.08f, -1.848f, 1);
  phyUbo.uBoxMax = glm::vec4(4.928f, 0.55f, 4.928f / 2.f, 1);

  // ============================================================
  // DEBUG: Box dimensions
  // ============================================================
  glm::vec3 boxSize = glm::vec3(phyUbo.uBoxMax) - glm::vec3(phyUbo.uBoxMin);
  std::cout << "========================================" << std::endl;
  std::cout << "SIMULATION BOX" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Box Min: (" << phyUbo.uBoxMin.x << ", " << phyUbo.uBoxMin.y << ", "
            << phyUbo.uBoxMin.z << ")" << std::endl;
  std::cout << "Box Max: (" << phyUbo.uBoxMax.x << ", " << phyUbo.uBoxMax.y << ", "
            << phyUbo.uBoxMax.z << ")" << std::endl;
  std::cout << "Box Size: " << boxSize.x << " x " << boxSize.y << " x " << boxSize.z << " units"
            << std::endl;
  std::cout << "Box Volume: " << (boxSize.x * boxSize.y * boxSize.z) << " cubic units" << std::endl;
  std::cout << "========================================\n" << std::endl;

  // ============================================================
  // FIXED: AUTO-CALCULATE GRID DIMENSIONS
  // ============================================================
  float desiredCellSize = 0.1f;

  // Calculate grid dimensions to fully cover the box
  int gridX = (int)std::ceil(boxSize.x / desiredCellSize);
  int gridY = (int)std::ceil(boxSize.y / desiredCellSize);
  int gridZ = (int)std::ceil(boxSize.z / desiredCellSize);

  // Ensure minimum 2 cells per dimension (needed for wall boundaries)
  gridX = std::max(gridX, 2);
  gridY = std::max(gridY, 2);
  gridZ = std::max(gridZ, 2);

  phyUbo.uGridDim = glm::ivec4(gridX, gridY, gridZ, 1);
  phyUbo.uCellSize = desiredCellSize;

  // ============================================================
  // DEBUG: Grid configuration and coverage check
  // ============================================================
  float gridCoverageX = phyUbo.uGridDim.x * phyUbo.uCellSize;
  float gridCoverageY = phyUbo.uGridDim.y * phyUbo.uCellSize;
  float gridCoverageZ = phyUbo.uGridDim.z * phyUbo.uCellSize;

  std::cout << "========================================" << std::endl;
  std::cout << "GRID CONFIGURATION" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Grid Dimensions: " << phyUbo.uGridDim.x << " x " << phyUbo.uGridDim.y << " x "
            << phyUbo.uGridDim.z << " cells" << std::endl;
  std::cout << "Cell Size: " << phyUbo.uCellSize << " units" << std::endl;
  std::cout << "Grid Coverage: " << gridCoverageX << " x " << gridCoverageY << " x "
            << gridCoverageZ << " units" << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "COVERAGE CHECK:" << std::endl;

  bool coverageOK = true;
  if (gridCoverageX < boxSize.x) {
    std::cout << "  X-axis: Grid covers " << gridCoverageX << " but box needs " << boxSize.x
              << " (SHORT by " << (boxSize.x - gridCoverageX) << ")" << std::endl;
    coverageOK = false;
  } else {
    std::cout << "  X-axis: Grid covers " << gridCoverageX << " >= box " << boxSize.x << std::endl;
  }

  if (gridCoverageY < boxSize.y) {
    std::cout << "  Y-axis: Grid covers " << gridCoverageY << " but box needs " << boxSize.y
              << " (SHORT by " << (boxSize.y - gridCoverageY) << ")" << std::endl;
    coverageOK = false;
  } else {
    std::cout << "  Y-axis: Grid covers " << gridCoverageY << " >= box " << boxSize.y << std::endl;
  }

  if (gridCoverageZ < boxSize.z) {
    std::cout << "  Z-axis: Grid covers " << gridCoverageZ << " but box needs " << boxSize.z
              << " (SHORT by " << (boxSize.z - gridCoverageZ) << ")" << std::endl;
    coverageOK = false;
  } else {
    std::cout << "  Z-axis: Grid covers " << gridCoverageZ << " >= box " << boxSize.z << std::endl;
  }

  std::cout << "----------------------------------------" << std::endl;
  if (!coverageOK) {
    std::cout << " WARNING: GRID DOES NOT FULLY COVER BOX!" << std::endl;
    std::cout << "Particles outside grid will FREEZE!" << std::endl;
  } else {
    std::cout << "Grid fully covers the simulation box" << std::endl;
  }
  std::cout << "========================================\n" << std::endl;

  // Rest of UBO setup
  smoothingRadius = 0.3f;
  phyUbo.uH = smoothingRadius;
  phyUbo.uH2 = smoothingRadius * smoothingRadius;
  phyUbo.overRelaxation = 1.01f;  // Match JS (was 1.0)
  phyUbo.spikyGradCoeff = -45.f / (glm::pi<float>() * pow(smoothingRadius, 6));
  phyUbo.viscLapCoeff = 45.0f / (glm::pi<float>() * pow(smoothingRadius, 6));
  phyUbo.uRestDensity = 1000000.f;  // Much lower (was 1000.0) - JS uses ~1.0 for normalized units
  //phyUbo.uViscosity = 0.10f;
  phyUbo.uEps = 0.01f * smoothingRadius * smoothingRadius;
  phyUbo.uGravity = glm::vec4(0.f, 9.81f, 0.f, 0.f);  // Negative Y for downward gravity
  phyUbo.uDamping = 0.95f;
  phyUbo.uNumParticles = particleCount;
  std::cout << waterParticles.size() << "\n";
  phyUbo.uNumCells = phyUbo.uGridDim.x * phyUbo.uGridDim.y * phyUbo.uGridDim.z;

  // ============================================================
  // Spawn particles
  // ============================================================
  // Before the spawn loop:
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> jitter(-0.5f, 0.5f);
  float jitterScale = particleSpacing * 0.9f;  // ~30% of spacing
 
  {
    glm::vec3 boxMin = glm::vec3(phyUbo.uBoxMin) * 0.97f;
    glm::vec3 boxMax = glm::vec3(phyUbo.uBoxMax) * 0.97f;
    float margin = (phyUbo.uCellSize > 0.0f) ? phyUbo.uCellSize : (particleScale * 0.5f);
    glm::vec3 spawnMin = boxMin + glm::vec3(margin);
    glm::vec3 spawnMax = boxMax - glm::vec3(margin);

    if (spawnMax.x <= spawnMin.x || spawnMax.y <= spawnMin.y || spawnMax.z <= spawnMin.z) {
      glm::vec3 center = 0.5f * (boxMin + boxMax);
      float fallbackExtent = glm::min(glm::max(boxMax.x - boxMin.x, 0.1f), 1.0f);
      spawnMin = center - glm::vec3(fallbackExtent * 0.25f);
      spawnMax = center + glm::vec3(fallbackExtent * 0.25f);
    }

    glm::vec3 spawnSize = spawnMax - spawnMin;

    for (size_t x = 0; x < (size_t)particleLen; x++) {
      for (size_t y = 0; y < (size_t)particleLen; y++) {
        for (size_t z = 0; z < (size_t)particleLen; z++) {
          float eps = 0.001f;

          float nx =
              (particleLen == 1) ? 0.5f : (float(x) + eps) / (float(particleLen - 1) + 2.0f * eps);

          float ny =
              (particleLen == 1) ? 0.5f : (float(y) + eps) / (float(particleLen - 1) + 2.0f * eps);

          float nz =
              (particleLen == 1) ? 0.5f : (float(z) + eps) / (float(particleLen - 1) + 2.0f * eps);
          glm::vec3 pos = spawnMin + glm::vec3(nx, ny, nz) * spawnSize;
          size_t idx = x + y * particleLen + z * particleLen * particleLen;


          pos += glm::vec3(
              jitter(rng) * jitterScale,
              jitter(rng) * jitterScale,
              jitter(rng) * jitterScale);


          posTemp[idx] = glm::vec4(pos, 0.1f);
        }
      }
    }

    // ============================================================
    // DEBUG: Particle spawn verification
    // ============================================================
    std::cout << "========================================" << std::endl;
    std::cout << "PARTICLE SPAWN" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Spawn region Min: (" << spawnMin.x << ", " << spawnMin.y << ", " << spawnMin.z
              << ")" << std::endl;
    std::cout << "Spawn region Max: (" << spawnMax.x << ", " << spawnMax.y << ", " << spawnMax.z
              << ")" << std::endl;
    std::cout << "Spawn region size: " << spawnSize.x << " x " << spawnSize.y << " x "
              << spawnSize.z << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "Sample particle positions:" << std::endl;
    for (int i = 0; i < std::min(5, particleCount); i++) {
      std::cout << "  Particle " << i << ": (" << posTemp[i].x << ", " << posTemp[i].y << ", "
                << posTemp[i].z << ")" << std::endl;
    }
    std::cout << "  ..." << std::endl;
    for (int i = std::max(0, particleCount - 5); i < particleCount; i++) {
      std::cout << "  Particle " << i << ": (" << posTemp[i].x << ", " << posTemp[i].y << ", "
                << posTemp[i].z << ")" << std::endl;
    }

    int particlesOutsideBox = 0;
    int particlesOutsideGrid = 0;
    glm::vec3 gridMax =
        glm::vec3(phyUbo.uBoxMin) + glm::vec3(gridCoverageX, gridCoverageY, gridCoverageZ);

    for (int i = 0; i < particleCount; i++) {
      glm::vec3 pos = glm::vec3(posTemp[i]);

      if (pos.x < boxMin.x || pos.x > boxMax.x || pos.y < boxMin.y || pos.y > boxMax.y ||
          pos.z < boxMin.z || pos.z > boxMax.z) {
        particlesOutsideBox++;
      }

      if (pos.x < boxMin.x || pos.x > gridMax.x || pos.y < boxMin.y || pos.y > gridMax.y ||
          pos.z < boxMin.z || pos.z > gridMax.z) {
        particlesOutsideGrid++;
      }
    }

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Particles outside box: " << particlesOutsideBox << " / " << particleCount
              << std::endl;
    std::cout << "Particles outside grid coverage: " << particlesOutsideGrid << " / "
              << particleCount << std::endl;

    if (particlesOutsideBox > 0) {
      std::cout << "  WARNING: " << particlesOutsideBox << " particles spawned outside the box!"
                << std::endl;
    }
    if (particlesOutsideGrid > 0) {
      std::cout << "  WARNING: " << particlesOutsideGrid
                << " particles spawned outside grid coverage!" << std::endl;
      std::cout << "These particles will likely FREEZE!" << std::endl;
    }
    std::cout << "========================================\n" << std::endl;
  }

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
      1.f,
      0.3f,
      1.0f,
      &lveDevice,
  };
  std::cout << "passed smth \n" << std::endl;

  // Descriptor Set Writers
  for (int i = 0; i < (int)computeDescriptorSetsPing.size(); i++) {
    auto uboInfo = phyUboBuffers[i]->descriptorInfo();
    auto partPosInfo = waterPhysics.getPartPosDescInfo();
    auto partVelInfo = waterPhysics.getPartVelDescInfo();
    auto gridUInfo = waterPhysics.getGridUDescInfo();
    auto gridVInfo = waterPhysics.getGridVDescInfo();
    auto gridWInfo = waterPhysics.getGridWDescInfo();
    auto prevGridUInfo = waterPhysics.getPrevGridUDescInfo();
    auto prevGridVInfo = waterPhysics.getPrevGridVDescInfo();
    auto prevGridWInfo = waterPhysics.getPrevGridWDescInfo();
    auto gridFlagsInfo = waterPhysics.getGridFlagsDescInfo();
    auto colorWriteInfo = waterPhysics.getColorDescInfo();
    auto debugInfo = waterPhysics.getDebugInfo();
    auto gridDUInfo = waterPhysics.getGridDUDescInfo();
    auto gridDVInfo = waterPhysics.getGridDVDescInfo();
    auto gridDWInfo = waterPhysics.getGridDWDescInfo();
    auto gridSInfo = waterPhysics.getGridSDescInfo();
    auto pressureReadInfo = waterPhysics.getPressureReadDescInfo();
    auto pressureWriteInfo = waterPhysics.getPressureWriteDescInfo();
    auto gridUAccumInfo = waterPhysics.getGridUAccumDescInfo();
    auto gridVAccumInfo = waterPhysics.getGridVAccumDescInfo();
    auto gridWAccumInfo = waterPhysics.getGridWAccumDescInfo();

    auto gridUWeightInfo = waterPhysics.getGridUWeightDescInfo();
    auto gridVWeightInfo = waterPhysics.getGridVWeightDescInfo();
    auto gridWWeightInfo = waterPhysics.getGridWWeightDescInfo();

    // PING descriptor set
    LveDescriptorWriter(*computeSetLayout, *globalPool)
        .writeBuffer(0, &partPosInfo)
        .writeBuffer(1, &partVelInfo)
        .writeBuffer(2, &gridUInfo)
        .writeBuffer(3, &gridVInfo)
        .writeBuffer(4, &gridWInfo)
        .writeBuffer(5, &prevGridUInfo)
        .writeBuffer(6, &prevGridVInfo)
        .writeBuffer(7, &prevGridWInfo)
        .writeBuffer(8, &gridFlagsInfo)
        .writeBuffer(9, &colorWriteInfo)
        .writeBuffer(10, &uboInfo)
        .writeBuffer(11, &debugInfo)
        .writeBuffer(12, &gridDUInfo)
        .writeBuffer(13, &gridDVInfo)
        .writeBuffer(14, &gridDWInfo)
        .writeBuffer(15, &gridSInfo)
        .writeBuffer(16, &pressureReadInfo)
        .writeBuffer(17, &pressureWriteInfo)
        .writeBuffer(18, &gridUAccumInfo)
        .writeBuffer(19, &gridVAccumInfo)
        .writeBuffer(20, &gridWAccumInfo)
        .writeBuffer(21, &gridUWeightInfo)
        .writeBuffer(22, &gridVWeightInfo)
        .writeBuffer(23, &gridWWeightInfo)
        .build(computeDescriptorSetsPing[i]);

    // PONG descriptor set (swap pressure buffers)
    LveDescriptorWriter(*computeSetLayout, *globalPool)
        .writeBuffer(0, &partPosInfo)
        .writeBuffer(1, &partVelInfo)
        .writeBuffer(2, &gridUInfo)
        .writeBuffer(3, &gridVInfo)
        .writeBuffer(4, &gridWInfo)
        .writeBuffer(5, &prevGridUInfo)
        .writeBuffer(6, &prevGridVInfo)
        .writeBuffer(7, &prevGridWInfo)
        .writeBuffer(8, &gridFlagsInfo)
        .writeBuffer(9, &colorWriteInfo)
        .writeBuffer(10, &uboInfo)
        .writeBuffer(11, &debugInfo)
        .writeBuffer(12, &gridDUInfo)
        .writeBuffer(13, &gridDVInfo)
        .writeBuffer(14, &gridDWInfo)
        .writeBuffer(15, &gridSInfo)
        .writeBuffer(16, &pressureWriteInfo)  // Swapped
        .writeBuffer(17, &pressureReadInfo)   // Swapped
        .writeBuffer(18, &gridUAccumInfo)
        .writeBuffer(19, &gridVAccumInfo)
        .writeBuffer(20, &gridWAccumInfo)
        .writeBuffer(21, &gridUWeightInfo)
        .writeBuffer(22, &gridVWeightInfo)
        .writeBuffer(23, &gridWWeightInfo)
        .build(computeDescriptorSetsPong[i]);
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
    auto bufferInfo2 = waterPhysics.getPartPosDescInfo();
    auto bufferInfo3 = waterPhysics.getColorDescInfo();

    LveDescriptorWriter(*globalSetLayout, *globalPool)
        .writeBuffer(0, &bufferInfo)
        .writeBuffer(1, &bufferInfo2)
        .writeBuffer(2, &bufferInfo3)
        .build(globalDescriptorSets[i]);
  }

  std::cout << "Pass 4.2\n";

  LveCamera camera{};
  auto viewerObject = LveGameObject::createGameObject();
  viewerObject.transform.translation.z = -2.5f;
  KeyboardMovementController cameraController{};

  std::cout << "Pass 5 \n";
  std::cout << "Pass 6 \n";

  bool hasMoved = false;

  std::cout << "\n========================================" << std::endl;
  std::cout << "SETUP COMPLETE - ENTERING MAIN LOOP" << std::endl;
  std::cout << "========================================\n" << std::endl;

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
          computeDescriptorSetsPing[0],
          computeDescriptorSetsPong[0],
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

      // NOW it's safe to read back (every N frames to avoid stalling)
      if (frameCount % 1000 == 0) {  // Only readback occasionally
        vkDeviceWaitIdle(lveDevice.device());

        float* data = (float*)waterPhysics.debugBuff->getMappedMemory();
        for (int i = 0; i < 10; ++i) {
            
            if (data[i] != -1) {
            std::cout << "div for " << i << " div:" << data[i] << std::endl;
          }
        }

      }
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

  
  float spc = 0.1f;  // we
  float particleVolume = spc * spc * spc;
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