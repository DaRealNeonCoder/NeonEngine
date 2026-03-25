#include "water_physics.hpp"

#include "lve_pipeline.hpp"

// glm
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm.hpp>
#include <gtc/constants.hpp>

// std
#include <cmath>
#include <iostream>

#include "lve_descriptors.hpp"

namespace lve {

glm::vec3 boxMin(-1.4f, -1.4f, -1.4f);
glm::vec3 boxMax(1.4f, 0.5f, 1.4f);

WaterPhysics::WaterPhysics(
    int _particleCount,
    std::vector<glm::vec4>& startPos,
    VkDescriptorSetLayout setLayout,
    WaterPhysUbo& ubo,
    std::unordered_map<unsigned int, LveGameObject>& curParticles,
    float _smoothingRadius,
    float _restDensity,
    float _viscosity,
    float _mu,
    LveDevice* _device)
    : particlesMap(curParticles),
      smoothingRadius(_smoothingRadius),
      restDensity(_restDensity),
      viscosity(_viscosity),
      mu(_mu),
      cellSize(_smoothingRadius),
      device{*_device} {
  particleCount = _particleCount;

  // Scale bounding box
  float scale = 1.1f;
  float scaleX = 3.2f * scale;
  float scaleY = 1.0f * scale;
  float scaleZ = 1.2f * scale;

  boxMin.x *= scaleX;
  boxMax.x *= scaleX;
  boxMin.y *= scaleY * 2.0f;
  boxMax.y *= scaleY;
  boxMin.z *= scaleZ;
  boxMax.z *= scaleZ;

  std::cout << "Initializing water physics..." << std::endl;

  constexpr float invPi = 1.0f / glm::pi<float>();

  // Filter active particles
  for (auto& kv : particlesMap) {
    LveGameObject& obj = kv.second;
    if (obj.getId() == 0) continue;
    if (obj.model == nullptr) continue;
    activeParticles.push_back(&obj);
  }

  std::cout << "Active particles: " << activeParticles.size() << std::endl;

  p_velocities.resize(particleCount);
  p_positions.resize(particleCount);
  p_forces.resize(particleCount);
  p_pressures.resize(particleCount);
  p_densities.resize(particleCount);

  // Initialize positions
  for (size_t i = 0; i < particleCount; i++) {
    p_positions[i] = startPos[i];
  }

  std::vector<glm::vec4> temp;
  temp.reserve(p_positions.size());

  for (const auto& v : p_positions) {
    temp.emplace_back(v, 1.0f);
  }
  std::cout << "Active particles: " << activeParticles.size() << std::endl;

  outPositions.resize(particleCount);

  mainGrid.gridDim = ubo.uGridDim;

  mainGrid.numCells = ubo.uNumCells;
  mainGrid.cellSize = ubo.uCellSize;

  mainGrid.cellCount.resize(ubo.uNumCells);
  mainGrid.cellCount.assign(ubo.uNumCells, 0);
  mainGrid.cellStart.resize(ubo.uNumCells);
  mainGrid.cellIndices.resize(ubo.uNumParticles);
  std::cout << "cellIndices Size  " << ubo.uNumParticles << std::endl;
  std::cout << "cellcount Size  " << ubo.uNumCells << std::endl;

  CreateBuffers(mainGrid);
  std::cout << "cellcount Size  " << ubo.uNumCells << std::endl;

  // Temporary upload of particles arranged in a grid

  std::unique_ptr<LveBuffer> outputStaging = makeHostVisible(sizeof(glm::vec4), particleCount);

  outputStaging->writeToBuffer(temp.data());
  outputStaging->flush();

  device.copyBuffer(
      outputStaging->getBuffer(),
      partPosBuff->getBuffer(),
      sizeof(glm::vec4) * particleCount);

  CreateComputePipelineLayout(setLayout);
  CreateComputePipeline();

  std::cout << "\n"
            << "done";
}

WaterPhysics::~WaterPhysics() {
  vkDestroyPipeline(device.device(), computePipeline, nullptr);
  vkDestroyPipelineLayout(device.device(), pipelineLayout, nullptr);
  vkDestroyShaderModule(device.device(), computeShaderModule, nullptr);
}

void WaterPhysics::RunSimulation(float dt, WaterFrameInfo& info) { RunAndReadback(info); }

GridCell WaterPhysics::GetGridCell(const glm::vec3& position) const {
  return GridCell{
      static_cast<int>(std::floor(position.x / cellSize)),
      static_cast<int>(std::floor(position.y / cellSize)),
      static_cast<int>(std::floor(position.z / cellSize))};
}

void WaterPhysics::GetNeighborCells(const GridCell& cell, std::vector<GridCell>& neighbors) const {
  neighbors.clear();

  // Check all 27 neighboring cells (including the cell itself)
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        neighbors.push_back(GridCell{cell.x + dx, cell.y + dy, cell.z + dz});
      }
    }
  }
}

void WaterPhysics::BuildSpatialGrid() {
  spatialGrid.clear();

  const size_t N = particleCount;
  for (size_t i = 0; i < N; ++i) {
    GridCell cell = GetGridCell(p_positions[i]);
    spatialGrid[cell].push_back(i);
  }
}


// no barriers?
void WaterPhysics::RunAndReadback(WaterFrameInfo& frameInfo) {

    vkCmdBindPipeline(frameInfo.commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        computePipeline);

    vkCmdBindDescriptorSets(
        frameInfo.commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipelineLayout,
        0,
        1,
        &frameInfo.computeDescriptorSetPing,
        0,
        nullptr);

    const uint32_t localSizeX = 64;
    uint32_t nParticles = particleCount;
    uint32_t nCells = mainGrid.numCells;

    WaterPushConstants pc;

    auto makeBufferBarrier = [&](VkBuffer buf,
        VkAccessFlags src,
        VkAccessFlags dst)
    {
        VkBufferMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        b.srcAccessMask = src;
        b.dstAccessMask = dst;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.buffer = buf;
        b.offset = 0;
        b.size = VK_WHOLE_SIZE;
        return b;
    };

    uint32_t groupsX_cells =
        (nCells + localSizeX - 1) / localSizeX;

    uint32_t groupsX_particles =
        (nParticles + localSizeX - 1) / localSizeX;



    // =====================================================
    // PASS 0 — Grid init
    // =====================================================

    pc.uPass = 0;
    pc.dt = 1 / 500.f;

    vkCmdPushConstants(
        frameInfo.commandBuffer,
        pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(pc),
        &pc);

    vkCmdDispatch(
        frameInfo.commandBuffer,
        groupsX_cells,
        1,
        1);



    // =====================================================
    // PASS 1 — Collisions (substeps)
    // =====================================================

    const int numCollisionSubsteps = 4;

    for (int i = 0; i < numCollisionSubsteps; i++)
    {
        pc.uPass = 1;

        vkCmdPushConstants(
            frameInfo.commandBuffer,
            pipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(pc),
            &pc);

        vkCmdDispatch(
            frameInfo.commandBuffer,
            groupsX_particles,
            1,
            1);
    }



    // =====================================================
    // PASS 2 — P2G
    // =====================================================

    pc.uPass = 2;

    vkCmdPushConstants(
        frameInfo.commandBuffer,
        pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(pc),
        &pc);

    vkCmdDispatch(
        frameInfo.commandBuffer,
        groupsX_particles,
        1,
        1);



    // =====================================================
    // PASS 3 — Grid finalize
    // =====================================================

    pc.uPass = 3;

    vkCmdPushConstants(
        frameInfo.commandBuffer,
        pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(pc),
        &pc);

    vkCmdDispatch(
        frameInfo.commandBuffer,
        groupsX_cells,
        1,
        1);



    // =====================================================
    // PASS 4 — Density
    // =====================================================

    pc.uPass = 4;

    vkCmdPushConstants(
        frameInfo.commandBuffer,
        pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(pc),
        &pc);

    vkCmdDispatch(
        frameInfo.commandBuffer,
        groupsX_particles,
        1,
        1);



    // =====================================================
    // PASS 5 — Pressure Jacobi
    // =====================================================

    VkDescriptorSet currentSet =
        frameInfo.computeDescriptorSetPing;

    const int numJacobiIterations = 50;

    for (int i = 0; i < numJacobiIterations; i++)
    {
        vkCmdBindDescriptorSets(
            frameInfo.commandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipelineLayout,
            0,
            1,
            &currentSet,
            0,
            nullptr);

        pc.uPass = 5;

        vkCmdPushConstants(
            frameInfo.commandBuffer,
            pipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(pc),
            &pc);

        vkCmdDispatch(
            frameInfo.commandBuffer,
            groupsX_cells,
            1,
            1);

        currentSet =
            (currentSet ==
                frameInfo.computeDescriptorSetPing)
            ? frameInfo.computeDescriptorSetPong
            : frameInfo.computeDescriptorSetPing;
    }



    // =====================================================
    // PASS 6 — Pressure apply
    // =====================================================

    pc.uPass = 6;

    vkCmdPushConstants(
        frameInfo.commandBuffer,
        pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(pc),
        &pc);

    vkCmdDispatch(
        frameInfo.commandBuffer,
        groupsX_cells,
        1,
        1);



    // =====================================================
    // PASS 7 — G2P
    // =====================================================

    pc.uPass = 7;

    vkCmdPushConstants(
        frameInfo.commandBuffer,
        pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(pc),
        &pc);

    vkCmdDispatch(
        frameInfo.commandBuffer,
        groupsX_particles,
        1,
        1);



    // =====================================================
    // PASS 8 — Integrate
    // =====================================================

    pc.uPass = 8;

    vkCmdPushConstants(
        frameInfo.commandBuffer,
        pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(pc),
        &pc);

    vkCmdDispatch(
        frameInfo.commandBuffer,
        groupsX_particles,
        1,
        1);
}


std::unique_ptr<LveBuffer> WaterPhysics::makeHostVisible(VkDeviceSize elemSize, uint32_t count) {
  std::unique_ptr<LveBuffer> buf = std::make_unique<LveBuffer>(
      device,
      elemSize,
      count,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  buf->map();
  return buf;
};// CreateBuffers - MAC grid version
void WaterPhysics::CreateBuffers(Grid& grid) {
  auto makeDeviceLocal = [&](VkDeviceSize elemSize, uint32_t count) {
    return std::make_unique<LveBuffer>(
        device,
        elemSize,
        count,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  };
  auto makeDeviceLocal2 = [&](VkDeviceSize elemSize, uint32_t count) {
    return std::make_unique<LveBuffer>(
        device,
        elemSize,
        count,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  };
  auto makeDeviceLocal_Output = [&](VkDeviceSize elemSize, uint32_t count) {
    return std::make_unique<LveBuffer>(
        device,
        elemSize,
        count,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  };

  auto makeHostVisible = [&](VkDeviceSize elemSize, uint32_t count) {
    return std::make_unique<LveBuffer>(
        device,
        elemSize,
        count,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  };

  // =====================================================
  // PARTICLE BUFFERS
  // =====================================================
  partPosBuff = makeDeviceLocal_Output(sizeof(glm::vec4), particleCount);  // binding 0
  partVelBuff = makeDeviceLocal_Output(sizeof(glm::vec4), particleCount);  // binding 1
  colorsBuff = makeDeviceLocal_Output(sizeof(glm::vec4), particleCount);   // binding 9

  debugBuff = std::make_unique<LveBuffer>(
      device,
      sizeof(float),
      particleCount,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  debugBuff->map();  // binding 11

  // =====================================================
  // MAC GRID VELOCITY BUFFERS (separate u, v, w)
  // =====================================================
  gridUBuff = makeDeviceLocal(sizeof(float), grid.numCells);  // binding 2
  gridVBuff = makeDeviceLocal(sizeof(float), grid.numCells);  // binding 3
  gridWBuff = makeDeviceLocal(sizeof(float), grid.numCells);  // binding 4

  prevGridUBuff = makeDeviceLocal(sizeof(float), grid.numCells);  // binding 5
  prevGridVBuff = makeDeviceLocal(sizeof(float), grid.numCells);  // binding 6
  prevGridWBuff = makeDeviceLocal(sizeof(float), grid.numCells);  // binding 7

  // =====================================================
  // GRID METADATA BUFFERS
  // =====================================================
  gridFlagsBuff = makeDeviceLocal(sizeof(uint32_t), grid.numCells);  // binding 8

  // Weight accumulators for P2G (stored as uint for atomic operations)
  gridDUBuff = makeDeviceLocal(sizeof(uint32_t), grid.numCells);  // binding 12
  gridDVBuff = makeDeviceLocal(sizeof(uint32_t), grid.numCells);  // binding 13
  gridDWBuff = makeDeviceLocal(sizeof(uint32_t), grid.numCells);  // binding 14

  // Solid flags (s array from JS)
  gridSBuff = makeDeviceLocal(sizeof(float), grid.numCells);  // binding 15

  // =====================================================
  // PRESSURE BUFFERS (PING-PONG)
  // =====================================================
  pressureReadBuff = makeDeviceLocal_Output(sizeof(float), grid.numCells);   // binding 16
  pressureWriteBuff = makeDeviceLocal_Output(sizeof(float), grid.numCells);  // binding 17

  // =====================================================
  // INITIALIZE BUFFERS
  // =====================================================

  // Zero particle velocities
  std::vector<glm::vec4> zeroVelocities(particleCount, glm::vec4(0.0f));
  std::unique_ptr<LveBuffer> velStaging = makeHostVisible(sizeof(glm::vec4), particleCount);
  velStaging->map();  // MUST MAP BEFORE WRITING
  velStaging->writeToBuffer(zeroVelocities.data());
  velStaging->flush();
  device.copyBuffer(
      velStaging->getBuffer(),
      partVelBuff->getBuffer(),
      sizeof(glm::vec4) * particleCount);

  // Zero pressure buffers - create separate staging buffers
  std::vector<float> zeroPressure(grid.numCells, 0.0f);

  // First pressure buffer
  std::unique_ptr<LveBuffer> pressureStaging1 = makeHostVisible(sizeof(float), grid.numCells);
  pressureStaging1->map();  // MUST MAP BEFORE WRITING
  pressureStaging1->writeToBuffer(zeroPressure.data());
  pressureStaging1->flush();
  device.copyBuffer(
      pressureStaging1->getBuffer(),
      pressureReadBuff->getBuffer(),
      sizeof(float) * grid.numCells);

  // Second pressure buffer (separate staging buffer)
  std::unique_ptr<LveBuffer> pressureStaging2 = makeHostVisible(sizeof(float), grid.numCells);
  pressureStaging2->map();  // MUST MAP BEFORE WRITING
  pressureStaging2->writeToBuffer(zeroPressure.data());
  pressureStaging2->flush();
  device.copyBuffer(
      pressureStaging2->getBuffer(),
      pressureWriteBuff->getBuffer(),
      sizeof(float) * grid.numCells);


  // sum(pvel * weight)
  gridUAccumBuff = makeDeviceLocal2(sizeof(uint32_t), grid.numCells);  // 18
  gridVAccumBuff = makeDeviceLocal2(sizeof(uint32_t), grid.numCells);  // 19
  gridWAccumBuff = makeDeviceLocal2(sizeof(uint32_t), grid.numCells);  // 20

  // sum(weight)
  gridUWeightBuff = makeDeviceLocal2(sizeof(uint32_t), grid.numCells);  // 21
  gridVWeightBuff = makeDeviceLocal2(sizeof(uint32_t), grid.numCells);  // 22
  gridWWeightBuff = makeDeviceLocal2(sizeof(uint32_t), grid.numCells);  // 23
  gridDensityBuff = makeDeviceLocal2(sizeof(uint32_t), grid.numCells);  // 23


  std::vector<uint32_t> zeroUint(grid.numCells, 0u);

  auto zeroUintBuffer = [&](std::unique_ptr<LveBuffer>& target) {
    auto staging = makeHostVisible(sizeof(uint32_t), grid.numCells);
    staging->map();
    staging->writeToBuffer(zeroUint.data());
    staging->flush();

    device.copyBuffer(staging->getBuffer(), target->getBuffer(), sizeof(uint32_t) * grid.numCells);
  };

  // Zero all
  zeroUintBuffer(gridUAccumBuff);
  zeroUintBuffer(gridVAccumBuff);
  zeroUintBuffer(gridWAccumBuff);

  zeroUintBuffer(gridUWeightBuff);
  zeroUintBuffer(gridVWeightBuff);
  zeroUintBuffer(gridWWeightBuff);

  zeroUintBuffer(gridDensityBuff);

}
void WaterPhysics::CreateComputePipelineLayout(VkDescriptorSetLayout setLayout) {
  VkPushConstantRange pushConstantRange{};
  pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pushConstantRange.offset = 0;
  pushConstantRange.size = sizeof(WaterPushConstants);

  VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};

  layoutInfo.setLayoutCount = 1;
  layoutInfo.pSetLayouts = &setLayout;
  layoutInfo.pushConstantRangeCount = 1;
  layoutInfo.pPushConstantRanges = &pushConstantRange;

  if (vkCreatePipelineLayout(device.device(), &layoutInfo, nullptr, &pipelineLayout) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create compute pipeline layout");
  }

}

void WaterPhysics::CreateComputePipeline() {
  // Create shader module
  computeShaderModule = LvePipeline::loadShaderModule(
      "C:\\Users\\ZyBros\\Downloads\\NeonEngine\\NeonEngine\\shaders\\water_grid.comp.spv",
      device.device());

  VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stageInfo.module = computeShaderModule;
  stageInfo.pName = "main";

  VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipelineInfo.stage = stageInfo;
  pipelineInfo.layout = pipelineLayout;

  vkCreateComputePipelines(
      device.device(),
      VK_NULL_HANDLE,
      1,
      &pipelineInfo,
      nullptr,
      &computePipeline);
}

}  // namespace lve