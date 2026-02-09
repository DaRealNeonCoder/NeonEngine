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

  BuildUniformGrid(mainGrid, p_positions);

  CreateBuffers(mainGrid);

  // Temporary upload of particles arranged in a grid

  std::unique_ptr<LveBuffer> outputStaging = makeHostVisible(sizeof(glm::vec4), particleCount);

  outputStaging->writeToBuffer(temp.data());
  outputStaging->flush();

  device.copyBuffer(
      outputStaging->getBuffer(),
      outputBuffer->getBuffer(),
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

void WaterPhysics::UploadBuffers(const Grid& grid) {
  cellIndices->writeToBuffer((void*)grid.cellIndices.data());
  cellIndices->flush();

  cellCount->writeToBuffer((void*)grid.cellCount.data());
  cellCount->flush();

  cellStart->writeToBuffer((void*)grid.cellStart.data());
  cellStart->flush();
  // keep mapped or unmap if your LveBuffer's writeToBuffer already unmaps
}
void WaterPhysics::RunAndReadback(WaterFrameInfo& frameInfo) {
  // Bind pipeline & descriptor set once
  vkCmdBindPipeline(frameInfo.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
  vkCmdBindDescriptorSets(
      frameInfo.commandBuffer,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      pipelineLayout,
      0,
      1,
      &frameInfo.computeDescriptorSet,
      0,
      nullptr);

  uint32_t groupSize = 64;
  uint32_t groups = (particleCount + groupSize - 1) / groupSize;

  WaterPushConstants pc;

  // Create a helper function for barriers
  auto makeBufferBarrier = [&](VkBuffer buf, VkAccessFlags srcAccess, VkAccessFlags dstAccess) {
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buf;
    barrier.offset = 0;
    barrier.size = VK_WHOLE_SIZE;
    return barrier;
  };

  // Get all buffer handles
  VkBuffer outputBuf = outputBuffer->getBuffer();
  VkBuffer velocitiesBuf = velocitiesBuff->getBuffer();
  VkBuffer positionsBuf = outputBuffer->getBuffer();  // Make sure this exists
  VkBuffer cellCountBuf = cellCount->getBuffer();
  VkBuffer cellStartBuf = cellStart->getBuffer();
  VkBuffer cellIndicesBuf = cellIndices->getBuffer();
  VkBuffer cellCursorBuf = cellCursorBuff->getBuffer();
  VkBuffer densitiesBuf = densitiesBuff->getBuffer();
  VkBuffer pressuresBuf = pressuresBuff->getBuffer();
  VkBuffer forcesBuf = forcesBuff->getBuffer();  // Make sure this exists

  // --------------- STEP 1: Host -> Compute barrier for all uploaded data ---------------
  std::vector<VkBufferMemoryBarrier> hostToComputeBarriers;

  // All buffers that host might have written to
  hostToComputeBarriers.push_back(makeBufferBarrier(
      positionsBuf,
      VK_ACCESS_HOST_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));

  hostToComputeBarriers.push_back(makeBufferBarrier(
      velocitiesBuf,
      VK_ACCESS_HOST_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));

  hostToComputeBarriers.push_back(makeBufferBarrier(
      cellCountBuf,
      VK_ACCESS_HOST_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));

  hostToComputeBarriers.push_back(makeBufferBarrier(
      cellStartBuf,
      VK_ACCESS_HOST_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));

  hostToComputeBarriers.push_back(makeBufferBarrier(
      cellIndicesBuf,
      VK_ACCESS_HOST_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));

  hostToComputeBarriers.push_back(makeBufferBarrier(
      cellCursorBuf,
      VK_ACCESS_HOST_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));

  if (!hostToComputeBarriers.empty()) {
    vkCmdPipelineBarrier(
        frameInfo.commandBuffer,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        static_cast<uint32_t>(hostToComputeBarriers.size()),
        hostToComputeBarriers.data(),
        0,
        nullptr);
  }

  // --------------- PASS -1: Clear grid ---------------
  pc.uPass = -1;
  pc.dt = 0.002f;
  vkCmdPushConstants(
      frameInfo.commandBuffer,
      pipelineLayout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(WaterPushConstants),
      &pc);

  // Dispatch for grid cells
  uint32_t gridGroups = (mainGrid.numCells + groupSize - 1) / groupSize;
  vkCmdDispatch(frameInfo.commandBuffer, gridGroups, 1, 1);

  // Barrier: Pass -1 writes -> Pass 0 reads
  std::vector<VkBufferMemoryBarrier> passNeg1Barriers;
  passNeg1Barriers.push_back(makeBufferBarrier(
      cellCountBuf,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));
  passNeg1Barriers.push_back(makeBufferBarrier(
      cellCursorBuf,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));

  vkCmdPipelineBarrier(
      frameInfo.commandBuffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      0,
      nullptr,
      static_cast<uint32_t>(passNeg1Barriers.size()),
      passNeg1Barriers.data(),
      0,
      nullptr);

  // --------------- PASS 0: Count particles per cell ---------------
  pc.uPass = 0;
  vkCmdPushConstants(
      frameInfo.commandBuffer,
      pipelineLayout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(WaterPushConstants),
      &pc);

  vkCmdDispatch(frameInfo.commandBuffer, groups, 1, 1);

  // Barrier: Pass 0 writes -> Pass 1 reads (atomic counters)
  std::vector<VkBufferMemoryBarrier> pass0Barriers;
  pass0Barriers.push_back(
      makeBufferBarrier(cellCountBuf, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT));

  vkCmdPipelineBarrier(
      frameInfo.commandBuffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      0,
      nullptr,
      static_cast<uint32_t>(pass0Barriers.size()),
      pass0Barriers.data(),
      0,
      nullptr);

  // --------------- PASS 1: Exclusive scan ---------------
  pc.uPass = 1;
  vkCmdPushConstants(
      frameInfo.commandBuffer,
      pipelineLayout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(WaterPushConstants),
      &pc);

  // Only 1 thread for scan, but we still need a dispatch
  vkCmdDispatch(frameInfo.commandBuffer, 1, 1, 1);

  // Barrier: Pass 1 writes -> Pass 2 reads
  std::vector<VkBufferMemoryBarrier> pass1Barriers;
  pass1Barriers.push_back(
      makeBufferBarrier(cellStartBuf, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT));
  pass1Barriers.push_back(makeBufferBarrier(
      cellCursorBuf,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));

  vkCmdPipelineBarrier(
      frameInfo.commandBuffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      0,
      nullptr,
      static_cast<uint32_t>(pass1Barriers.size()),
      pass1Barriers.data(),
      0,
      nullptr);

  // --------------- PASS 2: Fill cell indices ---------------
  pc.uPass = 2;
  vkCmdPushConstants(
      frameInfo.commandBuffer,
      pipelineLayout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(WaterPushConstants),
      &pc);

  vkCmdDispatch(frameInfo.commandBuffer, groups, 1, 1);

  // Barrier: Pass 2 writes -> Pass 3 reads (density computation)
  std::vector<VkBufferMemoryBarrier> pass2Barriers;
  pass2Barriers.push_back(
      makeBufferBarrier(cellIndicesBuf, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT));
  pass2Barriers.push_back(
      makeBufferBarrier(cellCursorBuf, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT));

  vkCmdPipelineBarrier(
      frameInfo.commandBuffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      0,
      nullptr,
      static_cast<uint32_t>(pass2Barriers.size()),
      pass2Barriers.data(),
      0,
      nullptr);

  // --------------- PASS 3: Compute density and pressure ---------------
  pc.uPass = 3;
  vkCmdPushConstants(
      frameInfo.commandBuffer,
      pipelineLayout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(WaterPushConstants),
      &pc);

  vkCmdDispatch(frameInfo.commandBuffer, groups, 1, 1);

  // Barrier: Pass 3 writes -> Pass 4 reads
  std::vector<VkBufferMemoryBarrier> pass3Barriers;
  pass3Barriers.push_back(
      makeBufferBarrier(densitiesBuf, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT));
  pass3Barriers.push_back(
      makeBufferBarrier(pressuresBuf, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT));

  vkCmdPipelineBarrier(
      frameInfo.commandBuffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      0,
      nullptr,
      static_cast<uint32_t>(pass3Barriers.size()),
      pass3Barriers.data(),
      0,
      nullptr);

  pc.uPass = 4;
  vkCmdPushConstants(
      frameInfo.commandBuffer,
      pipelineLayout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(WaterPushConstants),
      &pc);

  vkCmdDispatch(frameInfo.commandBuffer, groups, 1, 1);

  // --- make compute writes visible to the vertex input (so the renderer can read positions/colors)
  // ---
  std::vector<VkBufferMemoryBarrier> computeToVertexBarriers;
  computeToVertexBarriers.push_back(makeBufferBarrier(
      positionsBuf,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT));
  computeToVertexBarriers.push_back(makeBufferBarrier(
      colorsBuff->getBuffer(),
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT));

  vkCmdPipelineBarrier(
      frameInfo.commandBuffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,  // allow vertex input to consume these buffers
      0,
      0,
      nullptr,
      static_cast<uint32_t>(computeToVertexBarriers.size()),
      computeToVertexBarriers.data(),
      0,
      nullptr);

  // --------------- FINAL: Compute -> Host barrier for readback ---------------
  std::vector<VkBufferMemoryBarrier> finalBarriers;

  // Positions were written by pass 4
  finalBarriers.push_back(
      makeBufferBarrier(positionsBuf, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT));

  // Velocities were written by pass 4
  finalBarriers.push_back(
      makeBufferBarrier(velocitiesBuf, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT));

  // Forces were written by pass 4
  finalBarriers.push_back(
      makeBufferBarrier(forcesBuf, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT));

  // Densities and pressures if you want to read them
  finalBarriers.push_back(
      makeBufferBarrier(densitiesBuf, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT));
  finalBarriers.push_back(
      makeBufferBarrier(pressuresBuf, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT));

  vkCmdPipelineBarrier(
      frameInfo.commandBuffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_HOST_BIT,
      0,
      0,
      nullptr,
      static_cast<uint32_t>(finalBarriers.size()),
      finalBarriers.data(),
      0,
      nullptr);
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
};
void WaterPhysics::CreateBuffers(Grid& grid) {
  auto makeDeviceLocal = [&](VkDeviceSize elemSize, uint32_t count) {
    return std::make_unique<LveBuffer>(
        device,
        elemSize,
        count,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  };

  auto makeDeviceLocal_Output = [&](VkDeviceSize elemSize, uint32_t count) {
    auto buf = std::make_unique<LveBuffer>(
        device,
        elemSize,
        count,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    return buf;
  };

  colorsBuff = std::make_unique<LveBuffer>(
      device,
      sizeof(glm::vec4),
      particleCount,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  ;
  // =====================================================
  // GRID BUFFERS (GPU only)
  // =====================================================
  cellCount = makeDeviceLocal(sizeof(int), grid.numCells);
  cellCursorBuff = makeDeviceLocal(sizeof(int), grid.numCells);
  cellIndices = makeDeviceLocal(sizeof(int), particleCount);
  cellStart = makeDeviceLocal(sizeof(int), grid.numCells);

  // =====================================================
  // PARTICLE STATE (GPU compute heavy)
  // =====================================================
  velocitiesBuff = makeDeviceLocal(sizeof(glm::vec4), particleCount);
  densitiesBuff = makeDeviceLocal(sizeof(float), particleCount);
  pressuresBuff = makeDeviceLocal(sizeof(float), particleCount);
  forcesBuff = makeDeviceLocal(sizeof(glm::vec4), particleCount);

  // =====================================================
  // OUTPUT POSITIONS (CPU READBACK REQUIRED)
  // =====================================================
  outputBuffer = std::make_unique<LveBuffer>(
      device,
      sizeof(glm::vec4),
      particleCount,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  ;
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
      "C:\\Users\\ZyBros\\Downloads\\littleVulkanEngine-tut27\\littleVulkanEngine-"
      "tut27\\shaders\\water_phy.comp.spv",
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

glm::ivec3 WaterPhysics::PositionToGridCell(
    const glm::vec3& pos, float cellSize, const glm::ivec3& gridDim) {
  glm::vec3 extent = -0.5f * glm::vec3(gridDim) * cellSize;

  glm::vec3 relPos = pos - extent;  // position relative to grid origin
  glm::ivec3 cell;

  // center grid at origin
  cell.x = static_cast<int>(floor(relPos.x / cellSize));
  cell.y = static_cast<int>(floor(relPos.y / cellSize));
  cell.z = static_cast<int>(floor(relPos.z / cellSize));

  // Clamp to grid boundaries (optional)
  cell.x = glm::clamp(cell.x, 0, gridDim.x - 1);
  cell.y = glm::clamp(cell.y, 0, gridDim.y - 1);
  cell.z = glm::clamp(cell.z, 0, gridDim.z - 1);

  return cell;
}
int WaterPhysics::Cell3DToIndex(const glm::ivec3& cell, const glm::ivec3& gridDim) {
  return cell.x + cell.y * gridDim.x + cell.z * gridDim.x * gridDim.y;
}

void WaterPhysics::ClearGrid(Grid& grid) {
  std::fill(grid.cellStart.begin(), grid.cellStart.end(), 0);
  std::fill(grid.cellCount.begin(), grid.cellCount.end(), 0);
}

void WaterPhysics::CountParticlesPerCell(Grid& grid, const std::vector<glm::vec3>& positions) {
  for (int i = 0; i < positions.size(); ++i) {
    glm::ivec3 cell = PositionToGridCell(positions[i], grid.cellSize, grid.gridDim);

    int cellIndex = Cell3DToIndex(cell, grid.gridDim);
    grid.cellCount[cellIndex]++;
  }
}

void WaterPhysics::BuildCellStart(Grid& grid) {
  int running = 0;
  for (int c = 0; c < grid.numCells; ++c) {
    grid.cellStart[c] = running;
    running += grid.cellCount[c];
  }
}

void WaterPhysics::FillCellIndices(Grid& grid, const std::vector<glm::vec3>& positions) {
  const int N = positions.size();
  std::vector<int> cursor = grid.cellStart;

  for (int i = 0; i < N; ++i) {
    glm::ivec3 cell = PositionToGridCell(positions[i], grid.cellSize, grid.gridDim);
    int cellIndex = Cell3DToIndex(cell, grid.gridDim);
    int dst = cursor[cellIndex]++;

    grid.cellIndices[dst] = i;
  }
}

void WaterPhysics::BuildUniformGrid(Grid& grid, const std::vector<glm::vec3>& positions) {
  ClearGrid(grid);
  CountParticlesPerCell(grid, positions);
  BuildCellStart(grid);
  FillCellIndices(grid, positions);
}
}  // namespace lve