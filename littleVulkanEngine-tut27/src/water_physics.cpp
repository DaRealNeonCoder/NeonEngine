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
    VkDescriptorSetLayout setLayout, WaterPhysUbo &ubo, 
    std::unordered_map<unsigned int, LveGameObject>& curParticles,
    float _smoothingRadius,
    float _restDensity,
    float _viscosity,
    float _mu, LveDevice *_device)
    : particlesMap(curParticles),
      smoothingRadius(_smoothingRadius),
      restDensity(_restDensity),
      viscosity(_viscosity),
      mu(_mu),
      cellSize(_smoothingRadius),
     device{*_device}  
{


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

  p_velocities.resize(activeParticles.size());
  p_positions.resize(activeParticles.size());
  p_forces.resize(activeParticles.size());
  p_pressures.resize(activeParticles.size());
  p_densities.resize(activeParticles.size());

  // Initialize positions
  for (size_t i = 0; i < activeParticles.size(); i++) {
    p_positions[i] = activeParticles[i]->transform.translation;
  }

  std::vector<glm::vec4> temp;
  temp.reserve(p_positions.size());

  for (const auto& v : p_positions) {
    temp.emplace_back(v, 1.0f);
  }
  std::cout << "Active particles: " << activeParticles.size() << std::endl;

  particleCount = temp.size();
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

  vkGetDeviceQueue(device.device(), 0, 0, &computeQueue);

  CreateBuffers(mainGrid);

  outputBuffer->writeToBuffer(temp.data());
  outputBuffer->flush();

  std::vector<glm::vec4> zeroVel(particleCount, glm::vec4(0.0f));
  velocitiesBuff->writeToBuffer(zeroVel.data());
  velocitiesBuff->flush();

  std::vector<float> initDensity(particleCount, restDensity /* or phyUbo.uRestDensity */);
  densitiesBuff->writeToBuffer(initDensity.data());
  densitiesBuff->flush();

  std::vector<float> zeroFloat(particleCount, 0.0f);
  pressuresBuff->writeToBuffer(zeroFloat.data());
  pressuresBuff->flush();

  std::vector<glm::vec4> zeroForces(particleCount, glm::vec4(0.0f));
  forcesBuff->writeToBuffer(zeroForces.data());
  forcesBuff->flush();


  UploadBuffers(mainGrid);
  CreateComputePipelineLayout(setLayout);
  CreateComputePipeline();

  std::cout << "\n"<< "done";
}

WaterPhysics::~WaterPhysics() { 
    vkDestroyPipeline(device.device(), computePipeline, nullptr);
  vkDestroyPipelineLayout(device.device(), pipelineLayout, nullptr);
  vkDestroyShaderModule(device.device(), computeShaderModule, nullptr);
}

void WaterPhysics::RunSimulation(float dt, WaterFrameInfo &info) {
  RunAndReadback(info);
}

// ============================================================================
// Grid Functions
// ============================================================================

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

  const size_t N = activeParticles.size();
  for (size_t i = 0; i < N; ++i) {
    GridCell cell = GetGridCell(p_positions[i]);
    spatialGrid[cell].push_back(i);
  }
}

void WaterPhysics::UploadBuffers(const Grid &grid) {
  cellIndices->writeToBuffer((void*)grid.cellIndices.data());
  cellIndices->flush();

  cellCount->writeToBuffer((void*)grid.cellCount.data());
  cellCount->flush();

  cellStart->writeToBuffer((void*)grid.cellStart.data());
  cellStart->flush();
  // keep mapped or unmap if your LveBuffer's writeToBuffer already unmaps
}
void WaterPhysics::RunAndReadback(WaterFrameInfo& frameInfo) {
  BuildUniformGrid(mainGrid, p_positions);
  UploadBuffers(mainGrid);

  // 1. Verify grid population
  int totalInGrid = 0;
  int nonEmpty = 0;
  for (int c = 0; c < mainGrid.numCells; c++) {
    if (mainGrid.cellCount[c] > 0) {
      nonEmpty++;
      totalInGrid += mainGrid.cellCount[c];
    }
  }
  std::cout << "=== GRID DEBUG ===" << std::endl;
  std::cout << "Non-empty cells: " << nonEmpty << " / " << mainGrid.numCells << std::endl;
  std::cout << "Particles in grid: " << totalInGrid << " / " << particleCount << std::endl;

  // 2. Check first few particles
  for (int i = 0; i < std::min(3, (int)particleCount); i++) {
    glm::ivec3 cell = PositionToGridCell(p_positions[i], mainGrid.cellSize, mainGrid.gridDim);
    int cellIdx = Cell3DToIndex(cell, mainGrid.gridDim);
    std::cout << "Particle " << i << ": pos=" << p_positions[i].x << "," << p_positions[i].y << ","
              << p_positions[i].z << " -> cell(" << cell.x << "," << cell.y << "," << cell.z
              << ") = idx " << cellIdx << " has " << mainGrid.cellCount[cellIdx] << " particles"
              << std::endl;
  }

  // 3. Dump first cell's contents
  if (nonEmpty > 0) {
    int firstCellIdx = -1;
    for (int c = 0; c < mainGrid.numCells; c++) {
      if (mainGrid.cellCount[c] > 0) {
        firstCellIdx = c;
        break;
      }
    }
    std::cout << "First non-empty cell " << firstCellIdx << " contains:" << std::endl;
    int start = mainGrid.cellStart[firstCellIdx];
    int count = mainGrid.cellCount[firstCellIdx];
    for (int i = 0; i < count && i < 5; i++) {
      std::cout << "  cellIndices[" << (start + i) << "] = " << mainGrid.cellIndices[start + i]
                << std::endl;
    }
  }

  // DEBUG: Verify data made it into buffers
  void* mapped = cellCount->getMappedMemory();
  int* countData = static_cast<int*>(mapped);
  std::cout << "cellCount[4150] on CPU after upload: " << countData[4150] << std::endl;

  mapped = cellStart->getMappedMemory();
  int* startData = static_cast<int*>(mapped);
  std::cout << "cellStart[4150] on CPU after upload: " << startData[4150] << std::endl;

  std::cout << "==================" << std::endl;
  std::cout << p_positions[0].x << "  " << p_positions[0].y << "  " << p_positions[0].z << "\n";

  // ========================================
  // CRITICAL: Buffer-specific barriers for grid data
  // ========================================
  std::array<VkBufferMemoryBarrier, 3> bufferBarriers{};

  // cellIndices barrier
  bufferBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  bufferBarriers[0].pNext = nullptr;
  bufferBarriers[0].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
  bufferBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  bufferBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bufferBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bufferBarriers[0].buffer = cellIndices->getBuffer();
  bufferBarriers[0].offset = 0;
  bufferBarriers[0].size = VK_WHOLE_SIZE;

  // cellCount barrier
  bufferBarriers[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  bufferBarriers[1].pNext = nullptr;
  bufferBarriers[1].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
  bufferBarriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  bufferBarriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bufferBarriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bufferBarriers[1].buffer = cellCount->getBuffer();
  bufferBarriers[1].offset = 0;
  bufferBarriers[1].size = VK_WHOLE_SIZE;

  // cellStart barrier
  bufferBarriers[2].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  bufferBarriers[2].pNext = nullptr;
  bufferBarriers[2].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
  bufferBarriers[2].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  bufferBarriers[2].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bufferBarriers[2].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bufferBarriers[2].buffer = cellStart->getBuffer();
  bufferBarriers[2].offset = 0;
  bufferBarriers[2].size = VK_WHOLE_SIZE;

  vkCmdPipelineBarrier(
      frameInfo.commandBuffer,
      VK_PIPELINE_STAGE_HOST_BIT,            // src stage
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // dst stage
      0,
      0,
      nullptr,
      3,
      bufferBarriers.data(),
      0,
      nullptr);

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

  // Pass 0: Compute densities/pressures
  pc.uPass = 0;
  vkCmdPushConstants(
      frameInfo.commandBuffer,
      pipelineLayout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(WaterPushConstants),
      &pc);

  vkCmdDispatch(frameInfo.commandBuffer, groups, 1, 1);

  // ========================================
  // BARRIER BETWEEN PASSES
  // ========================================
  VkMemoryBarrier memoryBarrier{};
  memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  vkCmdPipelineBarrier(
      frameInfo.commandBuffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // src stage
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // dst stage
      0,
      1,
      &memoryBarrier,
      0,
      nullptr,
      0,
      nullptr);

  // Pass 1: Compute forces and integrate
  pc.uPass = 1;
  vkCmdPushConstants(
      frameInfo.commandBuffer,
      pipelineLayout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(WaterPushConstants),
      &pc);

  vkCmdDispatch(frameInfo.commandBuffer, groups, 1, 1);

  // ========================================
  // BARRIER FOR HOST READ
  // ========================================
  VkBufferMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.buffer = outputBuffer->getBuffer();
  barrier.offset = 0;
  barrier.size = VK_WHOLE_SIZE;

  vkCmdPipelineBarrier(
      frameInfo.commandBuffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_HOST_BIT,
      0,
      0,
      nullptr,
      1,
      &barrier,
      0,
      nullptr);

  vkQueueWaitIdle(computeQueue);

  // Copy data from buffer
  void* mapped2 = outputBuffer->getMappedMemory();
  memcpy(outPositions.data(), mapped2, sizeof(glm::vec4) * particleCount);
  std::cout << "Particle 0 neighbor count from GPU: " << outPositions[0].w << std::endl;
  std::cout << "Particle 0 density: " << outPositions[0].x << std::endl;

  // Update game objects
  for (size_t i = 0; i < activeParticles.size(); i++) {
    LveGameObject* p = activeParticles[i];
    p_positions[i] = glm::vec3(outPositions[i].x, outPositions[i].y, outPositions[i].z);
    p->transform.translation = glm::vec3(outPositions[i].x, outPositions[i].y, outPositions[i].z);
  }
}

void WaterPhysics::CreateBuffers(Grid& grid) {
  // Grid-related buffers
  cellCount = std::make_unique<LveBuffer>(
      device,
      sizeof(int),
      grid.numCells,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  cellCount->map();

  cellIndices = std::make_unique<LveBuffer>(
      device,
      sizeof(int),
      particleCount,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  cellIndices->map();

  cellStart = std::make_unique<LveBuffer>(
      device,
      sizeof(int),
      grid.numCells,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  cellStart->map();

  // Physics state buffers (per-particle)
  velocitiesBuff = std::make_unique<LveBuffer>(
      device,
      sizeof(glm::vec4),
      particleCount,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  velocitiesBuff->map();

  densitiesBuff = std::make_unique<LveBuffer>(
      device,
      sizeof(float),
      particleCount,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  densitiesBuff->map();

  pressuresBuff = std::make_unique<LveBuffer>(
      device,
      sizeof(float),
      particleCount,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  pressuresBuff->map();

  forcesBuff = std::make_unique<LveBuffer>(
      device,
      sizeof(glm::vec4),
      particleCount,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  forcesBuff->map();

  // Output positions buffer
  outputBuffer = std::make_unique<LveBuffer>(
      device,
      sizeof(glm::vec4),
      particleCount,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  outputBuffer->map();
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
  computeShaderModule = LvePipeline::loadShaderModule("C:\\Users\\ZyBros\\Downloads\\littleVulkanEngine-tut27\\littleVulkanEngine-"
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

};

