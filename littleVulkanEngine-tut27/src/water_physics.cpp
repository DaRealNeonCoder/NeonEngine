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
void WaterPhysics::RunAndReadback(WaterFrameInfo& frameInfo) {
  // Bind pipeline & descriptor set once
  vkCmdBindPipeline(frameInfo.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);

  vkCmdBindDescriptorSets(
      frameInfo.commandBuffer,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      pipelineLayout,
      0,
      1,
      &frameInfo.computeDescriptorSetPing,
      0,
      nullptr);

  // NOTE: shader uses local_size_x = 64 (1D). We'll keep y/z = 1 so 3D dispatch still works for
  // Jacobi.
  const uint32_t localSizeX = 64;
  const uint32_t localSizeY = 1;
  const uint32_t localSizeZ = 1;

  uint32_t nParticles = particleCount;
  uint32_t nCells = mainGrid.numCells;

  WaterPushConstants pc;

  // Helper for barriers
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

  // --- Buffers (match shader bindings) ---
  VkBuffer partPosBuf = partPosBuff->getBuffer();  // layout(binding = 0) PartPos { vec4 p[]; }
  VkBuffer partVelBuf = partVelBuff->getBuffer();  // layout(binding = 1) PartVel { vec4 v[]; }
  VkBuffer gridVelAccumBuf =
      gridVelAccumBuff->getBuffer();  // layout(binding = 2) GridVelAccum { uint velAccum[]; }
  VkBuffer gridWeightBuf =
      gridWeightBuff->getBuffer();  // layout(binding = 3) GridWeight { uint weight[]; }
  VkBuffer gridFlagsBuf =
      gridFlagsBuff->getBuffer();  // layout(binding = 4) GridFlags { uint flags[]; }
  VkBuffer gridVelBuf =
      gridVelBuff->getBuffer();  // layout(binding = 5) GridVel { vec4 gridVel[]; }
  VkBuffer prevGridVelBuf =
      prevGridVelBuff->getBuffer();  // layout(binding = 6) PrevGridVel { vec4 prevGridVel[]; }
  VkBuffer pressureReadBuf =
      pressureReadBuff->getBuffer();  // layout(binding = 7) PressureRead { float pRead[]; }
  VkBuffer pressureWriteBuf =
      pressureWriteBuff->getBuffer();  // layout(binding = 8) PressureWrite { float pWrite[]; }
  VkBuffer colorsBuf = colorsBuff->getBuffer();  // layout(binding = 9) Colors { vec4 colors[]; }

  // --------------- HOST -> COMPUTE barrier for any buffers the host wrote ---------------
  std::vector<VkBufferMemoryBarrier> hostToComputeBarriers;
  // If you uploaded positions/velocities/pressures etc from host earlier this frame, include them:
  hostToComputeBarriers.push_back(makeBufferBarrier(
      partPosBuf,
      VK_ACCESS_HOST_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));
  hostToComputeBarriers.push_back(makeBufferBarrier(
      partVelBuf,
      VK_ACCESS_HOST_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));
  // If you uploaded pressure initial guesses:
  hostToComputeBarriers.push_back(makeBufferBarrier(
      pressureReadBuf,
      VK_ACCESS_HOST_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));
  hostToComputeBarriers.push_back(makeBufferBarrier(
      pressureWriteBuf,
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

  // ---------------- PASS 0: Grid init (shader pass 0) ----------------
  pc.uPass = 0;
  pc.dt = frameInfo.frameTime;
  vkCmdPushConstants(
      frameInfo.commandBuffer,
      pipelineLayout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(WaterPushConstants),
      &pc);

  // Dispatch 1D over cells. Because shader's local_size_x = 64, we dispatch groups in X:
  uint32_t groupsX_cells = (nCells + localSizeX - 1) / localSizeX;
  vkCmdDispatch(frameInfo.commandBuffer, groupsX_cells, 1, 1);

  // Barrier: pass0 writes -> pass1 reads (velAccum, weight, flags, gridVel, prevGridVel)
  std::vector<VkBufferMemoryBarrier> pass0Barriers;
  pass0Barriers.push_back(makeBufferBarrier(
      gridVelAccumBuf,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));
  pass0Barriers.push_back(makeBufferBarrier(
      gridWeightBuf,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));
  pass0Barriers.push_back(makeBufferBarrier(
      gridFlagsBuf,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));
  pass0Barriers.push_back(makeBufferBarrier(
      gridVelBuf,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));
  pass0Barriers.push_back(makeBufferBarrier(
      prevGridVelBuf,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));

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

  // ---------------- PASS 1: P2G (particles -> grid) ----------------
  pc.uPass = 1;
  vkCmdPushConstants(
      frameInfo.commandBuffer,
      pipelineLayout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(WaterPushConstants),
      &pc);

  uint32_t groupsX_particles = (nParticles + localSizeX - 1) / localSizeX;
  vkCmdDispatch(frameInfo.commandBuffer, groupsX_particles, 1, 1);

  // Barrier: pass1 writes -> pass2 reads
  std::vector<VkBufferMemoryBarrier> pass1Barriers;
  pass1Barriers.push_back(makeBufferBarrier(
      gridVelAccumBuf,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));
  pass1Barriers.push_back(makeBufferBarrier(
      gridWeightBuf,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));
  pass1Barriers.push_back(makeBufferBarrier(
      gridFlagsBuf,
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

  // ---------------- PASS 2: Grid finalize ----------------
  pc.uPass = 2;
  vkCmdPushConstants(
      frameInfo.commandBuffer,
      pipelineLayout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(WaterPushConstants),
      &pc);

  // Dispatch over cells again:
  vkCmdDispatch(frameInfo.commandBuffer, groupsX_cells, 1, 1);

  // Barrier: pass2 writes -> pass3 reads (gridVel, prevGridVel, flags, weight)
  std::vector<VkBufferMemoryBarrier> pass2Barriers;
  pass2Barriers.push_back(makeBufferBarrier(
      gridVelBuf,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));
  pass2Barriers.push_back(makeBufferBarrier(
      prevGridVelBuf,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));
  pass2Barriers.push_back(makeBufferBarrier(
      gridFlagsBuf,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));
  pass2Barriers.push_back(makeBufferBarrier(
      gridWeightBuf,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));

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

  // ---------------- PASS 3: Pressure Jacobi (3D dispatch) ----------------
  // In RunAndReadback(), replace the single pass 3 dispatch with:
  VkDescriptorSet currentSet = frameInfo.computeDescriptorSetPing;

  // ---------------- PASS 3: Pressure Jacobi (iterate multiple times) ----------------
  const int numJacobiIterations = 50;  // adjust as needed (10-50 typical)
  for (int iter = 0; iter < numJacobiIterations; ++iter) {
    // bind descriptor set for THIS iteration
    vkCmdBindDescriptorSets(
        frameInfo.commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipelineLayout,
        0,
        1,
        &currentSet,
        0,
        nullptr);

    pc.uPass = 3;
    vkCmdPushConstants(
        frameInfo.commandBuffer,
        pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(WaterPushConstants),
        &pc);

    vkCmdDispatch(frameInfo.commandBuffer, groupsX_cells, 1, 1);
    // after vkCmdDispatch(...)
VkBufferMemoryBarrier barrierA =
    makeBufferBarrier(pressureReadBuf, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
VkBufferMemoryBarrier barrierB =
    makeBufferBarrier(pressureWriteBuf, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

VkBufferMemoryBarrier both[] = { barrierA, barrierB };

vkCmdPipelineBarrier(
    frameInfo.commandBuffer,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    0,
    0,
    nullptr,
    2,
    both,
    0,
    nullptr);

    // swap descriptor sets for next iteration
    currentSet = (currentSet == frameInfo.computeDescriptorSetPing)
                     ? frameInfo.computeDescriptorSetPong
                     : frameInfo.computeDescriptorSetPing;
  }

  // ---------------- PASS 4: Pressure apply (GRID only) ----------------

  vkCmdBindDescriptorSets(
      frameInfo.commandBuffer,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      pipelineLayout,
      0,
      1,
      &currentSet,  
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

  // Dispatch over cells (only grid pressure update)
  vkCmdDispatch(frameInfo.commandBuffer, groupsX_cells, 1, 1);

  // Barrier: pass4 writes gridVel -> pass5 (g2p) reads gridVel & prevGridVel
  std::vector<VkBufferMemoryBarrier> pass4Barriers;
  pass4Barriers.push_back(makeBufferBarrier(
      gridVelBuf,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));
  // prevGridVel is read by g2p as previous state (ensure it's visible)
  pass4Barriers.push_back(makeBufferBarrier(
      prevGridVelBuf,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));
  vkCmdPipelineBarrier(
      frameInfo.commandBuffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      0,
      nullptr,
      static_cast<uint32_t>(pass4Barriers.size()),
      pass4Barriers.data(),
      0,
      nullptr);

  // ---------------- PASS 5: G2P (PARTICLES only) ----------------
  pc.uPass = 5;
  vkCmdPushConstants(
      frameInfo.commandBuffer,
      pipelineLayout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(WaterPushConstants),
      &pc);

  // Dispatch over particles: this writes partVelBuf and colorsBuf
  vkCmdDispatch(frameInfo.commandBuffer, groupsX_particles, 1, 1);

  // Barrier: pass5 writes partVel (and colors) -> pass6 (integrate) reads partVel and partPos
  std::vector<VkBufferMemoryBarrier> pass5Barriers;
  pass5Barriers.push_back(makeBufferBarrier(
      partVelBuf,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));
  pass5Barriers.push_back(makeBufferBarrier(
      colorsBuf,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));
  vkCmdPipelineBarrier(
      frameInfo.commandBuffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      0,
      nullptr,
      static_cast<uint32_t>(pass5Barriers.size()),
      pass5Barriers.data(),
      0,
      nullptr);

  // ---------------- PASS 6: Integrate particles (update positions, preserve radius)
  // ----------------
  pc.uPass = 6;
  vkCmdPushConstants(
      frameInfo.commandBuffer,
      pipelineLayout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(WaterPushConstants),
      &pc);

  uint32_t groupsX_integrate = (nParticles + localSizeX - 1) / localSizeX;
  vkCmdDispatch(frameInfo.commandBuffer, groupsX_integrate, 1, 1);

  // --- Make compute writes visible to the vertex input (so the renderer can read positions/colors)
  std::vector<VkBufferMemoryBarrier> computeToVertexBarriers;
  computeToVertexBarriers.push_back(makeBufferBarrier(
      partPosBuf,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT));
  computeToVertexBarriers.push_back(makeBufferBarrier(
      partVelBuf,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT));
  computeToVertexBarriers.push_back(makeBufferBarrier(
      colorsBuf,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT));
  // If you render using gridVel or other buffers, add them here too.

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
  finalBarriers.push_back(
      makeBufferBarrier(partPosBuf, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT));
  finalBarriers.push_back(
      makeBufferBarrier(partVelBuf, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT));
  // If you need to read pressures/grid state on host:
  finalBarriers.push_back(
      makeBufferBarrier(pressureReadBuf, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT));
  finalBarriers.push_back(
      makeBufferBarrier(pressureWriteBuf, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT));
  finalBarriers.push_back(
      makeBufferBarrier(gridVelBuf, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT));
  finalBarriers.push_back(
      makeBufferBarrier(gridFlagsBuf, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT));

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
    return std::make_unique<LveBuffer>(
        device,
        elemSize,
        count,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  };

  // =====================================================
  // PARTICLE BUFFERS
  // =====================================================
  partPosBuff = makeDeviceLocal_Output(sizeof(glm::vec4), particleCount);  // binding 0
  colorsBuff = makeDeviceLocal_Output(sizeof(glm::vec4), particleCount);  // binding 0
  debugBuff = std::make_unique<LveBuffer>(
      device,
      sizeof(float),
      particleCount,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  debugBuff->map();  // Keep it mapped permanently

  partVelBuff = makeDeviceLocal_Output(sizeof(glm::vec4), particleCount);  // binding 1

  // =====================================================
  // GRID BUFFERS
  // =====================================================
  gridVelAccumBuff = makeDeviceLocal(sizeof(uint32_t), grid.numCells * 3);  // binding 2
  gridWeightBuff = makeDeviceLocal(sizeof(uint32_t), grid.numCells);    // binding 3
  gridFlagsBuff = makeDeviceLocal(sizeof(uint32_t), grid.numCells);     // binding 4
  gridVelBuff = makeDeviceLocal(sizeof(glm::vec4), grid.numCells);      // binding 5
  prevGridVelBuff = makeDeviceLocal(sizeof(glm::vec4), grid.numCells);  // binding 6

  // =====================================================
  // PRESSURE BUFFERS (PING-PONG)
  // =====================================================
  pressureReadBuff = makeDeviceLocal_Output(sizeof(float), grid.numCells);  // binding 7
  pressureWriteBuff = makeDeviceLocal_Output(sizeof(float), grid.numCells);  // binding 8

  
  std::vector<glm::vec4> zeroVelocities(particleCount, glm::vec4(0.0f));
  std::unique_ptr<LveBuffer> velStaging = makeHostVisible(sizeof(glm::vec4), particleCount);
  velStaging->writeToBuffer(zeroVelocities.data());
  velStaging->flush();
  device.copyBuffer(
      velStaging->getBuffer(),
      partVelBuff->getBuffer(),
      sizeof(glm::vec4) * particleCount);
  std::vector<float> zeroVelocities2(grid.numCells, 0.f);
  std::unique_ptr<LveBuffer> pressureRead = makeHostVisible(sizeof(float), grid.numCells);
  pressureRead->writeToBuffer(zeroVelocities2.data());
  pressureRead->flush();
  device.copyBuffer(
      pressureRead->getBuffer(),
      pressureReadBuff->getBuffer(),
      sizeof(float) * grid.numCells);

  std::unique_ptr<LveBuffer> pressureWrite = makeHostVisible(sizeof(float), grid.numCells);
  pressureWrite->writeToBuffer(zeroVelocities2.data());
  pressureWrite->flush();
  device.copyBuffer(
      pressureWrite->getBuffer(),
      pressureWriteBuff->getBuffer(),
      sizeof(float) * grid.numCells);
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
      "tut27\\shaders\\water_grid.comp.spv",
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