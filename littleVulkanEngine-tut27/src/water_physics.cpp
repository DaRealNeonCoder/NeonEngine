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
    VkDescriptorSetLayout setLayout, 
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

  p_velocities.resize(curParticles.size());
  p_positions.resize(curParticles.size());
  p_forces.resize(curParticles.size());
  p_pressures.resize(curParticles.size());
  p_densities.resize(curParticles.size());

  // Scale bounding box
  float scale = 0.5f;
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

  h = _smoothingRadius;
  h2 = h * h;
  h6 = h2 * h2 * h2;
  h9 = h6 * h2 * h;

  poly6Coeff = 315.0f * invPi / (64.0f * h9);
  spikyGradCoeff = -45.0f * invPi / h6;
  viscLapCoeff = 45.0f * invPi / h6;

  // Filter active particles
  for (auto& kv : particlesMap) {
    LveGameObject& obj = kv.second;
    if (obj.getId() == 0) continue;
    if (obj.model == nullptr) continue;
    activeParticles.push_back(&obj);
  }

  std::cout << "Active particles: " << activeParticles.size() << std::endl;

  // Initialize positions
  for (size_t i = 0; i < activeParticles.size(); i++) {
    p_positions[i] = activeParticles[i]->transform.translation;
  }

  std::vector<glm::vec4> temp;
  temp.reserve(p_positions.size());

  for (const auto& v : p_positions) {
    temp.emplace_back(v, 1.0f);
  }

  particleCount = temp.size();

  CreateBuffers();
  UploadBuffers(temp);
  CreateComputePipelineLayout(setLayout);
  CreateComputePipeline();

  std::cout << "\n"<< "done";
}

WaterPhysics::~WaterPhysics() { 
    vkDestroyPipeline(device.device(), computePipeline, nullptr);
  vkDestroyPipelineLayout(device.device(), pipelineLayout, nullptr);
  vkDestroyShaderModule(device.device(), computeShaderModule, nullptr);
}

void WaterPhysics::RunSimulation(float dt) {
  BuildSpatialGrid();
  ComputeDensities();
  ComputePressures();
  ComputeForces();
  UpdateParticles(dt);
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

// ============================================================================
// Kernel Functions
// ============================================================================

float WaterPhysics::SmoothingFunction(float r) {
  if (r >= 0.0f && r <= h) {
    float h2_minus_r2 = h2 - r * r;
    float x = h2_minus_r2 * h2_minus_r2 * h2_minus_r2;  
    return poly6Coeff * x;
  }
  return 0.0f;
}
glm::vec3 WaterPhysics::grad_W_spiky(const glm::vec3& r_vec) {
  float r = glm::length(r_vec);

  if (r > 0.0f && r <= h) {
    float hr = h - r;
    float scalar = spikyGradCoeff * (hr * hr) / r;
    return scalar * r_vec;
  }

  return glm::vec3(0.0f);
}
float WaterPhysics::laplacian_W_viscosity(float r) {
  if (r >= 0.0f && r <= h) {
    return viscLapCoeff * (h - r);
  }
  return 0.0f;
}

void WaterPhysics::ComputeDensities() {
  const size_t N = activeParticles.size();
  std::vector<GridCell> neighborCells;
  neighborCells.reserve(27);

  for (size_t i = 0; i < N; ++i) {
    glm::vec3 xi = p_positions[i];
    float rho = 0.0f;

    // Get grid cell for particle i
    GridCell cell = GetGridCell(xi);
    GetNeighborCells(cell, neighborCells);

    // Only check particles in neighboring cells
    for (const GridCell& neighborCell : neighborCells) {
      auto it = spatialGrid.find(neighborCell);
      if (it == spatialGrid.end()) continue;

      for (size_t j : it->second) {
        glm::vec3 xj = p_positions[j];
        glm::vec3 diff = xi - p_positions[j];
        float r2 = glm::dot(diff, diff);  // Squared distance
        if (r2 <= h2) {
          rho += mass * SmoothingFunction(glm::sqrt(r2));
        }
      }
    }

    p_densities[i] = glm::max(rho, eps);
  }
}

void WaterPhysics::ComputePressures() {
  for (size_t i = 0; i < p_densities.size(); ++i) {
    float error = p_densities[i] - restDensity;
    p_pressures[i] = error * mu;
  }
}

void WaterPhysics::ComputeForces() {
  const size_t N = activeParticles.size();
  std::vector<GridCell> neighborCells;
  neighborCells.reserve(27);

  for (size_t i = 0; i < N; ++i) {
    glm::vec3 ai(0.0f);
    glm::vec3 xi = p_positions[i];
    glm::vec3 vi = p_velocities[i];

    // Get grid cell for particle i
    GridCell cell = GetGridCell(xi);
    GetNeighborCells(cell, neighborCells);

    // Only check particles in neighboring cells
    for (const GridCell& neighborCell : neighborCells) {
      auto it = spatialGrid.find(neighborCell);
      if (it == spatialGrid.end()) continue;

      const std::vector<size_t>& particlesInCell = it->second;
      for (size_t j : particlesInCell) {
        if (i == j) continue;

        glm::vec3 xj = p_positions[j];
        glm::vec3 vj = p_velocities[j];
        glm::vec3 r_vec = xi - xj;
        float r = glm::length(r_vec);

        if (r > eps && r < smoothingRadius) {
          glm::vec3 grad = grad_W_spiky(r_vec);

          // Pressure force
          ai += -mass *
                (p_pressures[i] / (p_densities[i] * p_densities[i]) +
                 p_pressures[j] / (p_densities[j] * p_densities[j])) *
                grad;

          // Viscosity force
          ai += viscosity * mass * (vj - vi) / p_densities[j] *
                laplacian_W_viscosity(r);
        }
      }
    }

    // Gravity
    ai += glm::vec3(0.0f, 15.0f, 0.0f);
    p_forces[i] = ai;
  }
}

void WaterPhysics::UpdateParticles(float dt) {
  const size_t N = activeParticles.size();

  for (size_t i = 0; i < N; ++i) {
    // Integrate velocity
    p_velocities[i] += dt * p_forces[i];

    // Integrate position
    LveGameObject* p = activeParticles[i];
    glm::vec3& x = p_positions[i];
    glm::vec3& v = p_velocities[i];

    x += dt * v;

    // Boundary containment with damping
    const float damping = 0.5f;

    if (x.x < boxMin.x) {
      x.x = boxMin.x;
      v.x *= -damping;
    } else if (x.x > boxMax.x) {
      x.x = boxMax.x;
      v.x *= -damping;
    }

    if (x.y < boxMin.y) {
      x.y = boxMin.y;
      v.y *= -damping;
    } else if (x.y > boxMax.y) {
      x.y = boxMax.y;
      v.y *= -damping;
    }

    if (x.z < boxMin.z) {
      x.z = boxMin.z;
      v.z *= -damping;
    } else if (x.z > boxMax.z) {
      x.z = boxMax.z;
      v.z *= -damping;
    }

    // Update game object transform
    p->transform.translation = x;
  }
}


void WaterPhysics::UploadBuffers(const std::vector<glm::vec4>& positions) {
  inputBuffer->writeToBuffer((void*)positions.data());
  inputBuffer->flush();
  // keep mapped or unmap if your LveBuffer's writeToBuffer already unmaps
  inputBuffer->unmap();
}

// Dispatch compute and read back output into outPositions vector
void WaterPhysics::RunAndReadback(
    std::vector<glm::vec4>& outPositions, VkDescriptorSet& descriptorSet) {
  VkCommandBuffer cmd = device.beginSingleTimeCommands();

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
  vkCmdBindDescriptorSets(
      cmd,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      pipelineLayout,
      0,
      1,
      &descriptorSet,
      0,
      nullptr);

  uint32_t groupSize = 64;
  uint32_t groups = (particleCount + groupSize - 1) / groupSize;
  vkCmdDispatch(cmd, groups, 1, 1);

  VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  vkCreateFence(device.device(), &fenceInfo, nullptr, &fence);

  VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmd;

  vkQueueSubmit(computeQueue, 1, &submitInfo, fence);
  vkWaitForFences(device.device(), 1, &fence, VK_TRUE, UINT64_MAX);

  vkDestroyFence(device.device(), fence, nullptr);
 
   device.endSingleTimeCommands(cmd);

  // Map the output buffer and copy results to CPU vector
  outPositions.resize(particleCount);
  outputBuffer->map();
  void* mapped = outputBuffer->getMappedMemory();  
                                                  
  memcpy(outPositions.data(), mapped, sizeof(glm::vec4) * particleCount);
  outputBuffer->unmap();
}


void WaterPhysics::CreateBuffers() {
  // Input buffer: HOST_VISIBLE so we can write directly
  inputBuffer = std::make_unique<LveBuffer>(
      device,
      sizeof(glm::vec4),
      particleCount,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  inputBuffer->map();

  // Output buffer: HOST_VISIBLE so we can read back directly
  outputBuffer = std::make_unique<LveBuffer>(
      device,
      sizeof(glm::vec4),
      particleCount,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  outputBuffer->map();
}
void WaterPhysics::CreateComputePipelineLayout(VkDescriptorSetLayout setLayout) {
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};

    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &setLayout;
    layoutInfo.pushConstantRangeCount = 0;

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
};

