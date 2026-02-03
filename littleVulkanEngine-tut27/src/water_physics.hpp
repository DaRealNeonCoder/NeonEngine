#pragma once

#include <array>
#include <glm.hpp>
#include <unordered_map>
#include <vector>
#include "lve_game_object.hpp"
#include "lve_frame_info.hpp"

namespace lve {

// Hash function for grid cells
struct GridCell {
  int x, y, z;

  bool operator==(const GridCell& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct GridCellHash {
  std::size_t operator()(const GridCell& cell) const {
    // Simple hash combination
    std::size_t h1 = std::hash<int>{}(cell.x);
    std::size_t h2 = std::hash<int>{}(cell.y);
    std::size_t h3 = std::hash<int>{}(cell.z);
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};

class WaterPhysics {
  struct Grid {
    int numCells;
    glm::ivec3 gridDim;
    float cellSize;

    // Per-cell metadata
    std::vector<int> cellStart;  // size = numCells
    std::vector<int> cellCount;  // size = numCells

    // Packed particle indices
    std::vector<int> cellIndices;  // size = numParticles
  };
  struct WaterPushConstants {
    alignas(16) int uPass;
  };
 public:
  WaterPhysics(
      VkDescriptorSetLayout setLayout,
      WaterPhysUbo &ubo, 
      std::unordered_map<unsigned int, LveGameObject>& curParticles,
      float _smoothingRadius = 0.1f,
      float _restDensity = 1000.0f,
      float _viscosity = 0.01f,
      float _mu = 1000.0f, LveDevice *_device = nullptr);
  ~WaterPhysics();

  void RunSimulation(float dt, WaterFrameInfo &info);
  VkDescriptorBufferInfo getCellStartDescInfo() { return cellStart->descriptorInfo(); }
  VkDescriptorBufferInfo getCellIndiciesDescInfo() { return cellIndices->descriptorInfo(); }
  VkDescriptorBufferInfo getCellCountDescInfo() { return cellCount->descriptorInfo(); }
  VkDescriptorBufferInfo getVelocitiesDescInfo() { return velocitiesBuff->descriptorInfo(); }
  VkDescriptorBufferInfo getDensitiesDescInfo() { return densitiesBuff->descriptorInfo(); }
  VkDescriptorBufferInfo getPressuresDescInfo() { return pressuresBuff->descriptorInfo(); }
  VkDescriptorBufferInfo getForcesDescInfo() { return forcesBuff->descriptorInfo(); }
  VkDescriptorBufferInfo getOutputDescInfo() { return outputBuffer->descriptorInfo(); }
  VkDescriptorBufferInfo getCellCursorDescInfo() { return cellCursorBuff->descriptorInfo(); }

  void UploadBuffers(const Grid& grid);
 private:
  // Kernel functions
  float SmoothingFunction(float x);
  glm::vec3 grad_W_spiky(const glm::vec3& r_vec);
  float laplacian_W_viscosity(float r);

  // SPH computation steps
  void ComputeDensities();
  void ComputePressures();
  void ComputeForces();
  void UpdateParticles(float dt);
  void CreateBuffers(Grid& grid);
  void CreateComputePipeline();
  void CreateComputePipelineLayout(VkDescriptorSetLayout setLayout);
  void RunAndReadback(WaterFrameInfo &frameInfo);
  // Grid-based neighbor finding
  void BuildSpatialGrid();
  GridCell GetGridCell(const glm::vec3& position) const;
  void GetNeighborCells(const GridCell& cell, std::vector<GridCell>& neighbors) const;

  glm::ivec3 PositionToGridCell(const glm::vec3& pos, float cellSize, const glm::ivec3& gridDim);

  int Cell3DToIndex(const glm::ivec3& cell, const glm::ivec3& gridDim);

  void ClearGrid(Grid& grid);

  void CountParticlesPerCell(Grid& grid, const std::vector<glm::vec3>& positions);

  void BuildCellStart(Grid& grid);

  void FillCellIndices(Grid& grid, const std::vector<glm::vec3>& positions);

  void BuildUniformGrid(Grid& grid, const std::vector<glm::vec3>& positions);
  // Reference to the game objects map
  std::unordered_map<unsigned int, LveGameObject>& particlesMap;

  // Active particles (filtered)
  std::vector<LveGameObject*> activeParticles;

  // Per-particle data
  std::vector<glm::vec3> p_velocities;
  std::vector<glm::vec3> p_positions;
  std::vector<glm::vec3> p_forces;
  std::vector<glm::vec4> outPositions;
  std::vector<float> p_pressures;
  std::vector<float> p_densities;
  // Spatial grid for fast neighbor queries
  std::unordered_map<GridCell, std::vector<size_t>, GridCellHash> spatialGrid;
  float cellSize = 1.f;  // Should be >= smoothingRadius for optimal performance

  // SPH parameters
  float smoothingRadius;
  float restDensity;
  float viscosity;
  float mu;
  float poly6Coeff, spikyGradCoeff, viscLapCoeff;
  float h ;
  float h2;
  float h6 ;
  float h9 ;
  // Constants
  const float mass = 5.0f;
  const float eps = 1e-6f;
  std::vector<GridCell> particleNeighborCells;

  LveDevice& device;
  VkQueue computeQueue;

  uint32_t particleCount = 0;
  std::unique_ptr<LveBuffer> cellStart;
  std::unique_ptr<LveBuffer> cellCount;
  std::unique_ptr<LveBuffer> cellIndices;
  std::unique_ptr<LveBuffer> velocitiesBuff;
  std::unique_ptr<LveBuffer> densitiesBuff;
  std::unique_ptr<LveBuffer> pressuresBuff;
  std::unique_ptr<LveBuffer> forcesBuff;
  std::unique_ptr<LveBuffer> cellCursorBuff;

  std::unique_ptr<LveBuffer> outputBuffer;

  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkPipeline computePipeline = VK_NULL_HANDLE;
  VkShaderModule computeShaderModule = VK_NULL_HANDLE;
  Grid mainGrid;
};

}  // namespace lve