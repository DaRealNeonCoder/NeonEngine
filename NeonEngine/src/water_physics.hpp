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
    int uPass;
    int pad;
    int pad2;
    int pad3;

    float dt;
  };
 public:
  WaterPhysics(
      int _particleCount,
      std::vector<glm::vec4>& startPos,
      VkDescriptorSetLayout setLayout,
      WaterPhysUbo& ubo,
      std::unordered_map<unsigned int, LveGameObject>& curParticles,
      float _smoothingRadius,
      float _restDensity,
      float _viscosity,
      float _mu,
      LveDevice* _device);
  ~WaterPhysics();

  void RunSimulation(float dt, WaterFrameInfo &info);
// Descriptor Info Getters - add these to your header
VkDescriptorBufferInfo getPartPosDescInfo()      { return partPosBuff->descriptorInfo(); }       // binding 0
VkDescriptorBufferInfo getPartVelDescInfo()      { return partVelBuff->descriptorInfo(); }       // binding 1
VkDescriptorBufferInfo getGridUDescInfo()        { return gridUBuff->descriptorInfo(); }         // binding 2
VkDescriptorBufferInfo getGridVDescInfo()        { return gridVBuff->descriptorInfo(); }         // binding 3
VkDescriptorBufferInfo getGridWDescInfo()        { return gridWBuff->descriptorInfo(); }         // binding 4
VkDescriptorBufferInfo getPrevGridUDescInfo()    { return prevGridUBuff->descriptorInfo(); }     // binding 5
VkDescriptorBufferInfo getPrevGridVDescInfo()    { return prevGridVBuff->descriptorInfo(); }     // binding 6
VkDescriptorBufferInfo getPrevGridWDescInfo()    { return prevGridWBuff->descriptorInfo(); }     // binding 7
VkDescriptorBufferInfo getGridFlagsDescInfo()    { return gridFlagsBuff->descriptorInfo(); }     // binding 8
VkDescriptorBufferInfo getColorDescInfo()        { return colorsBuff->descriptorInfo(); }        // binding 9
VkDescriptorBufferInfo getDebugInfo()            { return debugBuff->descriptorInfo(); }         // binding 11
VkDescriptorBufferInfo getGridDUDescInfo()       { return gridDUBuff->descriptorInfo(); }        // binding 12
VkDescriptorBufferInfo getGridDVDescInfo()       { return gridDVBuff->descriptorInfo(); }        // binding 13
VkDescriptorBufferInfo getGridDWDescInfo()       { return gridDWBuff->descriptorInfo(); }        // binding 14
VkDescriptorBufferInfo getGridSDescInfo()        { return gridSBuff->descriptorInfo(); }         // binding 15
VkDescriptorBufferInfo getPressureReadDescInfo() { return pressureReadBuff->descriptorInfo(); }  // binding 16
VkDescriptorBufferInfo getPressureWriteDescInfo(){ return pressureWriteBuff->descriptorInfo(); } // binding 17
VkDescriptorBufferInfo getGridUAccumDescInfo() { return gridUAccumBuff->descriptorInfo(); }  // 18
VkDescriptorBufferInfo getGridVAccumDescInfo() { return gridVAccumBuff->descriptorInfo(); }  // 19
VkDescriptorBufferInfo getGridWAccumDescInfo() { return gridWAccumBuff->descriptorInfo(); }  // 20

VkDescriptorBufferInfo getGridUWeightDescInfo() { return gridUWeightBuff->descriptorInfo(); }  // 21
VkDescriptorBufferInfo getGridVWeightDescInfo() { return gridVWeightBuff->descriptorInfo(); }  // 22
VkDescriptorBufferInfo getGridWWeightDescInfo() { return gridWWeightBuff->descriptorInfo(); }  // 23

    std::unique_ptr<LveBuffer> makeHostVisible(VkDeviceSize elemSize, uint32_t count); 
  void UploadBuffers(const Grid& grid);
  std::vector<glm::vec4> outPositions;
  std::unique_ptr<LveBuffer> outputBuffer;
  std::unique_ptr<LveBuffer> partPosBuff;
  std::unique_ptr<LveBuffer> colorsBuff;
  std::unique_ptr<LveBuffer> debugBuff;
  std::unique_ptr<LveBuffer> debugStaging;
    

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
  // Particle buffers
  std::unique_ptr<LveBuffer> partVelBuff;  // binding 1

  // MAC grid velocity buffers (separate components)
  std::unique_ptr<LveBuffer> gridUBuff;  // binding 2 - u velocity at vertical faces
  std::unique_ptr<LveBuffer> gridVBuff;  // binding 3 - v velocity at horizontal faces
  std::unique_ptr<LveBuffer> gridWBuff;  // binding 4 - w velocity at depth faces

  std::unique_ptr<LveBuffer> prevGridUBuff;  // binding 5 - previous u
  std::unique_ptr<LveBuffer> prevGridVBuff;  // binding 6 - previous v
  std::unique_ptr<LveBuffer> prevGridWBuff;  // binding 7 - previous w

  // Grid metadata
  std::unique_ptr<LveBuffer> gridFlagsBuff;  // binding 8 - cell type flags

  // Weight accumulators for P2G (stored as uint for atomic ops)
  std::unique_ptr<LveBuffer> gridDUBuff;  // binding 12
  std::unique_ptr<LveBuffer> gridDVBuff;  // binding 13
  std::unique_ptr<LveBuffer> gridDWBuff;  // binding 14

  // Solid flags (s array from JS: 0=solid, 1=fluid)
  std::unique_ptr<LveBuffer> gridSBuff;  // binding 15

  // Pressure buffers (ping-pong)
  std::unique_ptr<LveBuffer> pressureReadBuff;   // binding 16
  std::unique_ptr<LveBuffer> pressureWriteBuff;  // binding 17

  std::unique_ptr<LveBuffer> gridUAccumBuff;  // binding 18
  std::unique_ptr<LveBuffer> gridVAccumBuff;  // binding 19
  std::unique_ptr<LveBuffer> gridWAccumBuff;  // binding 20

  // Pure weight accumulators
  std::unique_ptr<LveBuffer> gridUWeightBuff;  // binding 21
  std::unique_ptr<LveBuffer> gridVWeightBuff;  // binding 22
  std::unique_ptr<LveBuffer> gridWWeightBuff;  // binding 23
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkPipeline computePipeline = VK_NULL_HANDLE;
  VkShaderModule computeShaderModule = VK_NULL_HANDLE;
  Grid mainGrid;
};

}  // namespace lve