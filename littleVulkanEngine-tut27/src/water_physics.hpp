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
    VkDescriptorBufferInfo getPartPosDescInfo()       { return partPosBuff->descriptorInfo(); }         // binding 0
    VkDescriptorBufferInfo getPartVelDescInfo()       { return partVelBuff->descriptorInfo(); }         // binding 1
    VkDescriptorBufferInfo getGridVelAccumDescInfo()  { return gridVelAccumBuff->descriptorInfo(); }    // binding 2
    VkDescriptorBufferInfo getGridWeightDescInfo()    { return gridWeightBuff->descriptorInfo(); }      // binding 3
    VkDescriptorBufferInfo getGridFlagsDescInfo()     { return gridFlagsBuff->descriptorInfo(); }       // binding 4
    VkDescriptorBufferInfo getGridVelDescInfo()       { return gridVelBuff->descriptorInfo(); }         // binding 5
    VkDescriptorBufferInfo getPrevGridVelDescInfo()   { return prevGridVelBuff->descriptorInfo(); }     // binding 6
    VkDescriptorBufferInfo getPressureReadDescInfo()  { return pressureReadBuff->descriptorInfo(); }    // binding 7
    VkDescriptorBufferInfo getPressureWriteDescInfo() { return pressureWriteBuff->descriptorInfo(); }   // binding 8
    VkDescriptorBufferInfo getColorDescInfo()         { return colorsBuff->descriptorInfo(); }
    VkDescriptorBufferInfo getDebugInfo() { return debugBuff->descriptorInfo(); }
  
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
  std::unique_ptr<LveBuffer> partVelBuff;     
  std::unique_ptr<LveBuffer> gridVelAccumBuff;  
  std::unique_ptr<LveBuffer> gridWeightBuff;     
  std::unique_ptr<LveBuffer> gridFlagsBuff;      
  std::unique_ptr<LveBuffer> gridVelBuff;       
  std::unique_ptr<LveBuffer> prevGridVelBuff;    
  std::unique_ptr<LveBuffer> pressureReadBuff;   
  std::unique_ptr<LveBuffer> pressureWriteBuff;  


  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkPipeline computePipeline = VK_NULL_HANDLE;
  VkShaderModule computeShaderModule = VK_NULL_HANDLE;
  Grid mainGrid;
};

}  // namespace lve