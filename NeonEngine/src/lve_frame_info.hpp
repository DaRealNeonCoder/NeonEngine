#pragma once

#include "lve_camera.hpp"
#include "lve_game_object.hpp"

// lib
#include <vulkan.h>

namespace lve {

#define MAX_LIGHTS 1

struct PointLight {
  glm::vec4 position{};  // ignore w
  glm::vec4 color{};     // w is intensity
};

struct GlobalUbo {
  glm::mat4 projection{1.f};
  glm::mat4 inverseProjection{1.f};
  glm::mat4 view{1.f};
  glm::mat4 inverseView{1.f};
  glm::vec4 ambientLightColor{1.f, 1.f, 1.f, .02f};  // w is intensity
  PointLight pointLights[MAX_LIGHTS];
  glm::vec4 numLights;

  glm::mat4 lightSpaceMatrix;  
  glm::vec4 lightDir; 
};
struct WaterUbo {
  glm::mat4 projection{1.f};
  glm::mat4 inverseProjection{1.f};
  glm::mat4 view{1.f};
  glm::mat4 inverseView{1.f};
};
struct alignas(16) WaterPhysUbo {
  int32_t uNumParticles;  // 4
  int32_t uNumCells;      // 4
  int32_t pad0;           // 4
  int32_t pad1;           // 4  -> first 16 bytes done

  glm::ivec4 uGridDim;  // 16 bytes

  float uH;              // 4
  float uH2;             // 4
  float overRelaxation;  // 4
  float spikyGradCoeff;  // 4

  float viscLapCoeff;  // 4
  float uMass;         // 4
  float uRestDensity;  // 4
  float uMu;           // 4

  float uViscosity;  // 4
  float uEps;        // 4
  float uDt;         // 4
  float uCellSize;   // 4

  glm::vec4 uBoxMin;   // 16
  glm::vec4 uBoxMax;   // 16
  glm::vec4 uGravity;  // 16

  float uDamping;  // 4
  float pad3;      // 4
  float pad4;      // 4
  float pad5;      // 4
};

//wrk
struct RestirPathTracerParams
{
	glm::uvec2 frameDim;
	uint32_t seed;
};


struct ReSTIRUbo
{
    // Base params
    RestirPathTracerParams params;

    // Spatial
    int   gSpatialRoundId;
    int   gNumSpatialRounds;
    uint32_t  gSpatialReusePattern;
    int   gNeighborCount;

    float gGatherRadius;
    int   gSmallWindowRadius;
    int   gFeatureBasedRejection;
    int   gIsLastRound;

    // Temporal
    int   gEnableTemporalReprojection;
    float gTemporalHistoryLength;
    int   gNoResamplingForTemporalReuse;
    float _pad0;

    // Camera / geometry
    glm::vec3  posW;
    float _pad1;
    glm::vec3  cameraW;
    float _pad2;

    float nearZ;
    float farZ;

    // Lighting / sampling
    int   useDirectLighting;
    int   useTalbotMIS;
    int   restirMISkind;
    int   pathSamplingMode;

};


struct RayUbo {
	glm::mat4 projection{1.f};
	glm::mat4 view{1.f};
	glm::mat4 inverseProjection{1.f};
	glm::mat4 inverseView{1.f};

	glm::mat4 prevView{1.f};

	uint32_t lightNum;
};


struct FrameInfo {
  int frameIndex;
  float frameTime;
  VkCommandBuffer commandBuffer;
  LveCamera &camera;
  VkDescriptorSet globalDescriptorSet;
  LveGameObject::Map &gameObjects;
};


struct ReSTIRFrameInfo {
	int frameIndex;
	float frameTime;
	VkCommandBuffer commandBuffer;
	LveCamera& camera;
	VkDescriptorSet globalDescriptorSet;
	LveGameObject::Map& gameObjects;
	uint32_t width;
	uint32_t height;
};

struct RayFrameInfo {
	int frameIndex;
	float frameTime;
	VkCommandBuffer commandBuffer;
	LveCamera& camera;
	VkDescriptorSet globalDescriptorSet;
	LveGameObject::Map& gameObjects;
};

struct WaterFrameInfo {
  float frameTime;
  VkCommandBuffer commandBuffer;
  LveCamera &camera;
  VkDescriptorSet globalDescriptorSet;
  VkDescriptorSet computeDescriptorSetPing;
  VkDescriptorSet computeDescriptorSetPong;
  LveGameObject::Map &gameObjects;
};
}  // namespace lve
