#pragma once

#include <glm.hpp>
#include <unordered_map>
#include <vector>

#include "lve_game_object.hpp"  // adjust include to your project

namespace lve {

class WaterPhysics {
 public:
  // now takes an unordered_map of game objects (e.g. frameInfo.gameObjects)
  WaterPhysics(
      std::unordered_map<uint32_t, LveGameObject>& curParticles,
      float _smoothingRadius,
      float _restDensity,
      float _viscosity,
      float _mu);

  void RunSimulation(float dt);

 private:
  // reference to the map owned by the caller (frameInfo.gameObjects)
  std::unordered_map<uint32_t, LveGameObject>& particlesMap;

  // working list of pointers to active particles (filtered each step)
  std::vector<LveGameObject*> activeParticles;

  float smoothingRadius;
  float restDensity;
  float viscosity;
  float mu;

  // kernel helpers
  float SmoothingFunction(float x, float h);
  glm::vec3 grad_W_spiky(const glm::vec3& r_vec, double h);
  float laplacian_W_viscosity(float r, float h);

  // physics steps
  void ComputeDensities();
  void ComputePressures();
  void ComputeForces();
  void UpdateParticles(float dt);

  // small epsilon to avoid divide-by-zero
  const float eps = 1e-6f;
};

}  // namespace lve
