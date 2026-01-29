#include "water_physics.hpp"

// glm
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>

// std
#include <cmath>
#include <iostream>
#include <unordered_map>

namespace lve {
glm::vec3 boxMin(-1.4f, -1.4f, -1.4f);  // bottom-left-back corner
glm::vec3 boxMax(1.4f, 0.5, 1.4f);      // top-right-front corner
// Constructor — takes a reference to the unordered_map of game objects
WaterPhysics::WaterPhysics(
    std::unordered_map<uint32_t, LveGameObject>& curParticles,
    float _smoothingRadius,
    float _restDensity,
    float _viscosity,
    float _mu)
    : particlesMap(curParticles),
      smoothingRadius(_smoothingRadius),
      restDensity(_restDensity),
      viscosity(_viscosity),
      mu(_mu) {
  p_velocities.resize(curParticles.size());
  p_positions.resize(curParticles.size());
  p_forces.resize(curParticles.size());
  p_pressures.resize(curParticles.size());
  p_densities.resize(curParticles.size());

  float scale = 1.1f;//orignal = 1.1f
  float scaleX = 3.2f * scale;
  float scaleY = 1.f * scale;
  float scaleZ = 1.2f * scale;

  boxMin.x *= scaleX;
  boxMax.x *= scaleX;
  
  boxMin.y *= scaleY * 2.f;
  boxMax.y *= scaleY;

  boxMin.z *= scaleZ;
  boxMax.z *= scaleZ;
  std::cout << "workin";

  for (auto& kv : particlesMap) {
    LveGameObject& obj = kv.second;
    if (obj.getId() == 0) continue;
    if (obj.model == nullptr) continue;
    activeParticles.push_back(&obj);
  }

  std::cout << "also workin";

  for (size_t i = 0; i < activeParticles.size(); i++) {
    p_positions[i] = activeParticles[i]->transform.translation;
  }
  // nothing else here
}

// Run one simulation step
void WaterPhysics::RunSimulation(float dt) {
  // rebuild the activeParticles list each step, filtering out objects we don't want


  ComputeDensities();
  ComputePressures();
  ComputeForces();
  UpdateParticles(dt);
}

// Kernel functions
float WaterPhysics::SmoothingFunction(float x, float h) {
  if (x >= 0.0f && x <= h) {
    float h2_minus_x2 = h * h - x * x;
    // you used pi earlier; use glm::pi<float>() or define pi
    const float coeff = 315.0f / (64.0f * glm::pi<float>() * std::pow(h, 9.0f));
    return coeff * std::pow(h2_minus_x2, 3.0f);
  }
  return 0.0f;
}

glm::vec3 WaterPhysics::grad_W_spiky(const glm::vec3& r_vec, double h) {
  double r = glm::length(r_vec);
  if (r > 0.0 && r <= h) {
    double coeff = -45.0 / (glm::pi<double>() * std::pow(h, 6));
    // note we divide by r to get directional gradient
    double scalar = coeff * std::pow(h - r, 2) / r;
    return static_cast<float>(scalar) * r_vec;
  }
  return glm::vec3(0.0f);
}

float WaterPhysics::laplacian_W_viscosity(float r, float h) {
  if (r >= 0.0 && r <= h) {
    return 45.0f / (glm::pi<float>() * std::pow(h, 6)) * (h - r);
  }
  return 0.0f;
}
void WaterPhysics::ComputeDensities() {
  const size_t N = activeParticles.size();


  for (size_t i = 0; i < N; ++i) {
    glm::vec3 xi = p_positions[i];
    float rho = 0.0f;

    for (size_t j = 0; j < N; ++j) {
      glm::vec3 xj = p_positions[j];
      float r = glm::distance(xi, xj);

      if (r <= smoothingRadius) {
        rho += mass * SmoothingFunction(r, smoothingRadius);
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

  for (size_t i = 0; i < N; ++i) {
    glm::vec3 ai(0.0f);
    glm::vec3 xi = p_positions[i];
    glm::vec3 vi = p_velocities[i];

    for (size_t j = 0; j < N; ++j) {
      if (i == j) continue;

      glm::vec3 xj = p_positions[j];
      glm::vec3 vj = p_velocities[j];

      glm::vec3 r_vec = xi - xj;
      float r = glm::length(r_vec);

      if (r > eps && r < smoothingRadius) {
        glm::vec3 grad = grad_W_spiky(r_vec, smoothingRadius);

        // pressure
        ai += -mass *
              (p_pressures[i] / (p_densities[i] * p_densities[i]) +
               p_pressures[j] / (p_densities[j] * p_densities[j])) *
              grad;

        // viscosity
        ai += viscosity * mass * (vj - vi) / p_densities[j] *
              laplacian_W_viscosity(r, smoothingRadius);
      }
    }

    // gravity (acceleration)
    ai += glm::vec3(0.0f, 15.f, 0.0f);

    p_forces[i] = ai;
  }
}
void WaterPhysics::UpdateParticles(float dt) {
  const size_t N = activeParticles.size();

  for (size_t i = 0; i < N; ++i) {
    // integrate velocity (a = p_forces)
    p_velocities[i] += dt * p_forces[i];

    // integrate position (only thing written back to ECS object)
    LveGameObject* p = activeParticles[i];
    glm::vec3& x = p_positions[i];
    glm::vec3& v = p_velocities[i];

    x += dt * v;

    // ---- boundary containment ----

    if (x.x < boxMin.x) {
      x.x = boxMin.x;
      v.x *= -0.5f;
    } else if (x.x > boxMax.x) {
      x.x = boxMax.x;
      v.x *= -0.5f;
    }

    if (x.y < boxMin.y) {
      x.y = boxMin.y;
      v.y *= -0.5f;
    } else if (x.y > boxMax.y) {
      x.y = boxMax.y;
      v.y *= -0.5f;
    }

    if (x.z < boxMin.z) {
      x.z = boxMin.z;
      v.z *= -0.5f;
    } else if (x.z > boxMax.z) {
      x.z = boxMax.z;
      v.z *= -0.5f;
    }

    p->transform.translation = x;
  }
}

}  // namespace lve
