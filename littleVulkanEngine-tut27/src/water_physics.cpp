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
  

  float scale = 1.1f;
  float scaleX = 3.2f * scale;
  float scaleY = 1.f * scale;
  float scaleZ = 1.2f * scale;

  boxMin.x *= scaleX;
  boxMax.x *= scaleX;
  
  boxMin.y *= scaleY * 2.f;
  boxMax.y *= scaleY;

  boxMin.z *= scaleZ;
  boxMax.z *= scaleZ;

  // nothing else here
}

// Run one simulation step
void WaterPhysics::RunSimulation(float dt) {
  // rebuild the activeParticles list each step, filtering out objects we don't want
  activeParticles.clear();
  for (auto& kv : particlesMap) {
    LveGameObject& obj = kv.second;
    // skip if explicitly asked (id == 0) or when model is null (similar to render pass)
    if (obj.getId() == 0) continue;
    if (obj.model == nullptr) continue;
    activeParticles.push_back(&obj);
  }

  // if there are fewer than 1 particle, nothing to do
  if (activeParticles.size() < 1) return;

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
    LveGameObject* pi = activeParticles[i];
    //float rho_i = pi->mass * SmoothingFunction(0.0f, smoothingRadius);

    int neighborCount = 0;  // Debug

    for (size_t j = 0; j < N; ++j) {
      if (i == j) continue;
     
      LveGameObject* pj = activeParticles[j];
      float dist = glm::distance(pi->transform.translation, pj->transform.translation);
      
      if (dist < smoothingRadius) {
        
        pi-> density += pj->mass * SmoothingFunction(dist, smoothingRadius);
        neighborCount++;
      
      }
    }

    pi->density = glm::max(pi->density, eps);

    // Debug first particle
    if (i == 0) {
      std::cout << "Neighbors: " << neighborCount << " density: " << pi->density << std::endl;
    }
  }
}

void WaterPhysics::ComputePressures() {
  for (LveGameObject* p : activeParticles) {
    // Tait equation of state
    float gamma = 7.0f;
    float error = p->density - restDensity;
    p->pressure = error * mu;
    //p->pressure = mu * (std::pow(p->density / restDensity, gamma) - 1.0f);
  }
}


// compute acceleration contributions (rename conceptually from 'force' to 'accel')
void WaterPhysics::ComputeForces() {
  const size_t N = activeParticles.size();
  for (size_t i = 0; i < N; ++i) {
    LveGameObject* pi = activeParticles[i];

    glm::vec3 a_pressure(0.0f);
    glm::vec3 a_viscosity(0.0f);

    for (size_t j = 0; j < N; ++j) {
      if (i == j) continue;
      LveGameObject* pj = activeParticles[j];

      glm::vec3 r_vec = pi->transform.translation - pj->transform.translation;
      float r = glm::length(r_vec);

      if (r < smoothingRadius && r > eps) {
        glm::vec3 grad =
            grad_W_spiky(r_vec, smoothingRadius);  // returns vector already including /r
        // --- pressure acceleration contribution (do NOT multiply by pi->mass) ---
      
        a_pressure += -pj->mass *
                      (pi->pressure / (pi->density * pi->density) +
                       pj->pressure / (pj->density * pj->density)) *
                      grad;

        // --- viscosity acceleration contribution ---
        a_viscosity += viscosity * pj->mass * (pj->velocity - pi->velocity) / pj->density *
                       laplacian_W_viscosity(r, smoothingRadius);
      }
    }

    // gravity as acceleration (downwards)
    glm::vec3 a_gravity(0.0f, 15.f, 0.0f);

    // store acceleration (not force). You can rename field or keep p->force but treat it as accel.
    pi->force = a_pressure + a_viscosity + a_gravity;
  }
}

void WaterPhysics::UpdateParticles(float dt) {
  for (LveGameObject* p : activeParticles) {
    // 'p->force' now holds acceleration (m/s^2). Integrate to velocity/position.
    p->velocity += dt * p->force;
    p->transform.translation += dt * p->velocity;

    // simple boundary containment (same as yours)
    if (p->transform.translation.x < boxMin.x) {
      p->transform.translation.x = boxMin.x;
      p->velocity.x *= -0.5f;
    }
    if (p->transform.translation.x > boxMax.x) {
      p->transform.translation.x = boxMax.x;
      p->velocity.x *= -0.5f;
    }
    if (p->transform.translation.y < boxMin.y) {
      p->transform.translation.y = boxMin.y;
      p->velocity.y *= -0.5f;
    }
    if (p->transform.translation.y > boxMax.y) {
      p->transform.translation.y = boxMax.y;
      p->velocity.y *= -0.5f;
    }
    if (p->transform.translation.z < boxMin.z) {
      p->transform.translation.z = boxMin.z;
      p->velocity.z *= -0.5f;
    }
    if (p->transform.translation.z > boxMax.z) {
      p->transform.translation.z = boxMax.z;
      p->velocity.z *= -0.5f;
    }
  }
}

}  // namespace lve
