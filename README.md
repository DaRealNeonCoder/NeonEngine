# Neon Engine

## Path Tracer
Implemented physically accurate light simulation with support for multiple material types:

- Specular (mirror reflections)  
- Diffuse (matte surfaces)  
- Glossy (rough reflections)  

![Path Tracer 1](NeonEngine/gitImages/ray0.png)  
![Path Tracer 2](./gitImages/ray1.png)  
![Path Tracer 3](./gitImages/ray2.png)  
![Path Tracer 4](./gitImages/ray3.png)

---

## Rasterizer
Developed a custom graphics pipeline for real-time rendering:

- Texture mapping  
- Shadow rendering  

![Rasterizer](./gitImages/rast.png)

---

## Water Simulation (In Progress)
Currently developing a hybrid Euler–Lagrangian water simulation:

- Using FLIP (Fluid-Implicit Particle) to optimize performance  
- Hybrid grid + particle-based simulation  
- Surface generated via ray marching  
- Rendered through ray tracing  

![Water Simulation](./gitImages/wat.png)
