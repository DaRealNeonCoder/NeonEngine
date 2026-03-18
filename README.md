# Neon Engine

## Path Tracer
Implemented physically accurate light simulation with support for multiple material types:

- Specular (mirror reflections)  
- Diffuse (matte surfaces)  
- Glossy (rough reflections)  

<img src="NeonEngine/gitImages/ray0.png" width="300"/>
<img src="NeonEngine/gitImages/ray1.png" width="300"/>
<img src="NeonEngine/gitImages/ray2.png" width="300"/>
<img src="NeonEngine/gitImages/ray3.png" width="300"/>

---

## Rasterizer
Developed a custom graphics pipeline for real-time rendering:

- Texture mapping  
- Shadow rendering  

<img src="NeonEngine/gitImages/rast.png" width="400"/>

---

## Water Simulation (In Progress)
Currently developing a hybrid Euler–Lagrangian water simulation:

- Using FLIP (Fluid-Implicit Particle) to optimize performance  
- Hybrid grid + particle-based simulation  
- Surface generated via ray marching  
- Rendered through ray tracing  

<img src="NeonEngine/gitImages/wat.png" width="400"/>
