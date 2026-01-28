#version 450


layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec2 uv;

//some of these are useless. should really clean this.
layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragPosWorld;
layout(location = 2) out vec3 fragNormalWorld;

struct PointLight {
vec4 position; // ignore w
vec4 color; // w is intensity
};


layout(set = 0, binding = 0, std140) uniform WaterUbo {
    mat4 projection;
    mat4 inverseProjection;
    mat4 view;
    mat4 inverseView;
} ubo;


layout(push_constant) uniform Push {
mat4 modelMatrix;
mat4 normalMatrix;
vec4 thisIsStupid;
} push;


out gl_PerVertex {
    vec4 gl_Position;
};


void main() 
{
    vec4 worldPos = push.modelMatrix * vec4(position, 1.0);
    fragPosWorld = worldPos.xyz;
    fragNormalWorld = normalize(mat3(push.normalMatrix) * normal);
    fragColor = color;
    
    gl_Position = ubo.projection * ubo.view *  push.modelMatrix * vec4(position, 1.0);
}