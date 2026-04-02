#version 450
// Vertex attributes
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec2 uv;


// Outputs to fragment shader
layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragPosWorld;
layout(location = 2) out vec3 fragNormalWorld;
layout(location = 3) out vec3 fragBarycentric;
layout(location = 4) out vec2 fragUV;
layout(location = 5) out vec4 fragClipPos;

// Push constants for per-object transform
layout(push_constant) uniform Push {
    mat4 modelMatrix;
    mat4 normalMatrix;
} push;
// Global UBO with camera matrices
layout(set = 0, binding = 0, std140) uniform GlobalUbo {
    mat4 projection;
    mat4 view;
    mat4 inverseProjection;
    mat4 inverseView;
    mat4 prevView;
} ubo;
out gl_PerVertex { vec4 gl_Position; };

const vec3 barycentrics[3] = vec3[3](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

void main() {
    vec4 worldPos = push.modelMatrix * vec4(position, 1.0);
    fragPosWorld = worldPos.xyz;
    fragNormalWorld = normalize(mat3(push.normalMatrix) * normal);
    fragColor = color;
    fragUV = uv;
    fragBarycentric = barycentrics[gl_VertexIndex % 3];

    gl_Position = ubo.projection * ubo.view * worldPos;
    fragClipPos = gl_Position;

}