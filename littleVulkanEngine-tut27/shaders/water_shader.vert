#version 450

// Instance attributes only - no per-vertex attributes
layout(location = 0) in vec4 instancePosRadius;  // xyz = position, w = radius
layout(location = 1) in vec4 instanceColor;       // rgba color

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragPosWorld;
layout(location = 2) out vec2 fragLocalPos;

layout(set = 0, binding = 0) uniform WaterUbo {
    mat4 projection;
    mat4 inverseProjection;
    mat4 view;
    mat4 inverseView;
} ubo;

out gl_PerVertex {
    vec4 gl_Position;
};

// Billboard quad positions (for 6 vertices of a quad)
const vec3 quadPositions[6] = vec3[](
    vec3(-1.0, -1.0, 0.0),
    vec3( 1.0, -1.0, 0.0),
    vec3( 1.0,  1.0, 0.0),
    vec3(-1.0, -1.0, 0.0),
    vec3( 1.0,  1.0, 0.0),
    vec3(-1.0,  1.0, 0.0)
);

// Billboard quad UVs (optional, if needed for fragment shader)
const vec2 quadUVs[6] = vec2[](
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),
    vec2(1.0, 1.0),
    vec2(0.0, 0.0),
    vec2(1.0, 1.0),
    vec2(0.0, 1.0)
);
void main() {
    vec3 instancePos = instancePosRadius.xyz;
    float radius = instancePosRadius.w;

    vec2 quadPos = quadPositions[gl_VertexIndex % 6].xy;
    // Extract camera axes
    vec3 camRight = ubo.inverseView[0].xyz;
    vec3 camUp    = ubo.inverseView[1].xyz;

    // Build billboard vertex
    vec3 worldPos =
        instancePos +
        camRight * quadPos.x * radius +
        -camUp    * quadPos.y * radius;

    fragPosWorld = worldPos;
    fragColor = instanceColor.rgb;
    fragLocalPos = quadPos;

    gl_Position = ubo.projection * ubo.view * vec4(worldPos, 1.0);
}
