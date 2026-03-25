#version 450

// Inputs from vertex shader
layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragPosWorld;
layout(location = 2) in vec3 fragNormalWorld;
layout(location = 3) in vec3 fragBarycentric;
layout(location = 4) in vec2 fragUV;

// Outputs (4 MRTs)
layout(location = 0) out vec4 outPosition;    // xyz = world pos, w = triangle ID
layout(location = 1) out vec4 outNormal;      // xyz = world normal, w = material ID
layout(location = 2) out vec4 outBarycentric; // xyz = barycentrics, w = instance ID
layout(location = 3) out vec4 outMotion;      // xy = motion vector, z = depth, w = depth derivative

layout(set = 0, binding = 0, std140) uniform GlobalUbo {
    mat4 projection;
    mat4 view;
    mat4 inverseProjection;
    mat4 inverseView;
    mat4 prevView;
} ubo;

void main() {
    outPosition    = vec4(fragPosWorld, 1.0);
    outNormal      = vec4(normalize(fragNormalWorld), 0.0);
    outBarycentric = vec4(fragBarycentric, 0.0);

    vec4 currClip = ubo.projection * ubo.view * vec4(fragPosWorld, 1.0);
    vec2 currNDC  = currClip.xy / currClip.w;
    vec2 currUV   = currNDC * 0.5 + 0.5;

    vec4 prevClip = ubo.projection * ubo.prevView * vec4(fragPosWorld, 1.0);
    vec2 prevNDC  = prevClip.xy / prevClip.w;
    vec2 prevUV   = prevNDC * 0.5 + 0.5;

    vec2 motionVector = currUV - prevUV;

    float viewZ = -(ubo.view * vec4(fragPosWorld, 1.0)).z;

    outMotion.xy = motionVector;
    outMotion.z  = viewZ;
    outMotion.w  = 0.0;
}