#version 460
#extension GL_EXT_ray_tracing : enable

struct RayPayload {
    vec4 color;
    vec4 throughput;
    uvec4 misc;   // seed, depth
};
layout(location = 0) rayPayloadInEXT RayPayload payload;
layout(set = 0, binding = 6) uniform sampler2D[] textureMaps; 

const float PI = 3.14159265358979;

vec2 directionToUV(vec3 dir) {
    // Equirectangular mapping
    float u = atan(dir.z, dir.x) / (2.0 * PI) + 0.5;
    float v = asin(clamp(dir.y, -1.0, 1.0)) / PI + 0.5;
    return vec2(u, v);
}

void main()
{
    vec3 dir = normalize(gl_WorldRayDirectionEXT);
    vec2 uv  = directionToUV(dir);
    payload.color.xyz += payload.throughput.xyz * texture(textureMaps[0], uv).rgb;
}