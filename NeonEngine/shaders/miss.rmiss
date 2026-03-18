#version 460
#extension GL_EXT_ray_tracing : enable

struct RayPayload {
    vec4 color;
    vec4 throughput;
    uvec2 misc;   // seed, depth
};
layout(location = 0) rayPayloadInEXT RayPayload payload;
void main()
{
    payload.color = vec4(payload.throughput.xyz * vec3(0.1, 0.1, 0.1), 1.0); // Gray sky
}