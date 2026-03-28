#version 460
#extension GL_EXT_ray_tracing : enable

struct RayPayload {
    vec4 color;
    vec4 throughput;
    uvec4 misc;   // seed, depth
};
layout(location = 0) rayPayloadInEXT RayPayload payload;
void main()
{
    payload.color.xyz += payload.throughput.xyz * vec3(0.1, 0.1, 0.1);
}