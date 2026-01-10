#version 450


struct PointLight {
vec4 position;
vec4 color;
};


layout(set = 0, binding = 0, std140) uniform GlobalUbo {
    mat4 projection;
    mat4 view;
    mat4 inverseView;

    vec4 ambientLightColor;

    PointLight pointLights[1];
    vec4 numLights;

    mat4 lightSpaceMatrix;
    vec4 lightDir;
} ubo;

// Must match pipeline vertex input descriptions
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec2 uv;


layout(push_constant) uniform PushConstants {
mat4 model;
} pushConst;


void main() {

gl_Position = ubo.lightSpaceMatrix * pushConst.model *vec4(position, 1.0);
}