#version 450


layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec2 uv;

//some of these are useless. should really clean this.
layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragPosWorld;
layout(location = 2) out vec3 fragNormalWorld;
layout (location = 3) out vec3 outViewVec;
layout (location = 4) out vec3 outLightVec;
layout(location = 5) out vec4 outShadowCoord;
layout(location = 6) out vec2 outuv;

struct PointLight {
vec4 position; // ignore w
vec4 color; // w is intensity
};


layout(set = 0, binding = 0, std140) uniform GlobalUbo {
    mat4 projection;
    mat4 inverseProjection;
    mat4 view;
    mat4 inverseView;

    vec4 ambientLightColor;

    PointLight pointLights[1];
    vec4 numLights;

    mat4 lightSpaceMatrix;
    vec4 lightDir;
} ubo;


layout(push_constant) uniform Push {
mat4 modelMatrix;
mat4 normalMatrix;
vec4 thisIsStupid;
} push;


out gl_PerVertex {
    vec4 gl_Position;
};

const mat4 biasMat = mat4( 
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0 );

/*

const mat4 biasMat = mat4(
    0.5, 0.0, 0.0, 0.0,
    0.0, 0.5, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.5, 0.5, 0.5, 1.0
);
*/
// Then in your vertex shader, simplify to:
void main() 
{
    vec4 worldPos = push.modelMatrix * vec4(position, 1.0);
    fragPosWorld = worldPos.xyz;
    fragNormalWorld = normalize(mat3(push.normalMatrix) * normal);
    fragColor = color;
    
    outShadowCoord = (biasMat * ubo.lightSpaceMatrix *  push.modelMatrix) * vec4(position, 1.0);
    
    gl_Position = ubo.projection * ubo.view *  push.modelMatrix * vec4(position, 1.0);

    outuv = uv;
}