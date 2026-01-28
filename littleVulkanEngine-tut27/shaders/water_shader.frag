 #version 450

// y'know what I really hate? people who answer their own questions.

layout(location = 0) in vec3 fragColor;          
layout(location = 1) in vec3 fragPosWorld;      
layout(location = 2) in vec3 fragNormalWorld;   

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0, std140) uniform WaterUbo {
    mat4 projection;
    mat4 inverseProjection;
    mat4 view;
    mat4 inverseView;

} ubo;

layout(push_constant) uniform Push {
    mat4 modelMatrix;
    mat4 normalMatrix;
    vec4 color;
} push;


void main() {
    if(push.color.w == 2) //for da water tank.
    {
    outColor = vec4(push.color.xyz, 1.0);
    return;
    }
    vec3 ambient =  0.3 * vec3(0.0, 0.4, 1.0);
    vec3 diffuseLight = vec3(0.0);
    vec3 surfaceNormal = normalize(fragNormalWorld);

    vec3 cameraPosWorld = ubo.inverseView[3].xyz;
    vec3 viewDirection = normalize(cameraPosWorld - fragPosWorld);

    vec3 directionToLight = normalize(vec3(-10,-10,-10) - fragPosWorld);

    float cosAngIncidence = max(dot(surfaceNormal, directionToLight), 0);
    vec3 intensity = vec3(0.9, 0.9, 0.9);

    diffuseLight += intensity * cosAngIncidence;

    outColor = vec4((diffuseLight * push.color.xyz)+ ambient, 1.0);
}


