#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragPosWorld;
layout(location = 2) in vec2 fragLocalPos;

layout(location = 0) out vec4 outColor;

void main() {

    // Convert quad coords (-1..1) into circle radius
    float r2 = dot(fragLocalPos, fragLocalPos);

    // Discard pixels outside circle
    if (r2 > 1.0)
        discard;

    // Reconstruct sphere normal in view-facing space
    float z = sqrt(1.0 - r2);
    vec3 normal = normalize(vec3(fragLocalPos.xy, z));

    // Simple directional light
    vec3 lightDir = normalize(vec3(0.4, 0.7, 0.2));

    float diffuse = max(dot(normal, lightDir), 0.0);

    // Soft ambient term
    float ambient = 0.2;

    vec3 color = fragColor * (ambient + diffuse);

    outColor = vec4(color, 1.0);
}
