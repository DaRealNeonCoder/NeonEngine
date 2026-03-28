#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable

const int MAX_DEPTH = 6;

struct RayPayload {
    vec4 color;
    vec4 throughput;
    uvec4 misc;
};

layout(location = 0) rayPayloadInEXT RayPayload payload;
layout(location = 1) rayPayloadEXT float shadowPayload;

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;

hitAttributeEXT vec2 attribs;

struct Material {
    vec4 emission;
    vec4 albedo;
    vec4 position;
    vec4 misc;
};

struct Vertex {
    vec4 pos;
    vec4 normal;
    uint materialIndex;
};

layout(set = 0, binding = 3) readonly buffer MaterialBuffer {
    Material m[];
} materials;

layout(set = 0, binding = 4) readonly buffer VertexBuffer {
    Vertex v[];
} vertices;

layout(set = 0, binding = 5) readonly buffer IndexBuffer {
    uint i[];
} indices;



float random(inout uint seed) {
    seed = seed * 747796405u + 2891336453u;
    uint result = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
    result = (result >> 22u) ^ result;
    return float(result) / 4294967295.0;
}

float fresnelSchlick(float cosTheta, float eta) {
    float r0 = (1.0 - eta) / (1.0 + eta);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cosTheta, 5.0);
}


vec3 sampleSphereLightMat(
    Material light,
    vec3 p,
    inout uint seed,
    out vec3 lightDir,
    out float pdf
) {
    vec3 lightPos = light.position.xyz;
    float radius = light.position.w;

    float dist = length(lightPos - p);
    float sinThetaMax2 = (radius * radius) / (dist * dist);
    float cosThetaMax = sqrt(max(0.0, 1.0 - sinThetaMax2));

    float r1 = random(seed);
    float r2 = random(seed);

    float cosTheta = 1.0 - r1 * (1.0 - cosThetaMax);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi = 2.0 * 3.14159265 * r2;

    vec3 toLight = normalize(lightPos - p);

    vec3 tangent =
        abs(toLight.x) > abs(toLight.y)
        ? normalize(vec3(-toLight.z, 0, toLight.x))
        : normalize(vec3(0, toLight.z, -toLight.y));

    vec3 bitangent = cross(toLight, tangent);

    lightDir = normalize(
        tangent * (sinTheta * cos(phi)) +
        bitangent * (sinTheta * sin(phi)) +
        toLight * cosTheta
    );

    pdf = 1.0 / (2.0 * 3.14159265 * (1.0 - cosThetaMax));

    return light.emission.xyz;
}


void main() {

    vec3 barycentrics = vec3(
        1.0 - attribs.x - attribs.y,
        attribs.x,
        attribs.y
    );

    ivec3 index = ivec3(
        indices.i[3 * gl_PrimitiveID],
        indices.i[3 * gl_PrimitiveID + 1],
        indices.i[3 * gl_PrimitiveID + 2]
    );

    Vertex v0 = vertices.v[index.x];
    Vertex v1 = vertices.v[index.y];
    Vertex v2 = vertices.v[index.z];

    vec3 normal = normalize(
        v0.normal.xyz * barycentrics.x +
        v1.normal.xyz * barycentrics.y +
        v2.normal.xyz * barycentrics.z
    );

    normal = faceforward(normal, gl_WorldRayDirectionEXT, normal);

    vec3 worldPos = vec3(0);
    vec3 offsetPos = vec3(0);

    Material mat = materials.m[v0.materialIndex];


    if (length(mat.emission.xyz) > 0.0) {
        if (payload.misc.y == 0u || payload.misc.z == 1u) {
            payload.color += vec4(
                payload.throughput.xyz * mat.emission.xyz,
                1.0
            );
        }
        return;
    }


    if (payload.misc.y >= uint(MAX_DEPTH)) return;


    float maxThroughput = max(
        payload.throughput.r,
        max(payload.throughput.g, payload.throughput.b)
    );

    if (payload.misc.y > 2u) {
        float q = max(0.05, 1.0 - maxThroughput);
        if (random(payload.misc.x) < q) return;
        payload.throughput /= (1.0 - q);
    }


    const float INV_PI = 0.3183098861837907;

    payload.misc.y++;
    payload.throughput *= vec4(mat.albedo.xyz, 1.0);

    vec3 newDirection = vec3(0);


    if (int(mat.misc.z) == 0) {

        normal = faceforward(normal, gl_WorldRayDirectionEXT, normal);

        worldPos = gl_WorldRayOriginEXT +
                   gl_WorldRayDirectionEXT *
                   gl_HitTEXT;

        offsetPos = worldPos + normal * 0.001;


        float r1 = random(payload.misc.x);
        float r2 = random(payload.misc.x);

        float r = sqrt(r1);
        float theta = 2.0 * 3.14159265 * r2;

        vec3 tangent =
            abs(normal.x) > abs(normal.y)
            ? normalize(vec3(-normal.z, 0, normal.x))
            : normalize(vec3(0, normal.z, -normal.y));

        vec3 bitangent = cross(normal, tangent);

        vec3 diffuseDir = normalize(
            tangent * (r * cos(theta)) +
            bitangent * (r * sin(theta)) +
            normal * sqrt(1.0 - r1)
        );

        vec3 specularDir =
            reflect(
                normalize(gl_WorldRayDirectionEXT),
                normal
            );

        float rough = clamp(mat.misc.x, 0.0, 1.0);

        newDirection =
            normalize(
                mix(specularDir, diffuseDir, rough)
            );


        Material lightMat =
            materials.m[1];

        vec3 lightDir;
        float lightPDF;

        vec3 Le =
            sampleSphereLightMat(
                lightMat,
                offsetPos,
                payload.misc.x,
                lightDir,
                lightPDF
            );

        float cosTheta =
            dot(normal, lightDir);

        if (cosTheta > 0.0 && lightPDF > 0.0) {

            float shadowDist =
                length(lightMat.position.xyz - offsetPos)
                - lightMat.position.w
                - 0.01;

            shadowPayload = 0.0;

            traceRayEXT(
                topLevelAS,
                gl_RayFlagsTerminateOnFirstHitEXT |
                gl_RayFlagsSkipClosestHitShaderEXT,
                0xFF, 0, 0, 1,
                offsetPos, 0.001,
                lightDir,
                shadowDist,
                1
            );

            if (shadowPayload > 0.5) {

                vec3 brdf =
                    mat.albedo.xyz * INV_PI;

                vec3 direct =
                    payload.throughput.xyz *
                    brdf *
                    Le *
                    cosTheta /
                    lightPDF;

                payload.color.xyz += direct;
            }
        }

        payload.misc.z = 0u;
    }


    else if (int(mat.misc.z) == 67) {

        worldPos =
            gl_WorldRayOriginEXT +
            gl_WorldRayDirectionEXT *
            gl_HitTEXT;

        offsetPos =
            worldPos +
            normal * 0.001;

        vec3 incoming =
            normalize(gl_WorldRayDirectionEXT);

        float cosTheta =
            clamp(
                dot(-incoming, normal),
                0.0,
                1.0
            );

        bool entering =
            cosTheta > 0.0;

        float eta =
            entering
            ? (1.0 / mat.misc.y)
            : mat.misc.y;

        normal =
            entering
            ? normal
            : -normal;

        float F =
            fresnelSchlick(
                cosTheta,
                mat.misc.y
            );

        if (random(payload.misc.x) < F) {
            newDirection =
                reflect(incoming, normal);
        }
        else {
            newDirection =
                refract(incoming, normal, eta);

            if (length(newDirection) < 0.001)
                newDirection =
                    reflect(incoming, normal);
        }

        payload.misc.z = 1u;
    }


    traceRayEXT(
        topLevelAS,
        gl_RayFlagsOpaqueEXT |
        gl_RayFlagsCullBackFacingTrianglesEXT,
        0xFF, 0, 0, 0,
        offsetPos, 0.001,
        newDirection,
        10000.0,
        0
    );
}