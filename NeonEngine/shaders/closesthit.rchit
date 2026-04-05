#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable

// ReSTIR PT implementation adapted from https://github.com/DQLin/ReSTIR_PT/ 


const int MAX_DEPTH = 3;

struct HitInfo
{
    vec4 misc1

    /*

        packed vector of the following: 
        vec2 barycentrics;
        uint primitveID; // The triangle we hit on the model. 
                         // original implementation had another uint for the mesh instance ID, which imma ignore.
    
    */
};

struct PathReservoir
{
    vec4 F = vec4(0.0); // cached integrand (always updated after a new path is chosen in RIS)
    vec4 cachedJacobian; // saved previous vertex scatter PDF, scatter PDF, and geometry term at rcVertex (used when rcVertex is not v2)

    vec4 rcVertexWi[2]; // incident direction on reconnection vertex
    vec4 rcVertexIrradiance[2]; // sampled irradiance on reconnection vertex

    vec4 misc1;
    
    /* 
    
        packed vector of the following:
    
        float M = 0.f; // this is a float, because temporal history length is allowed to be a fraction. 
        float weight = 0.f; // during RIS and when used as a "RisState", this is w_sum; during RIS when used as an incoming reseroivr or after RIS, this is 1/p(y) * 1/M * w_sum
        uint pathFlags; // this is a path type indicator, see the struct definition for details
        uint rcRandomSeed; // saved random seed after rcVertex (due to the need of blending half-vector reuse and random number replay)
    
    */

    vec4 misc2;

    /* 
    
        packed vector of the following:
     
        float lightPdf; // NEE light pdf (might change after shift if transmission is included since light sampling considers "upperHemisphere" of the previous bounce)
        uint initRandomSeed; // saved random seed at the first bounce (for recovering the random distance threshold for hybrid shift)
        
    */

    HitInfo hitInfo;

};

struct RayPayload {
    vec4 color;
    vec4 throughput;
    uvec4 misc; // x = seed, y = path depth. z = material type. w = path ID.
};

layout(location = 0) rayPayloadInEXT RayPayload payload;
layout(location = 1) rayPayloadEXT float shadowPayload;

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;

hitAttributeEXT vec2 attribs;

struct Material {
    vec4 emission;
    vec4 albedo;
    vec4 position;// position of light, if light. hardcoded into the main material struc for now.
    vec4 misc;
};

struct RayTracingVertex {
    vec4 pos;
    vec4 normal;
    vec4 uv;
    uint materialIndex;
    uint pad1;
    uint pad2; 
    uint pad3; 
    
};


layout(set = 0, binding = 3) readonly buffer MaterialBuffer {
    Material m[];
} materials;

layout(set = 0, binding = 4) readonly buffer VertexBuffer {
    RayTracingVertex v[];
} vertices;

layout(set = 0, binding = 5) readonly buffer IndexBuffer {
    uint i[];
} indices;

layout(set = 0, binding = 6) readonly buffer ResevoirBuffer {
    PathReservoir p[];
} reservoirs;

layout(set = 0, binding = 7) uniform sampler2D textures[];  // unbounded array

float luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}


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


void firstHit(uint rayId, vec2 bary, vec3 newDirection, vec3 directContribution)
{
    reservoirs.p[rayId].rcVertexWi[0] = vec4(newDirection, 0.0);

    // rcVertexIrradiance — the incoming light at rcVertex (from NEE + indirect)
    // This is what gets reused when a neighbor shifts their path to hit your rcVertex
    reservoirs.p[rayId].rcVertexIrradiance[0] =
        vec4(directContribution, 0.0);

    // Store triangle data so neighbors can reconnect to this point
    reservoirs.p[rayId].hitInfo.misc1.z = gl_PrimitiveID ;
    // misc2.zw = barycentrics
    reservoirs.p[rayID].hitInfo.misc1.x = bary.x;
    reservoirs.p[rayID].hitInfo.misc1.y = bary.y;

    // Save the random seed at this point, for random replay of the suffix
    reservoir.misc2.y = /*current seed state*/;  // initRandomSeed or rcRandomSeed
}

void pathTerminate(uint rayID)
{
    reservoir[rayID].F = vec4(payload.color.xyz, 0.0);

    // RIS weight update — this is the WRS step
    // target = luminance(F), source pdf = 1/N for uniform candidate sampling
    float targetPdf = dot(reservoir[rayID].F.rgb, vec3(0.2126, 0.7152, 0.0722)); // luminance
    float risWeight = targetPdf > 0.0 ? 1.0 / targetPdf : 0.0;

    // misc2.x = M (sample count), misc2.y = w_sum
    reservoir[rayID].misc2.x += 1.0;  // M++
    reservoir[rayID].misc2.y += risWeight * targetPdf;  // w_sum += w_i * p_hat(x_i)
}

//move this somewhere else
void finalizeRIS(inout PathReservoir r) {
    float M = r.misc2.x;
    float wSum = r.misc2.y;
    float pHat = luminance(r.F.rgb);

    // weight = 1/p_hat(y) * 1/M * w_sum
    r.misc2.y = (pHat > 0.0) ? (wSum / (M * pHat)) : 0.0;
}
// add the jacobian stuff we're missing

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

// mat types (mat.misc.z) : glass = 67, normal diffuse = 0, texture = 2
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

    RayTracingVertex v0 = vertices.v[index.x];
    RayTracingVertex v1 = vertices.v[index.y];
    RayTracingVertex v2 = vertices.v[index.z];

    vec3 normal = normalize(
        v0.normal.xyz * barycentrics.x +
        v1.normal.xyz * barycentrics.y +
        v2.normal.xyz * barycentrics.z
    );

    vec2 uv = v0.uv.xy * barycentrics.x
        + v1.uv.xy * barycentrics.y
        + v2.uv.xy * barycentrics.z;

    normal = faceforward(normal, gl_WorldRayDirectionEXT, normal);

    vec3 worldPos = vec3(0);
    vec3 offsetPos = vec3(0);

    Material mat = materials.m[v0.materialIndex + 1]; // dodo hack but alas (light is the first material)


    if (length(mat.emission.xyz) > 0.0) {
        if (payload.misc.y == 0u || payload.misc.z == 1u) {
            payload.color += vec4(
                payload.throughput.xyz * mat.emission.xyz,
                1.0
            );
        }
        pathTerminate(payload.misc.w);
        return;
    }


    if (payload.misc.y >= uint(MAX_DEPTH)) return;

    if(payload.misc.y == 0) firstHit(payload.misc.w, barycentrics.xy, );
    float maxThroughput = max(
        payload.throughput.r,
        max(payload.throughput.g, payload.throughput.b)
    );

    if (payload.misc.y > 2u) {
        float q = max(0.05, 1.0 - maxThroughput);
        if (random(payload.misc.x) < q)
        {
            pathTerminate(payload.misc.w);
            return;
        }
        payload.throughput /= (1.0 - q);
    }


    const float INV_PI = 0.3183098861837907;

    payload.misc.y++;

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

        vec3 specularDir = reflect(normalize(gl_WorldRayDirectionEXT), normal);
        float rough = clamp(mat.misc.x, 0.0, 1.0);
        newDirection = normalize(mix(specularDir, diffuseDir, rough));

        // NEE — use throughput BEFORE albedo is applied
        Material lightMat = materials.m[0];
        vec3 lightDir;
        float lightPDF;

        vec3 Le = sampleSphereLightMat(
            lightMat, offsetPos, payload.misc.x, lightDir, lightPDF
        );

        float cosTheta = dot(normal, lightDir);

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
                lightDir, shadowDist, 1
            );

        if (shadowPayload > 0.5) {
            vec3 brdf = mat.albedo.xyz * INV_PI;

            vec3 direct =
                payload.throughput.xyz *
                brdf *
                Le *
                cosTheta /
                lightPDF;

            payload.color += vec4(direct, 1.0);
        }
        }

        // Apply albedo AFTER NEE, for the indirect bounce
        payload.throughput *= vec4(mat.albedo.xyz, 1.0);

        payload.misc.z = 0u;
    }
    else  if (int(mat.misc.z) == 2) {

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

        vec3 specularDir = reflect(normalize(gl_WorldRayDirectionEXT), normal);
        float rough = clamp(mat.misc.x, 0.0, 1.0);
        newDirection = normalize(mix(specularDir, diffuseDir, rough));

        // NEE — use throughput BEFORE albedo is applied
        Material lightMat = materials.m[0];
        vec3 lightDir;
        float lightPDF;

        vec3 Le = sampleSphereLightMat(
            lightMat, offsetPos, payload.misc.x, lightDir, lightPDF
        );

        float cosTheta = dot(normal, lightDir);

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
                lightDir, shadowDist, 1
            );

            if (shadowPayload > 0.5) {
                vec3 texAlbedo = texture(textures[1], uv).rgb;
                vec3 brdf = texAlbedo * INV_PI;

                vec3 direct =
                    payload.throughput.xyz *
                    brdf *
                    Le *
                    cosTheta /
                    lightPDF;

                payload.color += vec4(direct, 1.0);
            }
        }

        vec3 texAlbedo = texture(textures[1], uv).rgb;
        payload.throughput *= vec4(texAlbedo, 1.0);

        payload.misc.z = 2;
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