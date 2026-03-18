#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable

const int MAX_DEPTH = 6; // Add this at top of closest hit shader



struct RayPayload {
    vec4 color;
    vec4 throughput;
    uvec2 misc;   // seed, depth
};

layout(location = 0) rayPayloadInEXT RayPayload payload;

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;


hitAttributeEXT vec2 attribs;

struct Material {
    vec4 emission;
    vec4 albedo;
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

void main() {
    // Barycentric interpolation
    vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    
    ivec3 index = ivec3(
        indices.i[3 * gl_PrimitiveID],
        indices.i[3 * gl_PrimitiveID + 1],
        indices.i[3 * gl_PrimitiveID + 2]
    );
    
    Vertex v0 = vertices.v[index.x];
    Vertex v1 = vertices.v[index.y];
    Vertex v2 = vertices.v[index.z];
    
    // Interpolate normal
    vec3 normal = normalize(
        v0.normal.xyz * barycentrics.x +
        v1.normal.xyz * barycentrics.y +
        v2.normal.xyz * barycentrics.z
    );
    

    // Ensure normal faces ray origin
    normal = faceforward(normal, gl_WorldRayDirectionEXT, normal);
    
    // World position
    vec3 worldPos = vec3(0);
    
    vec3 offsetPos = vec3(0);
    
    //Per object materials
    Material mat = materials.m[v0.materialIndex];
    
    // Handle emissive materials
    if (length(mat.emission.xyz) > 0.0) {
        payload.color += vec4(payload.throughput.xyz * mat.emission.xyz, 1.0);
        return;
    }
    
    // Max depth check
    if (payload.misc.y >= MAX_DEPTH) {
        return;
    }
    
    // Russian Roulette 
    float maxThroughput = max(payload.throughput.r, max(payload.throughput.g, payload.throughput.b));
    if (payload.misc.y > 2) {
        float q = max(0.05, 1.0 - maxThroughput);
        if (random(payload.misc.x) < q) {
            return;
        }
        payload.throughput /= (1.0 - q);
    }
    
    const float INV_PI = 0.3183098861837907;
    
    payload.misc.y++;
    payload.throughput *= vec4(mat.albedo.xyz, 1.0);
    
    vec3 newDirection = vec3(0);

    if(mat.misc.z == 0)
        {

        normal = faceforward(normal, gl_WorldRayDirectionEXT, normal);
    
        worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    
        offsetPos = worldPos + normal * 0.001;
        
        // ---------- SAMPLE DIFFUSE DIRECTION ----------
        float r1 = random(payload.misc.x);
        float r2 = random(payload.misc.x);

        float r = sqrt(r1);
        float theta = 2.0 * 3.14159265 * r2;

        float x = r * cos(theta);
        float y = r * sin(theta);
        float z = sqrt(1.0 - r1);

        // Tangent frame
        vec3 tangent = abs(normal.x) > abs(normal.y)
            ? normalize(vec3(-normal.z, 0, normal.x))
            : normalize(vec3(0, normal.z, -normal.y));

        vec3 bitangent = cross(normal, tangent);

        // Diffuse bounce
        vec3 diffuseDir = normalize(tangent * x + bitangent * y + normal * z);

        vec3 incoming = normalize(gl_WorldRayDirectionEXT);
        vec3 specularDir = reflect(incoming, normal);

        float rough = clamp(mat.misc.x, 0.0, 1.0);

        // More rough = more diffuse
        newDirection = normalize(mix(specularDir, diffuseDir, rough));

    }
    else if (mat.misc.z == 67)
    {

        worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    
        offsetPos = worldPos + normal * 0.001;

        vec3 incoming = normalize(gl_WorldRayDirectionEXT);

        float cosTheta = clamp(dot(-incoming, normal), 0.0, 1.0);

        bool entering = cosTheta > 0.0;
        float eta = entering ? (1.0 / mat.misc.y) : mat.misc.y;
        normal = entering ? normal : -normal;

        float F = fresnelSchlick(cosTheta, mat.misc.y);

        if (random(payload.misc.x) < F) {
            newDirection = reflect(incoming, normal);
        } else {
            newDirection = refract(incoming, normal, eta);

            // Total internal reflection safety
            if (length(newDirection) < 0.001) {
                newDirection = reflect(incoming, normal);
            }
        }
    }
        traceRayEXT(
        topLevelAS,
        gl_RayFlagsOpaqueEXT | gl_RayFlagsCullBackFacingTrianglesEXT,
        0xFF,
        0, 0, 0,
        offsetPos,
        0.001,
        newDirection,
        10000.0,
        0
    );
}