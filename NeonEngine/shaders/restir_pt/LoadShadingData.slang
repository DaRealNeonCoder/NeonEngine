/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
//#include "Scene/ScenePrimitiveDefines.slangh"

//import Scene.Scene;
//__exported import Scene.HitInfo;
//__exported import Scene.Shading;
//import Utils.Math.MathHelpers;
//import Rendering.Materials.TexLODHelpers;
import Params;
//import Utils.Debug.PixelDebug;

/** Compute derivatives at primarty triangle hit.
*/
/*
void computeDerivativesAtPrimaryTriangleHit(const TriangleHit hit, uint2 pixel, uint2 frameDim, out float2 ddx, out float2 ddy)
{
    // Get world space position
    VertexData v = gScene.getVertexData(hit);
    float hitT = length(v.posW - gScene.camera.getPosition());
    const float3 nonNormalizedRayDir = gScene.camera.computeNonNormalizedRayDirPinhole(pixel, frameDim);
    float3 rayDir = normalize(nonNormalizedRayDir);

    float3 dDdx, dDdy;
    computeRayDirectionDifferentials(nonNormalizedRayDir, gScene.camera.data.cameraU, gScene.camera.data.cameraV, frameDim, dDdx, dDdy);

    RayDiff rd = RayDiff(float3(0.0f), float3(0.0f), dDdx, dDdy);
    rd = rd.propagate(v.posW.xyz, rayDir, hitT, v.faceNormalW);

    // Bounce using primary hit geometry.
    const float4x4 worldMat = gScene.getWorldMatrix(hit.instanceID);
    const float3x3 worldInvTransposeMat = gScene.getInverseTransposeWorldMatrix(hit.instanceID);
    const uint3 vertexIndices = gScene.getIndices(hit.instanceID, hit.primitiveIndex);
    StaticVertexData vertices[3] = { gScene.getVertex(vertexIndices[0]), gScene.getVertex(vertexIndices[1]), gScene.getVertex(vertexIndices[2]) };
    float2 dBarydx, dBarydy;
    float3 unnormalizedN, normals[3];

    prepareRayDiffAtHitPoint(v, vertices, hit.getBarycentricWeights(), rayDir, worldMat, worldInvTransposeMat, rd, unnormalizedN, normals, dBarydx, dBarydy, ddx, ddy);
}
*/
/** Setup vertex data based on hit point.
    \param[in] hit Hit information.
    \param[out] materialID Material ID.
    \param[out] curveSphereRadius Curve sphere radius.
    \return VertexData struct.
*/

/*
VertexData loadVertexData(const HitInfo hit, const float3 viewDir, out uint materialID, out float curveSphereRadius)
{
    VertexData v = {};

    const TriangleHit triangleHit = hit.getTriangleHit();
    v = gScene.getVertexData(triangleHit);
    materialID = gScene.getMaterialID(triangleHit.instanceID);
    
    return v;
}
*/


/** Adjust ShadingData
    \param[in,out] sd Sha
    data to update.
    \param[in] hit Hit information.
    \param[in] curveSphereRadius Curve sphere radius.
    \param[in] v Vertex data.
    \param[in] isPrimary True if this is the primary hit point.
*/

/** Setup ShadingData based on loaded vertex/material attributes for a hit point.
    \param[in] hit Hit information.
    \param[in] viewDir Normalized view direction.
    \param[in] isPrimary True if this is the primary hit point.
    \param[in] lod level of detail bias value.
    \return ShadingData struct.
*/




/*ShadingData loadShadingData(const HitInfo hit, const float3 viewDir, const bool isPrimary, const float lod = 0.f)
{
    float curveSphereRadius = {};
    uint materialID = {};
    VertexData v = loadVertexData(hit, viewDir, materialID, curveSphereRadius);
    ShadingData sd = prepareShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], viewDir, lod);
    adjustShadingData(sd, hit, curveSphereRadius, v, isPrimary);
    return sd;
}*/

    RWTexture2D<float4> gWorldFaceNormal;              
    RWTexture2D<float4> gWorldShadingNormal;               
    RWTexture2D<float4> gWorldPosition;                
    StructuredBuffer<Material> materials;
    StructuredBuffer<float4x4> matricies; // is this a good idea?

[[vk::binding(1, 0)]] StructuredBuffer<StaticVertexData>         gVertices;

    //float4 = hitinfo
    RWTexture2D<float4> vbuffer;               ///< Fullscreen V-buffer for the primary hits.

//POSSIBLE SOURCE OF ERRORc
bool loadShadingData(
    uint2 pixel,
    uint2 frameDim,
    out ShadingData sd,
    out HitInfo hit
)
{
    sd.posW = float3(0.0);
    sd.N = float3(0.0);
    sd.V = float3(0.0);
    sd.faceN = float3(0.0);
    sd.diffuse = float3(0.0);
    sd.materialID = 0u;
    sd.flags = 0u;
    sd.NdotV = 0.0;

    float3 rayDir = getPrimaryRayDir(pixel, frameDim);

    // ------------------------------------------------------------------
    // GBuffer fetch
    // ------------------------------------------------------------------

    float4 worldPos = gWorldPosition.Load(int2(pixel));

    if (worldPos.w == 0.0)
    {
        hit.misc1 = float4(0.0f);
        return false;
    }

    float3 normal =
        normalize(gWorldShadingNormal.Load(int2(pixel)).xyz);

    float3 faceNormal =
        normalize(gWorldFaceNormal.Load(int2(pixel)).xyz);

    uint materialID = uint(worldPos.w);

    // ------------------------------------------------------------------
    // Material fetch
    // ------------------------------------------------------------------

    Material mat = materials[materialID];

    // ------------------------------------------------------------------
    // Fill shading data
    // ------------------------------------------------------------------

    sd.posW = worldPos.xyz;
    sd.N = normal;
    sd.faceN = faceNormal;
    sd.V = normalize(-rayDir);

    sd.diffuse = mat.diffuseOpacity.xyz;

    sd.materialID = materialID;
    sd.flags = 0;

    sd.NdotV = max(dot(normal, -rayDir), 0.0);

    // ------------------------------------------------------------------
    // VBuffer hit (optional, but you already have it)
    // ------------------------------------------------------------------

    hit.misc1 = vbuffer.Load(int2(pixel));

    return true;
}


ShadingData loadShadingData(const HitInfo hit, const float3 viewDir, const bool isPrimary, const float lod = 0.f)
{
    float curveSphereRadius = {};
    uint materialID = {};
    VertexData v = loadVertexData(hit, viewDir, materialID, curveSphereRadius);
    ShadingData sd = prepareShadingData(v, materialID, materials[materialID], viewDir);
    adjustShadingData(sd, hit, curveSphereRadius, v, isPrimary);
    return sd;
}

VertexData loadVertexData(const HitInfo hit, const float3 viewDir, out uint materialID, out float curveSphereRadius)
{
    VertexData v = {};
    curveSphereRadius = 0.f;
    v = getVertexData((uint)hit.misc1.w, (uint)hit.misc1.z, getBarycentricWeights(hit));
    materialID = (int)hit.misc1.w;
    return v;
}
float3 getBarycentricWeights(HitInfo hit)
{
    return float3(1.f - hit.misc1.x - hit.misc1.y, hit.misc1.x, hit.misc1.y);
}

//instanceID = id of the transform, matricies[0] = worldmat [1] = inv;
VertexData getVertexData(const uint instanceID, const uint triangleIndex, const float3 barycentrics)
{
    StaticVertexData vertices2[3] = { gVertices[triangleIndex], gVertices[triangleIndex + 1], gVertices[triangleIndex + 2] };
    const bool isFrontFaceCW = true;//POSSIBLE SOURCE OF ERROR
    return computeVertexData(vertices2, matricies[instanceID], (float3x3)matricies[instanceID + 1], isFrontFaceCW, barycentrics);
}

VertexData computeVertexData(const StaticVertexData vertices[3], const float4x4 worldMat, const float3x3 worldInvTransposeMat, const bool isFrontFaceCW, const float3 barycentrics)
{
    VertexData v = {};

    v.posW = vertices[0].position * barycentrics[0];
    v.posW += vertices[1].position * barycentrics[1];
    v.posW += vertices[2].position * barycentrics[2];

    v.normalW = vertices[0].normal * barycentrics[0];
    v.normalW += vertices[1].normal * barycentrics[1];
    v.normalW += vertices[2].normal * barycentrics[2];

    v.tangentW = vertices[0].tangent * barycentrics[0];
    v.tangentW += vertices[1].tangent * barycentrics[1];
    v.tangentW += vertices[2].tangent * barycentrics[2];
    v.tangentW.w = sign(v.tangentW.w); // Preserve zero to indicate invalid tangent.

    v.texC = vertices[0].texCrd * barycentrics[0];
    v.texC += vertices[1].texCrd * barycentrics[1];
    v.texC += vertices[2].texCrd * barycentrics[2];

    v.faceNormalW = getFaceNormalInObjectSpace(vertices, isFrontFaceCW);

    v.posW = mul(float4(v.posW, 1.f), worldMat).xyz;
    v.normalW = mul(v.normalW, worldInvTransposeMat);
    v.faceNormalW = mul(v.faceNormalW, worldInvTransposeMat);
    v.tangentW.xyz = mul(v.tangentW.xyz, (float3x3)worldMat);

    v.normalW = normalize(v.normalW);
    v.faceNormalW = normalize(v.faceNormalW);
    v.tangentW.xyz = normalize(v.tangentW.xyz);
    return v;
}
float3 getFaceNormalInObjectSpace(const StaticVertexData vertices[3], const bool isFrontFaceCW)
{
    float3 p0 = vertices[0].position;
    float3 p1 = vertices[1].position;
    float3 p2 = vertices[2].position;
    float3 N = normalize(cross(p1 - p0, p2 - p0));
    return isFrontFaceCW ? -N : N;
}
//can use normal + lod
ShadingData prepareShadingData(VertexData v, uint materialID, Material mat, float3 viewDir)
{
    ShadingData sd = {};

    // Load base color and opacity. These are sampled from the same texture but either one can be uniform parameters.
    // TODO: Rewrite as single texture fetch followed by extraction of color and opacity to make it easier to optimize.
    //float3 baseColor = sampleTexture(mr.baseColor, mr.samplerState, v.texC, mat.baseColor, EXTRACT_BASE_COLOR_TYPE(mat.flags), lod).rgb;
    //sd.opacity = sampleTexture(mr.baseColor, mr.samplerState, v.texC, mat.baseColor, EXTRACT_ALPHA_TYPE(mat.flags), lod).a;
    float3 baseColor = mat.diffuseOpacity.xyz;
    sd.posW = v.posW;
    sd.uv = v.texC;
    sd.V = viewDir;
    sd.N = v.normalW;

    sd.faceN = v.faceNormalW;
    sd.frontFacing = dot(sd.V, sd.faceN) >= 0.f;

    sd.materialID = materialID;
    sd.setMaterialType(mat.getType());
    sd.setActiveLobes((uint)LobeType::All);
    sd.setDoubleSided(extractDoubleSided(mat.flags) != 0);
    sd.setThinSurface(extractThinSurface(mat.flags) != 0);
    
sd.IoR = mat.IoR;
    sd.diffuseTransmission = -1;
    sd.specularTransmission = -1;
    if (sd.diffuseTransmission > 0.f || sd.specularTransmission > 0.f)
    {
//transmission texture? yeah imma pass.
        //sd.transmission = sampleTexture(mr.transmission, mr.samplerState, v.texC, float4(mat.transmission, 0.f), EXTRACT_TRANS_TYPE(mat.flags), lod).rgb;
    }
    sd.eta = sd.frontFacing ? (1 / sd.IoR) : sd.IoR;

    // Setup tangent space.
    bool validTangentSpace = computeTangentSpace(sd, v.tangentW);

    // Sample the specular texture.
    // Depending on the shading model this encodes the material parameters differently.
    // float4 spec = sampleTexture(mr.specular, mr.samplerState, v.texC, mat.specular, EXTRACT_SPECULAR_TYPE(mat.flags), lod);
//simple diffuse for now, I guess  
    // Calculate the specular reflectance for dielectrics from the IoR, as in the Disney BSDF [Burley 2015].
    // UE4 uses 0.08 multiplied by a default specular value of 0.5, hence F0=0.04 as default. The default IoR=1.5 gives the same result.
   
     float f = (sd.IoR - 1.f) / (sd.IoR + 1.f);
    float F0 = f * f;



    if (mat.getType() == MaterialType::Standard)
    {
        // G - Roughness; B - Metallic
        sd.diffuse = baseColor;
        sd.specular = baseColor; //POSSIBLE SOURCE OF ERROR 
        sd.linearRoughness = 0;
        sd.metallic = 0;
    }

    // Sample the emissive texture. Note that triangles are emissive only on the front-facing side.
    if (sd.frontFacing)
    {
        sd.emissive = mat.emissive.xyz;
        //sd.emissive = sampleTexture(mr.emissive, mr.samplerState, v.texC, float4(mat.emissive, 1), EXTRACT_EMISSIVE_TYPE(mat.flags), lod).rgb * mat.emissiveFactor;
    }

    // Apply normal mapping only if we have a valid tangent space.
    //if (useNormalMap && validTangentSpace) applyNormalMap(mat, mr, sd, v.tangentW, lod);
    sd.NdotV = dot(sd.N, sd.V);

    // Flip the shading normal for back-facing hits on double-sided materials.
/*    if (!sd.frontFacing && sd.isDoubleSided())
    {
        sd.N = -sd.N;
        sd.NdotV = -sd.NdotV;
    }
*/
    return sd;
}

bool computeTangentSpace(inout ShadingData sd, const float4 tangentW)
{
    // Check that tangent space exists and can be safely orthonormalized.
    // Otherwise invent a tanget frame based on the normal.
    // We check that:
    //  - Tangent exists, this is indicated by a nonzero sign (w).
    //  - It has nonzero length. Zeros can occur due to interpolation or bad assets.
    //  - It is not parallel to the normal. This can occur due to normal mapping or bad assets.
    //  - It does not have NaNs. These will propagate and trigger the fallback.

    float NdotT = dot(tangentW.xyz, sd.N);
    bool nonParallel = abs(NdotT) < 0.9999f;
    bool nonZero = dot(tangentW.xyz, tangentW.xyz) > 0.f;

    bool valid = tangentW.w != 0.f && nonZero && nonParallel;
    if (valid)
    {
        sd.T = normalize(tangentW.xyz - sd.N * NdotT);
        sd.B = cross(sd.N, sd.T) * tangentW.w;
    }
    else
    {
        sd.T = perp_stark(sd.N);
        sd.B = cross(sd.N, sd.T);
    }

    return valid;
}

inline float3 perp_stark(const float3 u)
{
    // TODO: Validate this and look at numerical precision etc. Are there better ways to do it?
    float3 a = abs(u);
    uint32_t uyx = (a.x - a.y) < 0 ? 1 : 0;
    uint32_t uzx = (a.x - a.z) < 0 ? 1 : 0;
    uint32_t uzy = (a.y - a.z) < 0 ? 1 : 0;
    uint32_t xm = uyx & uzx;
    uint32_t ym = (1 ^ xm) & uzy;
    uint32_t zm = 1 ^ (xm | ym); // 1 ^ (xm & ym)
    float3 v = normalize(cross(u, float3(xm, ym, zm)));
    return v;
}


// Extracts the double-sided flag (1 bit) starting at bit 12
static inline uint32_t extractDoubleSided(uint32_t value)
{
    const uint32_t bits   = 1;
    const uint32_t offset = 12;
    return (value >> offset) & ((1u << bits) - 1);
}

// Extracts the thin-surface flag (1 bit) starting at bit 17
static inline uint32_t extractThinSurface(uint32_t value)
{
    const uint32_t bits   = 1;
    const uint32_t offset = 17;
    return (value >> offset) & ((1u << bits) - 1);
}
static const bool kAdjustShadingNormals = true;
//assumes:GBUFFER_ADJUST_SHADING_NORMALS
void adjustShadingData(inout ShadingData sd, const HitInfo hit, const float curveSphereRadius, const VertexData v, const bool isPrimary)
{
    if (kAdjustShadingNormals || isPrimary) adjustShadingNormal(sd, v);
}

void adjustShadingNormal(inout ShadingData sd, VertexData v)
{
    float3 Ng = sd.frontFacing ? v.faceNormalW : -v.faceNormalW;
    float3 Ns = sd.N;


    // Blend the shading normal towards the geometric normal at grazing angles.
    // This is to avoid the view vector from becoming back-facing.
    const float kCosThetaThreshold = 0.1f;
    float cosTheta = dot(sd.V, Ns);
    if (cosTheta <= kCosThetaThreshold)
    {
        float t = saturate(cosTheta * (1.f / kCosThetaThreshold));
        sd.N = normalize(lerp(Ng, Ns, t));
        sd.NdotV = dot(sd.N, sd.V);
        computeTangentSpace(sd, v.tangentW);
    }
}
/** Setup ShadingData based on loaded vertex/material attributes for a hit point.
    \param[in] hit Hit information.
    \param[in] viewDir Normalized view direction.
    \param[in] isPrimary True if this is the primary hit point.
    \param[in] lod level of detail bias value.
    \return ShadingData struct.
*/

struct StaticVertexData
{
    float3 position;    ///< Position.
    float3 normal;      ///< Shading normal.
    float4 tangent;     ///< Shading tangent. The bitangent is computed: cross(normal, tangent.xyz) * tangent.w. NOTE: The tangent is *only* valid when tangent.w != 0.
    float2 texCrd;      ///< Texture coordinates.
};