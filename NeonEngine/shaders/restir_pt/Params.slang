/***************************************************************************
 # Copyright (c) 2022, Daqi Lin.  All rights reserved.
 **************************************************************************/
#pragma once



ConstantBuffer<RestirPathTracerParams> params;


/** Enumeration of the atomic counters used.
*/
enum class Counters
{
    // TODO: Better naming
    kGeneralPaths = 0,
    kGeneralPathsProcessed,
    kSpecularPaths,
    kSpecularPathsProcessed,
    kDeltaReflectionPaths,
    kDeltaReflectionPathsProcessed,
    kDeltaTransmissionPaths,
    kDeltaTransmissionPathsProcessed,

    kGeneralQueued,
    kGeneralQueuedProcessed,

    kThreadCountSpecular,
    kThreadCountGeneral,
    kThreadCountDeltaReflection,
    kThreadCountDeltaTransmission,

    // Must be last
    kCount
};

/** Per-sample color format.
*/
enum class ColorFormat
    // TODO: Remove the ifdefs and the include when Slang supports enum type specifiers.
{
    RGBA32F = 0,
    LogLuvHDR = 1,
};

enum class MISHeuristic
    // TODO: Remove the ifdefs and the include when Slang supports enum type specifiers.
{
    Balance = 0,    ///< Balance heuristic.
    PowerTwo = 1,    ///< Power heuristic (exponent = 2.0).
    PowerExp = 2,    ///< Power heuristic (variable exponent).
};

// Define tile sizes in pixels.
// The frame is divided into tiles stored in scanline order, with pixels in tiles enumerated in Morton order.
static const uint2 kScreenTileDim = { 16, 16 };     ///< Screen-tile dimension in pixels.
static const uint2 kScreenTileBits = { 4, 4 };      ///< Bits needed to describe pixel position within a screen-tile.

// Define path configuration limits.
static const uint kMaximumPathLength = 15;         ///< Maximum supported sample count. We can use tiling to support large sample counts if needed.
static const uint kMaxSamplesPerPixel = 64;         ///< Maximum supported sample count. We can use tiling to support large sample counts if needed.
static const uint kMaxFrameDimension = 4096;        ///< Maximum supported frame dimension in pixels along x or y. We can increase the bit allocation if needed.
static const uint kMaxBounces = 14;                ///< Maximum supported number of bounces per bounce category (value 255 is reserved for internal use). The resulting path length may be longer than this.
static const uint kMaxLightSamplesPerVertex = 8;    ///< Maximum number of shadow rays per path vertex for next-event estimation.
static const uint kLobeTypeBits = 8;  
static const uint kMaterialTypeBits = 8; 
enum class ShiftMapping
{
    Reconnection = 0,
    RandomReplay = 1,
    Hybrid = 2,
};

enum class ReSTIRMISKind
{
    Constant = 0,
    Talbot = 1,
    Pairwise = 2,
    ConstantBinary = 3,
    ConstantBiased = 4
};

enum class SpatialReusePattern
{
    Default = 0, // same as in ScreenSpaceReSTIR
    SmallWindow = 1,
};

enum class PathSamplingMode
{
    ReSTIR,
    PathReuse,
    PathTracing
};

enum class PathReusePattern
{
    Block,
    NRooks,
    NRooksShift
};


struct VertexData
{
    float3 posW;            ///< Position in world space.
    float3 normalW;         ///< Shading normal in world space (normalized).
    float4 tangentW;        ///< Shading tangent in world space (normalized). The last component is guaranteed to be +-1.0 or zero if tangents are missing.
    float2 texC;            ///< Texture coordinate.
    float3 faceNormalW;     ///< Face normal in world space (normalized).
    float  coneTexLODValue; ///< Texture LOD data for cone tracing. This is zero, unless getVertexDataRayCones() is used.
};


struct ShadingData
{
    float3  posW;                   ///< Shading hit position in world space.
    float3  V;                      ///< Direction to the eye at shading hit.
    float3  N;                      ///< Shading normal at shading hit.
    float3  T;                      ///< Shading tangent at shading hit.
    float3  B;                      ///< Shading bitangent at shading hit.
    float2  uv;                     ///< Texture mapping coordinates.
    float   NdotV;                  ///< Unclamped, can be negative.
    float   normalCurvature;        ///< Normal curvature.

    // Primitive data
    float3  faceN;                  ///< Face normal in world space, always on the front-facing side.
    bool    frontFacing;            ///< True if primitive seen from the front-facing side.

    // Material data
    uint    materialID;             ///< Material ID at shading location.
    uint    flags;                  ///< Flags storing additional fields. Use the accessor functions below.

    // Pre-loaded material properties
    float3  diffuse;                ///< Diffuse albedo.
    float   opacity;
    float3  specular;               ///< Specular albedo.
    float   linearRoughness;        ///< This is the original roughness, before re-mapping.
    float3  transmission;           ///< Transmission color.
    float3  emissive;
    float   IoR;                    ///< Index of refraction of the medium "behind" the surface.
    float   metallic;               ///< Metallic parameter, blends between dielectric and conducting BSDFs. It is used for storing the curve sphere radius when it refers to shading curves.
    float   diffuseTransmission;    ///< Diffuse transmission, blends between diffuse reflection and transmission lobes.
    float   specularTransmission;   ///< Specular transmission, blends between opaque dielectric BRDF and specular transmissive BSDF.
    float   eta;                    ///< Relative index of refraction (incident IoR / transmissive IoR).


    // Accessor functions

    static const uint kMaterialTypeOffset = 0;
    static const uint kActiveLobesOffset = kMaterialTypeOffset + kMaterialTypeBits; // Note: kMaterialTypeBits is declared in MaterialData.slang.
    static const uint kDoubleSidedOffset = kActiveLobesOffset + kLobeTypeBits; // Note: kLobeTypeBits is declared in IBxDF.slang.
    static const uint kThinSurfaceOffset = kDoubleSidedOffset + 1;

    /** Set material type at the hit.
    */
    [mutating] void setMaterialType(MaterialType type)
    {
        flags = ((((uint)type) & ((1 << kMaterialTypeBits) - 1)) << kMaterialTypeOffset) |
                (flags & (~(((1 << kMaterialTypeBits) - 1) << kMaterialTypeOffset)));
    }

    /** Get material type at the hit.
    */
    MaterialType getMaterialType()
    {
        return MaterialType((flags >> kMaterialTypeOffset) & ((1 << kMaterialTypeBits) - 1));
    }

    /** Set active BSDF lobes. See LobeType in IBxDF.slang.
    */
    [mutating] void setActiveLobes(uint activeLobes)
    {
        flags = ((activeLobes & ((1 << kLobeTypeBits) - 1)) << kActiveLobesOffset) |
                (flags & (~(((1 << kLobeTypeBits) - 1) << kActiveLobesOffset)));
    }

    /** Get active BSDF lobes.
    */
    uint getActiveLobes()
    {
        return (flags >> kActiveLobesOffset) & ((1 << kLobeTypeBits) - 1);
    }

    /** Set double-sided flag, if false only shade front face.
    */
    [mutating] void setDoubleSided(bool doubleSided)
    {
        flags = ((doubleSided ? 1u : 0u) << kDoubleSidedOffset) |
                (flags & (~(1u << kDoubleSidedOffset)));
    }

    /** Get double-sided flag.
    */
    bool isDoubleSided()
    {
        return ((flags >> kDoubleSidedOffset) & 1u) != 0;
    }

    /** Set thin surface flag, if true material has a thin surface.
    */
    [mutating] void setThinSurface(bool thinSurface)
    {
        flags = ((thinSurface ? 1u : 0u) << kThinSurfaceOffset) |
                (flags & (~(1u << kThinSurfaceOffset)));
    }

    /** Get thin surface flag.
    */
    bool isThinSurface()
    {
        return ((flags >> kThinSurfaceOffset) & 1u) != 0;
    }
    // Utility functions

    /** Computes new ray origin based on the hit point to avoid self-intersection.
        The method is described in Ray Tracing Gems, Chapter 6, "A Fast and Robust
        Method for Avoiding Self-Intersection" by Carsten W�chter and Nikolaus Binder.
        \param[in] viewside True if the origin should be on the view side (reflection) or false otherwise (transmission).
        \return Ray origin of the new ray.
    */
    float3 computeNewRayOrigin(bool viewside = true)
    {
        return computeRayOrigin(posW, (frontFacing == viewside) ? faceN : -faceN);
    }
    
    
    /** Transform vector from the local surface frame to world space.
        \param[in] v Vector in local space.
        \return Vector in world space.
    */
    float3 fromLocal(float3 v)
    {
        return T * v.x + B * v.y + N * v.z;
    }

    /** Transform vector from world space to the local surface frame.
        \param[in] v Vector in world space.
        \return Vector in local space.
    */
    float3 toLocal(float3 v)
    {
        return float3(dot(v, T), dot(v, B), dot(v, N));
    }
};
// TODO: get rid of this.
// TODO: Specify the backing type when Slang issue has been resolved
enum class MaterialType // : uint32_t
{
    Standard,
    Cloth,
    Hair,

    Count // Must be last
};

//cls
float3 getPrimaryRayDir(uint2 pixel, uint2 frameDim)
{
    float2 uv  = (float2(pixel) + 0.5) / float2(frameDim);
    float2 ndc = uv * 2.0 - 1.0;

    float3 dir = normalize(float3(ndc, 1.0));

    return dir;
}

struct HitInfo
{
    float4 misc1;

    /*

        packed vector of the following: 
        vec2 barycentrics;
        float primitveID; // The triangle we hit on the model. 
                         // original implementation had another uint for the mesh instance ID, which imma ignore.
    */

   [mutating] void clear()
    {
        misc1.x = 0;
    }

    /** Return true if object represents a valid hit.
    */
    bool isValid()
    {
        return misc1.x != 0;
    }
};

struct Material
{
    float4 diffuseOpacity;     // xyz = albedo, w = opacity
    float4 specularRoughness;  // xyz = specular, w = roughness
    float4 emissive;           // xyz = emission, 
    float4 extraParams;        // whatever you want (metallic, etc.)
    
    uint flags;
    float IoR;
    uint2 dummy;
    
    MaterialType getType() { return MaterialType::Standard; }
};

/** Path tracer parmeters shared between host/device code.
*/
struct RestirPathTracerParams
{
    int     useFixedSeed = false;       ///< Use fixed random seed. This is useful for debugging.
    uint    seed = 0;                   ///< Random seed. This will get updated from the host depending on settings.
    uint    fixedSeed = 1;              ///< The seed to use when 'useFixedSeed' is enabled.
    float   lodBias = 0.f;              ///< LOD bias applied to secondary hits.

    // Runtime values
    uint2   frameDim = { 0, 0 };        ///< Frame dimension in pixels.
    uint2   screenTiles = { 0, 0 };     ///< Number of screen-tiles. Screen tiles may extend outside the frame.

    uint    frameCount = 0;             ///< Frames rendered. This is used as random seed.
    uint    localStrategyType = (uint)LocalStrategy::RoughnessCondition | (uint)LocalStrategy::DistanceCondition;
    int     rejectShiftBasedOnJacobian = false; // can be helpful for dynamic geometry
    float   jacobianRejectionThreshold = 10.f;


    float posW = 0.0f;
    float nearZ = 0.0f;
    float farZ = 0.0f;
    float cameraWLen = 0.0f;

    float prevPosW = 0.0f;
    float prevNearZ = 0.0f;
    float prevFarZ = 0.0f;
    float prevCameraWLen = 0.0f;

    
    float3   cameraU                = float3(0, 0, 1);          ///< Camera base vector U. Normalized it indicates the right image plane vector. The length is dependent on the FOV.                
    float   specularRoughnessThreshold = 0.2f;                  ///< Specular reflection events are only classified as specular if the material's roughness value is equal or smaller than this threshold. Otherwise they are classified diffuse.
    
    float3   cameraV                = float3(0, 1, 0);          ///< Camera base vector V. Normalized it indicates the up image plane vector. The length is dependent on the FOV.
    float    jitterX                = 0.0f;                     ///< Eventual camera jitter along the x axis expressed as a subpixel offset divided by screen width (positive value shifts the image right).
    
    float3   cameraW                = float3(1, 0, 0);          ///< Camera base vector W. Normalized it indicates the forward direction. The length is the camera focal distance.
    float    jitterY                = 0.0f;                     ///< Eventual camera jitter along the y axis expressed as a subpixel offset divided by screen height (positive value shifts the image up).

    float3   prevCameraU                = float3(0, 0, 1);          ///< Camera base vector U. Normalized it indicates the right image plane vector. The length is dependent on the FOV.                
    float   nearFieldDistance = 0.1f;                  ///< Specular reflection events are only classified as specular if the material's roughness value is equal or smaller than this threshold. Otherwise they are classified diffuse.
    
    float3   prevCameraV                = float3(0, 1, 0);          ///< Camera base vector V. Normalized it indicates the up image plane vector. The length is dependent on the FOV.
    float    prevJitterX                = 0.0f;                     ///< Eventual camera jitter along the x axis expressed as a subpixel offset divided by screen width (positive value shifts the image right).
    
    float3   prevCameraW                = float3(1, 0, 0);          ///< Camera base vector W. Normalized it indicates the forward direction. The length is the camera focal distance.
    float    prevJitterY                = 0.0f;                     ///< Eventual camera jitter along the y axis expressed as a subpixel offset divided by screen height (positive value shifts the image up).

    uint getReservoirOffset(const uint2 pixel)
    {
        uint2 tileID = pixel >> kScreenTileBits;

        uint stride = kScreenTileDim.x * kScreenTileDim.y;
        uint tileIdx = tileID.y * screenTiles.x + tileID.x;
        uint tileOffset = tileIdx * stride;

        uint tileBits = kScreenTileBits.x + kScreenTileBits.y;
        uint pixelIdx = interleave_16bit(pixel) & ((1 << tileBits) - 1); // TODO: Use interleave_8bit() if kScreenTileBits <= 4.
        return tileOffset + pixelIdx;
    }
    //from https://github.com/DQLin/ReSTIR_PT/blob/8d12332228eb64bc234e27c6f7e0913a926285ab/Source/Falcor/Utils/Math/BitTricks.slang#L32
    uint interleave_16bit(uint2 v)
    {
        v &= 0xff;
        uint j = (v.y << 16) | v.x;             // j = ---- ---- (   y   ) ---- ---- (   x   )
                                                // j = ---- ---- fedc ba98 ---- ---- 7654 3210
        j = (j ^ (j << 4)) & 0x0f0f0f0f;        // j = ---- fedc ---- ba98 ---- 7654 ---- 3210
        j = (j ^ (j << 2)) & 0x33333333;        // j = --fe --dc --ba --98 --76 --54 --32 --10
        j = (j ^ (j << 1)) & 0x55555555;        // j = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
        return (j >> 15) | (j & 0xffff);        // j = ---- ---- ---- ---- f7e6 d5c4 b3a2 9180
    }

};

enum class LocalStrategy
{
    None = 0x0,
    RoughnessCondition = 0x1,
    DistanceCondition = 0x2,
};

struct S
{
    uint state;

    [mutating]
    uint next()
    {
        const uint A = 1664525u;
        const uint C = 1013904223u;

        state = A * state + C;
        return state;
    }
    
    uint getCurrentSeed()
    {
        return state;
    }
}

float sampleNext1D(inout S sg)
{
    // Use upper 24 bits to get a value in [0, 1).
    uint bits = sg.next();
    return float(bits >> 8) * 0x1p-24;
}


float2 sampleNext2D(inout S sg)
{
    float2 sample;
    // Don't use the float2 initializer to ensure consistent order of evaluation.
    sample.x = sampleNext1D(sg);
    sample.y = sampleNext1D(sg);
    return sample;
}

/*

gonna keep this just in case.


static Ray computeRayPinhole(uint2 pixel, uint2 frameDim)
{
    Ray ray;

    // Compute the normalized ray direction assuming a pinhole camera.
    ray.origin = params.posW;

    float2 uv = (float2(pixel) + 0.5) / float2(frameDim);
    float2 ndc = uv * 2.0 - 1.0;

    ray.dir = normalize(float3(ndc, 1.0));

    float invCos = 1.0 / dot(normalize(params.cameraW), ray.dir);
    ray.tMin = params.nearZ * invCos;
    ray.tMax = params.farZ * invCos;

    return ray;
}
*/

static Ray computeRayPinhole(uint2 pixel, uint2 frameDim)
{
    Ray ray;
    // Compute sample position in screen space in [0,1] with origin at the top-left corner.
    float2 p = (pixel + float2(0.5f, 0.5f)) / frameDim - float2(params.jitterX, params.jitterY);
    float2 ndc = 2.f * p - 1.f;
    // Compute the normalized ray direction assuming a pinhole camera.
    ray.origin = params.posW;
    ray.dir = normalize(ndc.x * params.cameraU + ndc.y * params.cameraV + params.cameraW);
    float invCos = 1.f / dot(normalize(params.cameraW), ray.dir);
    ray.tMin = params.nearZ * invCos;
    ray.tMax = params.farZ * invCos;
    return ray;
}

static Ray computeRayPinholePrevFrame(uint2 pixel, uint2 frameDim)
{
    Ray ray;
    // Compute sample position in screen space in [0,1] with origin at the top-left corner.
    float2 p = (pixel + float2(0.5f, 0.5f)) / frameDim - float2(params.prevJitterX, params.prevJitterY);
    float2 ndc = 2.f * p - 1.f;
    // Compute the normalized ray direction assuming a pinhole camera.
    ray.origin = params.prevPosW;
    ray.dir = normalize(ndc.x * params.prevCameraU + ndc.y * params.prevCameraV + params.prevCameraW);
    float invCos = 1.f / dot(normalize(params.prevCameraW), ray.dir);
    ray.tMin = params.prevNearZ * invCos;
    ray.tMax = params.prevFarZ * invCos;
    return ray;
}


// posssible isssue 
struct Ray
{
    float3 origin;
    float3 dir;
    float tMin;
    float tMax;
};




struct BSDFSample_
{
    float3  wi;             ///< Incident direction in world space (normalized).
    float   pdf;            ///< pdf with respect to solid angle for sampling incident direction wi.
    float3  weight;         ///< Sample weight f(wo, wi) * dot(wi, n) / pdf(wi).
    uint    lobe;           ///< Sampled lobe. This is a combination of LobeType flags (see BxDFTypes.slang).
    float   pdfSingle;

    bool isLobe(LobeType type)
    {
        return (lobe & uint(type)) != 0;
    }
};


//TODO: clean this up. We dont use all of em.
enum class LobeType
{
    DiffuseReflection       = 0x01,
    SpecularReflection      = 0x02,
    DeltaReflection         = 0x04,

    DiffuseTransmission     = 0x10,
    SpecularTransmission    = 0x20,
    DeltaTransmission       = 0x40,

    Diffuse                 = 0x11,
    Specular                = 0x22,
    Delta                   = 0x44,
    NonDelta                = 0x33,

    SpecularOrDelta         = 0x66,

    Reflection              = 0x0f,
    Transmission            = 0xf0,

    NonDeltaReflection      = 0x03,
    NonDeltaTransmission    = 0x30,

    All                     = 0xff,
};

//POSSIBLE SOURCE OF ERROR
float3 computeRayOrigin(float3 pos, float3 normal)
{
    const float origin = 1.f / 32.f;
    const float fScale = 1.f / 65536.f;
    const float iScale = 256.f;

    // Per-component integer offset to bit representation of fp32 position.
    int3 iOff = int3(normal * iScale);

    bool3 negative = pos < 0.f;
    float3 iPos = asfloat(asint(pos) + select(negative, -iOff, iOff));

    // Select per-component between small fixed offset or above variable offset
    // depending on distance to origin.
    bool3 nearOrigin = abs(pos) < origin;
    return select(nearOrigin, pos + normal * fScale, iPos);
}