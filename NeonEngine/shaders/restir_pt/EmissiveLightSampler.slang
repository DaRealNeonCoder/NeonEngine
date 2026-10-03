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
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
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
//#include "Utils/Math/MathConstants.slangh"

//import Scene.Scene;
//import Utils.Sampling.SampleGeneratorInterface;
//import Rendering.Lights.EmissiveLightSamplerHelpers;
//import Rendering.Lights.EmissiveLightSamplerInterface;
import Params;
#define FLT_MIN             1.175494351e-38F  
/** Emissive light sampler using uniform sampling of the emissive triangles.

    The sampler implements the IEmissiveLightSampler interface (see
    EmissiveLightSamplerInterface.slang for usage information).

    The struct wraps a LightCollection that stores the pre-processed lights.
    The program should instantiate the struct below. See EmissiveLightSampler.slang.
*/

//TODO: Use a better sampler.
struct EmissiveLightSampler
{
    uint4 _dummy;   ///< Dummy data necessary to have EmissiveUniformSampler as member on another struct.

    /** Draw a single light sample.
        \param[in] posW Shading point in world space.
        \param[in] normalW Normal at the shading point in world space.
        \param[in] upperHemisphere True if only upper hemisphere should be considered.
        \param[in,out] sg Sample generator.
        \param[out] ls Light sample. Only valid if true is returned.
        \return True if a sample was generated, false otherwise.
    */
    bool sampleLight(const float3 posW, const float3 normalW, const bool upperHemisphere, inout S sg, out TriangleLightSample ls)
    {
        // gonna assume this isnt false. if (gScene.lightCollection.getActiveTriangleCount() == 0) return false;

        // Randomly pick a triangle out of the global list of active triangles with uniform probability.
        float uLight = sampleNext1D(sg);
        uint triangleCount = getActiveTriangleCount();
        uint idx = min((uint)(uLight * triangleCount), triangleCount - 1); // Safety precaution as the result of the multiplication may be rounded to triangleCount even if uLight < 1.0 when triangleCount is large.
        uint triangleIndex = idx;
        float triangleSelectionPdf = 1.f / (float)triangleCount;

        // Sample the triangle uniformly.
        float2 u = sampleNext2D(sg);
        if (!sampleTriangle(posW, triangleIndex, u, ls)) return false;

        // The final probability density is the product of the sampling probabilities.
        ls.pdf *= triangleSelectionPdf;
        return true;
    }

    void skipRandomNumber(inout S sg)
    {
        //if (gScene.lightCollection.getActiveTriangleCount() == 0) return ;
        sampleNext1D(sg);
        sampleNext2D(sg);
    }

    /** Evaluate the PDF associated with selecting an emissive triangle.
        \param[in] posW Shading point in world space.
        \param[in] normalW Normal at the shading point in world space.
        \param[in] upperHemisphere True if only upper hemisphere should be considered.
        \param[in] triangleIndex Index of selected triangle.
        \return Probability of drawing the input triangle from an uniform distribution.
    */
    float evalTriangleSelectionPdf(const float3 posW, const float3 normalW, const bool upperHemisphere, const uint triangleIndex)
    {
        //invalid
        if (triangleIndex == -1) return 0.f;

        // Lights are chosen uniformly so the selection probability is just one over the number of lights.
        return getActiveTriangleCount() > 0 ? 1.f / (float)getActiveTriangleCount() : 0.f;
    }

    /** Evaluate the PDF at a shading point given a hit point on an emissive triangle.
        \param[in] posW Shading point in world space.
        \param[in] normalW Normal at the shading point in world space.
        \param[in] upperHemisphere True if only upper hemisphere should be considered.
        \param[in] hit Triangle hit data.
        \return Probability density with respect to solid angle at the shading point.
    */
    float evalPdf(const float3 posW, const float3 normalW, const bool upperHemisphere, const TriangleLightHit hit)
    {
        if (hit.triangleIndex == -1) return 0.f;

        float triangleSelectionPdf = evalTriangleSelectionPdf(posW, normalW, upperHemisphere, hit.triangleIndex);

        // Compute triangle sampling probability with respect to solid angle from the shading point.
        float trianglePdf = evalTrianglePdf(posW, hit);

        // The final probability density is the product of the sampling probabilities.
        return triangleSelectionPdf * trianglePdf;
    }
};


struct TriangleLightHit
{
    uint    triangleIndex;      ///< Index of the triangle.
    float3  posW;               ///< Sampled point on the triangle in world space.
    float3  normalW;            ///< Face normal of the triangle in world space, flipped for back-facing hits.
};


/** Describes a light sample on an emissive triangle.
    If pdf == 0.0 is returned, the sample in invalid and the other fields should not be used.
*/
struct TriangleLightSample
{
    uint    triangleIndex;      ///< Index of the sampled triangle.
    float3  posW;               ///< Sampled point on the light source in world space.
    float3  normalW;            ///< Normal of the sampled point on the light source in world space.
    float3  dir;                ///< Normalized direction from the shading point to the sampled point on the light source in world space.
    float   distance;           ///< Distance from the shading point to the sampled point.
    float3  Le;                 ///< Emitted radiance. This is zero if the light is back-facing or sample is invalid.
    float   pdf;                ///< Probability density with respect to solid angle from the shading point. The range is [0,inf] (inclusive), where pdf == 0.0 indicated an invalid sample.
    float2  uv;                 ///< Light sample barycentric coords over the triangle
};




float evalTrianglePdf(const float3 posW, const TriangleLightHit hit)
{
    // Compute light vector and squared distance.
    float3 toLight = hit.posW - posW; // Unnormalized light vector
    const float distSqr = dot(toLight, toLight);
    if (distSqr <= FLT_MIN) return 0.f; // Avoid NaNs below
    float3 L = toLight / sqrt(distSqr);

    // Cosine of angle between the light's normal and the light vector (flip L since it points towards the light).
    float cosTheta = dot(hit.normalW, -L);
    if (cosTheta <= 0.f) return 0.f;

    // Compute probability density with respect to solid angle from the shading point.
    // The farther away the light is and the larger the angle it is at, the larger the pdf becomes. The probability goes to infinity in the limit.
    // Note: Guard against div-by-zero here by clamping.
    // TODO: Do we need the clamp here? distSqr is already clamped, so NaN should not be possible (but +inf is).
    float denom = max(FLT_MIN, cosTheta * getTriangleArea(hit.triangleIndex).area);
    return distSqr / denom;
}



struct PrecomputedEmissiveTriangle
{
    float3 p0, p1, p2;
    float3 normal;
    float  area;
    float3 Le;
};

StructuredBuffer<PrecomputedEmissiveTriangle> gEmissiveTriangles;

cbuffer PerFrameCB
{
    uint gEmissiveTriangleCount;
};


uint getActiveTriangleCount()
{
    return gEmissiveTriangleCount;
}

struct TriangleAreaData
{
    float3 normal;
    float  area;
};

TriangleAreaData getTriangleArea(uint triangleIndex)
{
    TriangleAreaData d;
    d.normal = gEmissiveTriangles[triangleIndex].normal;
    d.area   = gEmissiveTriangles[triangleIndex].area;
    return d;
}

//POSSIBLE SOURCE OF ERROR
bool sampleTriangle(const float3 posW, const uint triangleIndex, const float2 u, out TriangleLightSample ls)
{
    PrecomputedEmissiveTriangle tri = gEmissiveTriangles[triangleIndex];

    // Low-distortion uniform barycentric sample.
    float su0 = sqrt(u.x);
    float b0 = 1.f - su0;
    float b1 = u.y * su0;
    float b2 = 1.f - b0 - b1;

    ls.triangleIndex = triangleIndex;
    ls.posW    = b0 * tri.p0 + b1 * tri.p1 + b2 * tri.p2;
    ls.normalW = tri.normal;
    ls.uv      = float2(b1, b2); // only meaningful if you later add per-vertex texcrd

    float3 toLight = ls.posW - posW;
    float distSqr = dot(toLight, toLight);
    if (distSqr <= FLT_MIN)
    {
        ls.pdf = 0.f;
        ls.Le = float3(0, 0, 0);
        return true;
    }
    ls.distance = sqrt(distSqr);
    ls.dir = toLight / ls.distance;

    float cosTheta = dot(tri.normal, -ls.dir);
    if (cosTheta <= 0.f)
    {
        // Single-sided emitter, hit from the back.
        ls.pdf = 0.f;
        ls.Le = float3(0, 0, 0);
        return true;
    }

    // Same area->solid-angle conversion as evalTrianglePdf() — keep these in sync,
    // your MIS weights depend on sample pdf and eval pdf agreeing.
    ls.pdf = distSqr / (cosTheta * tri.area);
    ls.Le = tri.Le;
    return true;
}