/***************************************************************************
 # Copyright (c) 2022, Daqi Lin.  All rights reserved.
 **************************************************************************/
import Params;
import Shift;
///import Scene.HitInfo;
import PathReservoir;
import PathTracer;
//import Scene.Scene;
//import Utils.Debug.PixelDebug;
//import Rendering.Utils.PixelStats;
//import Utils.Sampling.TinyUniformSampleGenerator;
//import Utils.Math.Ray;
import LoadShadingData;
//import Rendering.Materials.MaterialShading;

/*

   TODO.

*/
struct PathRetracePass
{
    static const uint kNeighborOffsetCount = 8192;
    static const uint kNeighborOffsetMask = 8192 - 1;

    // Resources
    RestirPathTracerParams params;                        ///< Runtime parameters.

    //float4 = hitinfo
    Texture2D<float4> vbuffer;                     ///< Fullscreen V-buffer for the primary hits.
    Texture2D<float4> temporalVbuffer;
    Texture2D<float2> motionVectors;

    ByteAddressBuffer nRooksPattern;

    RWTexture2D<float4> outputColor;                      ///< Output resolved color.
    RWStructuredBuffer<PathReservoir> outputReservoirs;                  
    RWStructuredBuffer<PathReservoir> temporalReservoirs;
    RWStructuredBuffer<PixelReconnectionData> reconnectionDataBuffer;

    int  gNumSpatialRounds;
    bool gEnableTemporalReprojection;

    float gTemporalHistoryLength;
    bool gNoResamplingForTemporalReuse;

    bool isValidHitInfo(HitInfo packed)
    {
        return packed.misc1.x != 0;
    }

    bool isValidScreenRegion(int2 pixel) { return all(pixel >= 0 && pixel < params.frameDim); }

    ShadingData getPixelShadingData(int2 pixel, out HitInfo PrimaryHit)
    {
        ShadingData sd = {};
        Ray ray = Params::computeRayPinhole(pixel, params.frameDim);
        PrimaryHit = vbuffer[pixel];
        if (isValidHitInfo(PrimaryHit))
        {
            HitInfo PrimaryHit; PrimaryHit.unpack(PrimaryHit);
            sd = loadShadingData(PrimaryHit, -ray.dir, true);
        }
        return sd;
    }
    
    //TODO: why is this defined multiple times?
    
    ShadingData getPixelTemporalShadingData(int2 pixel, out HitInfo PrimaryHit)
    {
        ShadingData sd = {};
        Ray ray = computeRayPinholePrevFrame(pixel, params.frameDim);
        PrimaryHit = temporalVbuffer[pixel];
        if (isValidHitInfo(PrimaryHit))
        {
            HitInfo PrimaryHit; PrimaryHit.unpack(PrimaryHit);
            sd = loadShadingData(PrimaryHit, -ray.dir, true);
        }
        return sd;
    }

    /*
    * Build a reservoir of paths whose suffix can be merged to a pixel's prefix.
    * This method is optimized for small windows such that we can run the entire
    * algorithm in one pass.
    */
    void ReSTIR(const uint2 pixel)
    {
        // TODO: How to seed efficiently?
        var sg = S(pixel, (kCandidateSamples + 1 + gNumSpatialRounds) * params.seed + kCandidateSamples);

        // Compute offset into per-sample buffers. All samples are stored consecutively at this offset.
        const uint centralOffset = params.getReservoirOffset(pixel);
        PathReservoir centralReservoir = outputReservoirs[centralOffset];

        HitInfo centralPrimaryHit;
        ShadingData centralPrimarySd = getPixelShadingData(pixel, centralPrimaryHit);
        if (!isValidHitInfo(centralPrimaryHit)) return;

        // color averaging test

        // fetch temporal reservoir
        float2 motionVector = motionVectors[pixel];
        int2 prevPixel = pixel;

        bool foundTemporalSurface = true;

        if (gEnableTemporalReprojection)
        {
            prevPixel = pixel + motionVector * params.frameDim + (sampleNext2D(sg) * 1.f - 0.f);
        }

        if (!isValidScreenRegion(prevPixel) || !foundTemporalSurface) return;

        HitInfo temporalPrimaryHit;
        ShadingData temporalPrimarySd = getPixelTemporalShadingData(prevPixel, temporalPrimaryHit);
        if (!isValidHitInfo(temporalPrimaryHit)) return;

        PathReservoir temporalReservoir = temporalReservoirs[params.getReservoirOffset(prevPixel)];

        // talbot MIS
        // compute mis weight for current pixel

        HitInfo dstRcPrevVertexHit;
        HitInfo dstRcPrevVertexHit2;
        float3 dstRcPrevVertexWo;
        float3 dstRcPrevVertexWo2;
        float3 Tp;
        float3 Tp2;

        // write to memory
        if (centralReservoir.pathFlags.rcVertexLength() > 1)
        {
            Tp = traceHybridShiftRays(params, true, temporalPrimaryHit, temporalPrimarySd, centralReservoir, dstRcPrevVertexHit, dstRcPrevVertexWo);
            reconnectionDataBuffer[centralOffset].data[0] = ReconnectionData(dstRcPrevVertexHit, dstRcPrevVertexWo, Tp);
        }
        if (temporalReservoir.pathFlags.rcVertexLength() > 1)
        {
            Tp2 = traceHybridShiftRays(params, false, centralPrimaryHit, centralPrimarySd, temporalReservoir, dstRcPrevVertexHit2, dstRcPrevVertexWo2);
            reconnectionDataBuffer[centralOffset].data[1] = ReconnectionData(dstRcPrevVertexHit2, dstRcPrevVertexWo2, Tp2);
        }
    }

    /** Entry point for merge pass.
        \param[in] pixel Pixel coordinates.
    */
    void execute(const uint2 _pixel)
    {
        const uint2 pixel = _pixel;

        if (any(pixel >= params.frameDim)) return;

        printSetPixel(pixel);
        logSetPixel(pixel);

        ReSTIR(pixel);
    }

};

cbuffer CB
{
    PathRetracePass gPathRetracePass;
}


[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gPathRetracePass.execute(dispatchThreadId.xy);
}
