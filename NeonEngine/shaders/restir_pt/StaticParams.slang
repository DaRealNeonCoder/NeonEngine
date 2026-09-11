/***************************************************************************
 # Copyright (c) 2022, Daqi Lin.  All rights reserved.
 **************************************************************************/
//__exported import Rendering.Materials.TexLODTypes;

/** Translation of defines set by the host to compile-time constants used to
    configure the path tracer without overhead from dynamic control flow.
    This will eventually be replaced by specialization constants in Slang.
*/

/*
#ifndef SAMPLES_PER_PIXEL
#error StaticParams.slang is missing defines
#endif
*/
//finna adjust later
static const uint kSamplesPerPixel = 1;
static const uint kCandidateSamples = 1;
static const uint kMaxSurfaceBounces = 9;
static const uint kMaxDiffuseBounces = 9;
static const uint kMaxSpecularBounces = 9;
static const uint kMaxTransmissionBounces = 9;
static const bool kAdjustShadingNormals = false;
static const bool kUseBSDFSampling = true;
static const bool kUseNEE = true;
static const bool kUseMIS = true;
static const bool kUseRussianRoulette = false;
static const bool kUseAlphaTest = true;
static const bool kUseLightsInDielectricVolumes = false;
static const bool kLimitTransmission = false;
static const uint kMaxTransmissionReflectionDepth = 0;
static const uint kMaxTransmissionRefractionDepth = 0;
static const bool kDisableCaustics = false;
static const bool kDisableDirectIllumination = true;
//static const TexLODMode kPrimaryLodMode = TexLODMode(0); // Mip0 shouldnt be a problem right?
static const uint kColorFormat = 1; // LogLuvHDR
static const uint kMISHeuristic = 0; // Balance
static const float kMISPowerExponent = 2.0f;
static const uint kShiftStrategy = 2; // Hybrid
static const uint kSpatialReSTIRMISKind = 2; // Pairwise
static const uint kTemporalReSTIRMISKind = 1; // Talbot
static const bool kTemporalUpdateForDynamicScene = false;
static const int kMaximumPathLength = 15;
static const uint kPathSamplingMode = 0; // ReSTIR
static const bool kSeparatePathBSDF = true;
static const bool kUseNRDDemodulation = true;