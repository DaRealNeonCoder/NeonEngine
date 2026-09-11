/***************************************************************************
 # Copyright (c) 2022, Daqi Lin.  All rights reserved.
 **************************************************************************/
//import Utils.Math.PackedFormats;
import PathBuilder;
import PathReservoir;
//__exported import Scene.HitInfo;
//__exported import Utils.Sampling.SampleGenerator;
//import Rendering.Materials.InteriorListHelpers;
import Params;

static const uint kMaxRejectedHits = 16; // Maximum number of rejected hits along a path. The path is terminated if the limit is reached to avoid getting stuck in pathological cases.

static const float kRayTMax = 1e30f;

/** Path flags. The path flags are currently stored in a uint16_t.
*/
enum class PathFlags
{
    active                  = 0x0001,   ///< Path is active/terminated.
    hit                     = 0x0002,   ///< Result of the scatter ray (0 = miss, 1 = hit).

    transmission            = 0x0004,   ///< Scatter ray went through a transmission event.
    specular                = 0x0008,   ///< Scatter ray went through a specular event.
    delta                   = 0x0010,   ///< Scatter ray went through a delta event.
    volume                  = 0x0020,   ///< Scatter ray went through a volume event.

    insideDielectricVolume  = 0x0040,   ///< Path vertex is inside a dielectric volume.
    lightSampledUpper       = 0x0080,   ///< Last path vertex sampled lights using NEE (in upper hemisphere).
    lightSampledLower       = 0x0100,   ///< Last path vertex sampled lights using NEE (in lower hemisphere).

    diffusePrimaryHit       = 0x0200,   ///< Primary hit was sampled from the diffuse BSDF lobe.
    freePath                = 0x0400,   ///< This is a free path (volume sampling).

    specularBounce          = 0x0800,    ///< counts everything that is not diffuse

    specularPrimaryHit      = 0x1000,   ///< Primary hit was sampled from the specular BSDF lobe.

    // Bits 11-15 are still unused.
};

/** Bounce types. We keep separate counters for all of these.
*/
enum class BounceType
{
    Diffuse                 = 0,    ///< Diffuse reflection.
    Specular                = 1,    ///< Specular reflection (including delta).
    Transmission            = 2,    ///< Transmission (all kinds).
    Volume                  = 3,    ///< Scatter event in heterogeneous volume.
};

/** Packed path state. Contains all data from PathState in packed form.
    Default total size: 80 + (2*4) + (4-16) bytes.
*/
struct PackedPathState
{
    uint4 packed[5];
    InteriorList interiorList;  ///< Interior list. Keeping track of a stack of materials with medium properties. The size depends on the config (4B/item).
    S sg;         ///< Sample generator state. Typically 4-16B.
};

// TODO: Compact encoding to reduce live registers, e.g. flags and length together, packed HitInfo, packed normals.
/** Live state for the path tracer.
*/
struct PathState
{
    uint        id;                 ///< Path ID encodes (pixel, sampleIdx) with 12 bits each for pixel x|y and 8 bits for sample index.

    uint16_t    flags;              ///< Flags indicating the current status. This can be multiple PathFlags flags OR'ed together.
    uint16_t    length;             ///< Path length (0 at origin, 1 at first secondary hit, etc.).
    uint16_t    rejectedHits;       ///< Number of false intersections rejected along the path. This is used as a safeguard to avoid deadlock in pathological cases.
    float16_t   sceneLength;        ///< Path length in scene units (0.f at primary hit).
    uint        bounceCounters;     ///< Packed counters for different types of bounces (see BounceType).

    // Scatter ray
    float3      origin;             ///< Origin of the scatter ray.
    float3      dir;                ///< Scatter ray normalized direction.
    float       pdf;                ///< Pdf for generating the scatter ray.
    float3      normal;             ///< Shading normal at the scatter ray origin.
    HitInfo     hit;                ///< Hit information for the scatter ray. This is populated at committed triangle hits.

    float3      thp;                ///< Path throughput.
    float3      prefixThp;          /// daqi: used for computing rcVertexIrradiance[1]

    float3      rcVertexPathTreeIrradiance;

    float3      L;                  ///< Accumulated path contribution.
    float3      LDeltaDirect;     // daqi: save the direct lighting on delta surfaces

    float      russianRoulettePdf = 1.f;
    float3      sharedScatterDir;
    float      prevScatterPdf = 1.f;

    HitInfo rcPrevVertexHit;   // daqi: save the previous vertex of rcVertex, used in hybrid shift replay, for later reconnection
    float3 rcPrevVertexWo; // daqi: save the outgoing direction at the previous vertex of rcVertex, used in hybrid shift replay, for later reconnection

    float hitDist; // for NRD

    InteriorList interiorList;      ///< Interior list. Keeping track of a stack of materials with medium properties.
    S sg;             ///< Sample generator state. Typically 4-16B.

    PathBuilder pathBuilder;        ///< Importance samples the path from the path tree.
    PathReservoir pathReservoir;

    bool     enableRandomReplay; // daqi: indicate the pathtracer is doing random number replay (being called from resampling stage)
    bool     randomReplayIsNEE; // daqi: indicate the base path is a NEE path, therefore the random replay should also terminates as a NEE path
    bool     randomReplayIsEscaped; // daqi: indicate the base path is a escaped path, therefore the random replay should also terminates as a escaped path
    int      randomReplayLength; // daqi: the length of the random replay (same as the path length of the base path)
    bool     useHybridShift; 
    // this is used for hybrid shift or hybrid shift is being MIS'ed with (at both initial candidate generation and resampling)
    bool     isReplayForHybridShift; // daqi: indicate that a random number replay is done with the hybrid shift in mind (invalidated if invertibility is violated)
    bool     isLastVertexClassifiedAsRough; // daqi: remembers if last vertex is classified as a diffuse vertex (used for specularskip in hybrid shift)

    // Accessors
    float3 getCurrentThp() { return thp * prefixThp; }
    [mutating] void recordPrefixThp() { prefixThp *= thp; thp = 1.f; }

    bool isTerminated() { return !isActive(); }
    bool isActive() { return hasFlag(PathFlags::active); }
    bool isHit() { return hasFlag(PathFlags::hit); }
    bool isTransmission() { return hasFlag(PathFlags::transmission); }
    bool isSpecular() { return hasFlag(PathFlags::specular); }
    bool isDelta() { return hasFlag(PathFlags::delta); }
    bool isSpecularBounce() { return hasFlag(PathFlags::specularBounce); }
    bool isVolume() { return hasFlag(PathFlags::volume); }
    bool isInsideDielectricVolume() { return hasFlag(PathFlags::insideDielectricVolume); }
    bool isLightSampled(){return (flags & (uint(PathFlags::lightSampledUpper) |uint(PathFlags::lightSampledLower))) != 0;}
    bool isLightSampledUpper() { return hasFlag(PathFlags::lightSampledUpper); }
    bool isLightSampledLower() { return hasFlag(PathFlags::lightSampledLower); }
    bool isDiffusePrimaryHit() { return hasFlag(PathFlags::diffusePrimaryHit); }
    bool isSpecularPrimaryHit() { return hasFlag(PathFlags::specularPrimaryHit); }
    bool isFreePath() { return hasFlag(PathFlags::freePath); }

    // Check if the scatter event is samplable by the light sampling technique.
    bool isLightSamplable() { return !isDelta(); }

    [mutating] void terminate() { setFlag(PathFlags::active, false); }
    [mutating] void setActive() { setFlag(PathFlags::active); }
    [mutating] void setHit(HitInfo hitInfo) { hit = hitInfo; setFlag(PathFlags::hit); }
    [mutating] void clearHit() { setFlag(PathFlags::hit, false); }

    [mutating] void clearEventFlags() { flags &= ~uint16_t(uint(PathFlags::transmission) | uint(PathFlags::specular) | uint(PathFlags::delta) | uint(PathFlags::volume) | uint(PathFlags::specularBounce)); }
    [mutating] void setTransmission(bool value = true) { setFlag(PathFlags::transmission, value); }
    [mutating] void setSpecular(bool value = true) { setFlag(PathFlags::specular, value); }
    [mutating] void setDelta(bool value = true) { setFlag(PathFlags::delta, value); }
    [mutating] void setSpecularBounce(bool value = true) { setFlag(PathFlags::specularBounce, value); }
    [mutating] void setVolume(bool value = true) { setFlag(PathFlags::volume, value); }
    [mutating] void setInsideDielectricVolume(bool value = true) { setFlag(PathFlags::insideDielectricVolume, value); }
    [mutating] void setLightSampled(bool upper, bool lower) { setFlag(PathFlags::lightSampledUpper, upper); setFlag(PathFlags::lightSampledLower, lower); }
    [mutating] void setDiffusePrimaryHit(bool value = true) { setFlag(PathFlags::diffusePrimaryHit, value); }
    [mutating] void setSpecularPrimaryHit(bool value = true) { setFlag(PathFlags::specularPrimaryHit, value); }

    [mutating] void setFreePath(bool value = true) { setFlag(PathFlags::freePath, value); }

    bool hasFlag(PathFlags flag) { return (flags & uint(flag)) != 0; }
    [mutating] void setFlag(PathFlags flag, bool value = true)
    {
        if (value) flags |= uint16_t(uint(flag));
        else flags &= ~uint16_t(uint(flag));
    }

    uint getBounces(BounceType type)
    {
        const uint shift = (uint)type << 3;
        return (bounceCounters >> shift) & 0xff;
    }

    [mutating] void setBounces(BounceType type, uint bounces)
    {
        const uint shift = (uint)type << 3;
        bounceCounters = (bounceCounters & ~((uint)0xff << shift)) | ((bounces & 0xff) << shift);
    }

    [mutating] void incrementBounces(BounceType type)
    {
        const uint shift = (uint)type << 3;
        // We assume that bounce counters cannot overflow.
        bounceCounters += (1 << shift);
    }

    uint2 getPixel() { return uint2(id, id >> 12) & 0xfff; }
    uint getSampleIdx() { return id >> 24; }

    uint getLength() { return length; }

    RayDesc getScatterRay()
    {
        RayDesc ray = { origin, 0.f, dir, kRayTMax };
        return ray;
    }

    // Helpers for packing the full path state.

    PackedPathState encode()
    {
        PackedPathState s = {};

        s.packed[0].xyz = asuint(origin);
        s.packed[0].w = id;

        s.packed[1].xyz = asuint(dir);
        s.packed[1].w = uint(flags) | (uint(length) << 16);

        s.packed[2].xyz = asuint(normal);
        s.packed[2].w = uint(rejectedHits) | ((f32tof16(sceneLength) & 0xffff) << 16);

        s.packed[3].xyz = asuint(thp);
        s.packed[3].w = bounceCounters;

        s.packed[4].xyz = asuint(L);
        s.packed[4].w = asuint(pdf);

        s.interiorList = interiorList;
        s.sg = sg;

        return s;
    }

    [mutating] void decode(const PackedPathState s)
    {
        this = {};

        origin = asfloat(s.packed[0].xyz);
        id = s.packed[0].w;

        dir = asfloat(s.packed[1].xyz);
        flags = uint16_t(s.packed[1].w & 0xffff);
        length = uint16_t(s.packed[1].w >> 16);

        normal = asfloat(s.packed[2].xyz);
        rejectedHits = uint16_t(s.packed[2].w & 0xffff);
        sceneLength = float16_t(f16tof32(s.packed[2].w >> 16));

        thp = asfloat(s.packed[3].xyz);
        bounceCounters = s.packed[3].w;

        L = asfloat(s.packed[4].xyz);
        pdf = asfloat(s.packed[4].w);

        interiorList = s.interiorList;
        sg = s.sg;
    }

};


#ifndef INTERIOR_LIST_SLOT_COUNT
#define INTERIOR_LIST_SLOT_COUNT 2
#endif

struct InteriorList
{
    static const uint kNoMaterial           = 0xffffffff;

    static const uint kMaterialBits         = 28;
    static const uint kNestedPriorityBits   = 4;

    static const uint kMaterialOffset       = 0;
    static const uint kNestedPriorityOffset = kMaterialOffset + kMaterialBits;

    static const uint kMaterialMask         = ((1 << kMaterialBits) - 1) << kMaterialOffset;
    static const uint kNestedPriorityMask   = ((1 << kNestedPriorityBits) - 1) << kNestedPriorityOffset;

    static const uint kMaxNestedPriority    = ((1 << kNestedPriorityBits) - 1);

    uint slots[INTERIOR_LIST_SLOT_COUNT];

    /** Make an active material slot given the material and priority.
        \param[in] materialID Material ID.
        \param[in] nestedPriority Nested priority, 0 is reserved for empty slots.
        \return Returns the encoded slot.
    */
    uint makeSlot(uint materialID, uint nestedPriority)
    {
        return (nestedPriority << kNestedPriorityOffset) | (materialID & kMaterialMask);
    }

    /** Check if a slot is active.
        \param[in] slot Slot value.
        \return Returns true if slot is active.
    */
    bool isSlotActive(uint slot)
    {
        return slot != 0;
    }

    /** Check if the interior list is empty.
        \return Returns true if interior list is empty, false otherwise.
    */
    bool isEmpty()
    {
        return !isSlotActive(slots[0]);
    }

    /** Get the nested priority from a slot.
        \param[in] slot Slot value.
        \param Returns the nested priority or 0 for empty slots.
    */
    uint getSlotNestedPriority(uint slot)
    {
        return slot >> kNestedPriorityOffset;
    }

    /** Get the material ID from a slot.
        \param[in] slot Slot value.
        \return Returns the material ID or 0 for empty slots.
    */
    uint getSlotMaterialID(uint slot)
    {
        return slot & kMaterialMask;
    }

    /** Return current highest nested priority on interior list.
        Because the interior list is sorted by nested priority and empty slots have priority 0,
        we can simple fetch the first slot.
        \return Returns the highest nested priority on the interior list.
    */
    uint getTopNestedPriority()
    {
        return getSlotNestedPriority(slots[0]);
    }

    /** Return the material ID with the highest priority on the interior list.
        \return Returns the material ID or kNoMaterial if interior list is empty.
    */
    uint getTopMaterialID()
    {
        return isSlotActive(slots[0]) ? getSlotMaterialID(slots[0]) : kNoMaterial;
    }

    /** Return the material ID with the 2nd highest priority on the interior list.
        \return Returns the material ID or kNoMaterial if interior list is empty.
    */
    uint getNextMaterialID()
    {
        return isSlotActive(slots[1]) ? getSlotMaterialID(slots[1]) : kNoMaterial;
    }

    /** Check if an intersection with a given surface is a true intersection.
        True intersection occurs if nested priority of intersected mesh is equal or higher
        priority than the highest nested priority on the interior list.
        \param[in] nestedPriority Nested priority of intersected surface, with 0 reserved for the highest possible priority.
        \return Returns true if intersection is a true intersection, false otherwise.
    */
    bool isTrueIntersection(uint nestedPriority)
    {
        // Compare nested priority to current top of stack.
        return nestedPriority == 0 || nestedPriority >= getTopNestedPriority();
    }

    /** Handle an intersection with a given material:
        If material is already on interior list -> remove it.
        If material is not on interior list -> add it.
        \param[in] materialID Material ID of intersected material.
        \param[in] nestedPriority Nested priority of intersected surface, with 0 reserved for the highest possible priority.
        \param[in] entering True if material is entered, false if material is left.
    */
    [mutating] void handleIntersection(uint materialID, uint nestedPriority, bool entering)
    {
        // Remap priority 0 to the highest priority to allow sorting by high->low priority,
        // and as internally 0 is reserved for empty slots.
        if (nestedPriority == 0) nestedPriority = kMaxNestedPriority;

        // The following loop is manually unrolled below to avoid stack allocation.
#if 0
        for (uint slotIndex = 0; slotIndex < INTERIOR_LIST_SLOT_COUNT; ++slotIndex)
        {
            uint slot = slots[slotIndex];

            // Add new slot to interior list
            if (entering && slot == 0)
            {
                slots[slotIndex] = makeSlot(materialID, nestedPriority);
                break;
            }

            // Remove existing slot from interior list
            if (!entering && isSlotActive(slot) && getSlotMaterialID(slot) == materialID)
            {
                slots[slotIndex] = 0;
                break;
            }
        }
#endif

#if INTERIOR_LIST_SLOT_COUNT >= 1
        if (entering && slots[0] == 0)
        {
            slots[0] = makeSlot(materialID, nestedPriority);
        }
        else if (!entering && isSlotActive(slots[0]) && getSlotMaterialID(slots[0]) == materialID)
        {
            slots[0] = 0;
        }
#endif
#if INTERIOR_LIST_SLOT_COUNT >= 2
        else if (entering && slots[1] == 0)
        {
            slots[1] = makeSlot(materialID, nestedPriority);
        }
        else if (!entering && isSlotActive(slots[1]) && getSlotMaterialID(slots[1]) == materialID)
        {
            slots[1] = 0;
        }
#endif
#if INTERIOR_LIST_SLOT_COUNT >= 3
        else if (entering && slots[2] == 0)
        {
            slots[2] = makeSlot(materialID, nestedPriority);
        }
        else if (!entering && isSlotActive(slots[2]) && getSlotMaterialID(slots[2]) == materialID)
        {
            slots[2] = 0;
        }
#endif
#if INTERIOR_LIST_SLOT_COUNT >= 4
        else if (entering && slots[3] == 0)
        {
            slots[3] = makeSlot(materialID, nestedPriority);
        }
        else if (!entering && isSlotActive(slots[3]) && getSlotMaterialID(slots[3]) == materialID)
        {
            slots[3] = 0;
        }
#endif
#if INTERIOR_LIST_SLOT_COUNT >= 5
#error "intersection not handled for given number of slots"
#endif

        sortSlots();
    }

    /** Sort the interior list by priority.
    */
    [mutating] void sortSlots()
    {
#define CSWAP(_a, _b)               \
        if (slots[_a] < slots[_b])  \
        {                           \
            uint tmp = slots[_a];   \
            slots[_a] = slots[_b];  \
            slots[_b] = tmp;        \
        }

        // sorting networks: http://pages.ripco.net/~jgamble/nw.html
#if INTERIOR_LIST_SLOT_COUNT == 2
        CSWAP(0, 1)
#elif INTERIOR_LIST_SLOT_COUNT == 3
        CSWAP(0, 1)
        CSWAP(1, 2)
        CSWAP(0, 1)
#elif INTERIOR_LIST_SLOT_COUNT == 4
        CSWAP(0, 1)
        CSWAP(2, 3)
        CSWAP(0, 2)
        CSWAP(1, 3)
        CSWAP(1, 2)
#else
#error "sorting not handled for given number of slots"
#endif

#undef CSWAP
    }
};