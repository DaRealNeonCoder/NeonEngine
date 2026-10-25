implementing slangml_common;

#ifndef SLANGML_USE_BDA
#define SLANGML_USE_BDA 1
#endif

#ifndef SLANGML_INDEX64
#define SLANGML_INDEX64 0
#endif

public bool slangml_is_aligned_u32(uint64_t byteAddress)
{
    return (byteAddress & uint64_t(3)) == uint64_t(0);
}

public bool slangml_is_aligned_vec4(uint64_t byteAddress)
{
    return (byteAddress & uint64_t(15)) == uint64_t(0);
}

// This is the only routine in the hot addressing path that widens to 64-bit.
// The relative offset has already been computed with 32-bit shape/stride math.
public uint64_t slangml_final_byte_address(TensorDesc tensor, int relativeElement, uint elementSizeBytes)
{
    int64_t element = tensor.baseOffset + int64_t(relativeElement);
    return tensor.dataAddress + uint64_t(element) * uint64_t(elementSizeBytes);
}

#if SLANGML_INDEX64
public uint64_t slangml_final_byte_address64(TensorDesc tensor, int64_t relativeElement, uint elementSizeBytes)
{
    int64_t element = tensor.baseOffset + relativeElement;
    return tensor.dataAddress + uint64_t(element) * uint64_t(elementSizeBytes);
}
#endif

public bool slangml_can_vectorize4(TensorDesc tensor, int relativeElement, uint elementSizeBytes)
{
    return elementSizeBytes == 4u
        && (tensor.flags & SLANGML_FLAG_ALIGNED16) != 0u
        && slangml_is_aligned_vec4(slangml_final_byte_address(tensor, relativeElement, elementSizeBytes));
}

// BDA accessors.  They compile to physical-storage-buffer loads/stores in the
// default path.  The bound-buffer helpers below are only for small-tensor
// fallback kernels and are never selected by the fast BDA variants.
public uint slangml_bda_load_u32(uint64_t address)
{
    uint* pointer = reinterpret<uint*>(address);
    return pointer[0];
}

public uint4 slangml_bda_load_u32x4(uint64_t address)
{
    uint4* pointer = reinterpret<uint4*>(address);
    return pointer[0];
}

public float slangml_bda_load_f32(uint64_t address)
{
    return asfloat(slangml_bda_load_u32(address));
}

public float4 slangml_bda_load_f32x4(uint64_t address)
{
    return asfloat(slangml_bda_load_u32x4(address));
}

public void slangml_bda_store_u32(uint64_t address, uint value)
{
    uint* pointer = reinterpret<uint*>(address);
    pointer[0] = value;
}

public void slangml_bda_store_u32x4(uint64_t address, uint4 value)
{
    uint4* pointer = reinterpret<uint4*>(address);
    pointer[0] = value;
}

public void slangml_bda_store_f32(uint64_t address, float value)
{
    slangml_bda_store_u32(address, asuint(value));
}

public void slangml_bda_store_f32x4(uint64_t address, float4 value)
{
    slangml_bda_store_u32x4(address, asuint(value));
}

public uint4 slangml_bound_load_u32x4(ByteAddressBuffer buffer, uint byteOffset)
{
    return buffer.Load4(byteOffset);
}

public void slangml_bound_store_u32x4(RWByteAddressBuffer buffer, uint byteOffset, uint4 value)
{
    buffer.Store4(byteOffset, value);
}

public float4 slangml_bound_load_f32x4(ByteAddressBuffer buffer, uint byteOffset)
{
    return asfloat(buffer.Load4(byteOffset));
}

public void slangml_bound_store_f32x4(RWByteAddressBuffer buffer, uint byteOffset, float4 value)
{
    buffer.Store4(byteOffset, asuint(value));
}
