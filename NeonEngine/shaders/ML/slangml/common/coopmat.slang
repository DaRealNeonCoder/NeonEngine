implementing slangml_common;

#ifndef SLANGML_ENABLE_COOPMAT
#define SLANGML_ENABLE_COOPMAT 0
#endif

public struct CoopMatConfig
{
    public uint rows;
    public uint cols;
    public uint depth;
    public uint subgroupSize;
    public uint stages;
    public bool enabled;
};

public CoopMatConfig slangml_coopmat_config(uint rows, uint cols, uint depth, uint subgroupSize, uint stages)
{
    CoopMatConfig result;
    result.rows = rows;
    result.cols = cols;
    result.depth = depth;
    result.subgroupSize = subgroupSize;
    result.stages = stages;
#if SLANGML_ENABLE_COOPMAT
    result.enabled = true;
#else
    result.enabled = false;
#endif
    return result;
}

// This is the portable accumulator used by every cooperative-matrix wrapper.
// A cooperative specialization can replace the load/mma/store methods while
// preserving this ABI and the same tile dimensions.
public struct PortableMatrixTileF32<int Rows, int Cols>
{
    float value[Rows * Cols];

    [ForceInline]
    public float get(uint row, uint col)
    {
        return value[row * Cols + col];
    }

    [ForceInline]
    [mutating]
    public void set(uint row, uint col, float v)
    {
        value[row * Cols + col] = v;
    }
};

public void slangml_portable_tile_zero<int Rows, int Cols>(out PortableMatrixTileF32<Rows, Cols> tile)
{
    for (uint row = 0; row < Rows; ++row)
        for (uint col = 0; col < Cols; ++col)
            tile.set(row, col, 0.0f);
}

public void slangml_portable_tile_fma<int Rows, int Cols, int Depth>(
    inout PortableMatrixTileF32<Rows, Cols> accumulator,
    float a[Rows * Depth],
    float b[Depth * Cols])
{
    for (uint row = 0; row < Rows; ++row)
        for (uint col = 0; col < Cols; ++col)
        {
            float sum = accumulator.get(row, col);
            for (uint k = 0; k < Depth; ++k)
                sum += a[row * Depth + k] * b[k * Cols + col];
            accumulator.set(row, col, sum);
        }
}

#if SLANGML_ENABLE_COOPMAT
// Slang 2026 exposes Vulkan cooperative matrices through the built-in `linalg`
// namespace.  Keep the wrapper opt-in so portable builds never require the
// extension.
using namespace linalg;

public typealias SlangMLCoopMatF16A<int Rows, int Cols> =
    CoopMat<uint16_t, MemoryScope.Subgroup, Rows, Cols, CoopMatMatrixUse.MatrixA>;
public typealias SlangMLCoopMatF16B<int Rows, int Cols> =
    CoopMat<uint16_t, MemoryScope.Subgroup, Rows, Cols, CoopMatMatrixUse.MatrixB>;
public typealias SlangMLCoopMatF32Accumulator<int Rows, int Cols> =
    CoopMat<uint32_t, MemoryScope.Subgroup, Rows, Cols, CoopMatMatrixUse.MatrixAccumulator>;
#endif
