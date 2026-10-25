implementing slangml_common;

// Performance-first variant switches.  These are part of the pipeline variant
// key, never runtime state.  Release builds do not retain debug branches.
#ifndef SLANGML_USE_BDA
#define SLANGML_USE_BDA 1
#endif

#ifndef SLANGML_ENABLE_SUBGROUP_OPS
#define SLANGML_ENABLE_SUBGROUP_OPS 1
#endif

#ifndef SLANGML_ENABLE_COOPMAT
#define SLANGML_ENABLE_COOPMAT 0
#endif

#ifndef SLANGML_DETERMINISTIC
#define SLANGML_DETERMINISTIC 0
#endif

#ifndef SLANGML_INDEX64
#define SLANGML_INDEX64 0
#endif

#ifndef SLANGML_BOUND_BUFFER_FALLBACK
#define SLANGML_BOUND_BUFFER_FALLBACK 0
#endif

#ifndef SLANGML_FAST_MATH
#define SLANGML_FAST_MATH 1
#endif

#ifndef SLANGML_DEBUG_BOUNDS
#define SLANGML_DEBUG_BOUNDS 0
#endif

public static const uint SLANGML_MAX_RANK = 8;
public static const uint SLANGML_MAX_SUBGROUP_LANES = 128;
public static const uint SLANGML_MAX_WORKGROUP_LANES = 1024;
public static const uint SLANGML_TENSOR_DESC_BYTES = 96;
public static const uint SLANGML_KERNEL_HEADER_BYTES = 48;
public static const uint SLANGML_DISPATCH_BASE_BYTES = 16;

public static const uint SLANGML_FLAG_CONTIGUOUS = 1u << 0;
public static const uint SLANGML_FLAG_READ_ONLY = 1u << 1;
public static const uint SLANGML_FLAG_WRITE_ONLY = 1u << 2;
public static const uint SLANGML_FLAG_BROADCAST_VIEW = 1u << 3;
public static const uint SLANGML_FLAG_ALIGNED16 = 1u << 4;
public static const uint SLANGML_FLAG_RANK_LE3 = 1u << 5;
public static const uint SLANGML_FLAG_INNER_STRIDE1 = 1u << 6;
public static const uint SLANGML_FLAG_CANONICAL = 1u << 7;

public static const uint SLANGML_KERNEL_FLAG_GUARDED_BOUNDS = 1u << 0;
public static const uint SLANGML_KERNEL_FLAG_DETERMINISTIC = 1u << 1;
public static const uint SLANGML_KERNEL_FLAG_HAS_WORKSPACE = 1u << 2;

public static const uint SLANGML_INVALID_U32 = 0xFFFFFFFFu;
public static const int64_t SLANGML_INVALID_I64 = int64_t(-1);

// The host sets this specialization constant to 8, 16, 32, or 64 when the
// pipeline uses required-subgroup-size control.
[vk::constant_id(0)]
public const uint SLANGML_SUBGROUP_SIZE = 32u;
