implementing slangml_common;

#ifndef SLANGML_DEBUG_BOUNDS
#define SLANGML_DEBUG_BOUNDS 0
#endif

public bool slangml_bounds_ok(bool condition)
{
#if SLANGML_DEBUG_BOUNDS
    return condition;
#else
    return true;
#endif
}

public bool slangml_index_ok(uint64_t index, uint64_t count)
{
#if SLANGML_DEBUG_BOUNDS
    return index < count;
#else
    return true;
#endif
}

// Use this in a void kernel after computing a logical index.  In release
// variants it compiles to a constant-true branch; in debug variants it makes
// the out-of-range path explicit and easy to trap in a caller.
#if SLANGML_DEBUG_BOUNDS
#define SLANGML_BOUNDS_GUARD(condition) if (!(condition)) return
#else
#define SLANGML_BOUNDS_GUARD(condition)
#endif
