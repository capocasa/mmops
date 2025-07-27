# ARM NEON Nim Wrapper

A comprehensive Nim wrapper for all ARM NEON instructions with convenient operator overloads and sequence-like operations.

## Perfect for Apple Silicon Macs!

Current Macs (M1/M2/M3/M4) use ARM64 processors with NEON SIMD extensions. This wrapper provides native acceleration for Apple Silicon and all modern ARM processors.

## Features

- **Complete NEON coverage**: Wrappers for all major NEON instruction categories
- **Dual vector sizes**: Both 64-bit and 128-bit vector support
- **Operator overloads**: Natural arithmetic (`+`, `-`, `*`, `/`) and logical (`and`, `or`, `xor`, `not`) operators
- **Comparison operators**: `==`, `<`, `>` for vector comparisons
- **Sequtils-style operations**: `sum`, `avg`, `min`, `max` for horizontal operations
- **Multi-precision support**: Operations for 8, 16, 32, and 64-bit integers
- **Float support**: Both single (`float32x4_t`) and double (`float64x2_t`) precision floating point

## Vector Types

### 128-bit vectors (quad - most common):
- `int8x16_t`: 16 x 8-bit integers
- `int16x8_t`: 8 x 16-bit integers
- `int32x4_t`: 4 x 32-bit integers  
- `int64x2_t`: 2 x 64-bit integers
- `float32x4_t`: 4 x single-precision floats
- `float64x2_t`: 2 x double-precision floats

### 64-bit vectors (dual):
- `int8x8_t`: 8 x 8-bit integers
- `int16x4_t`: 4 x 16-bit integers
- `int32x2_t`: 2 x 32-bit integers
- `int64x1_t`: 1 x 64-bit integer
- `float32x2_t`: 2 x single-precision floats
- `float64x1_t`: 1 x double-precision float

## Basic Usage

```nim
import neon

# Create 128-bit vectors (4 elements)
let a = vec32(10)  # [10, 10, 10, 10]
let b = vec32(5)   # [5, 5, 5, 5]

# Arithmetic operations
let sum_result = a + b     # [15, 15, 15, 15]
let diff = a - b           # [5, 5, 5, 5]
let product = a * b        # [50, 50, 50, 50]

# Horizontal operations
echo sum(a)                # 40 (10 * 4)
echo avg(a)                # 10.0
echo min(a)                # 10
echo max(a)                # 10
```

## Instruction Categories Covered

### Arithmetic Operations
- Addition: `add8`, `add16`, `add32`, `add64`, `+`
- Subtraction: `sub8`, `sub16`, `sub32`, `sub64`, `-`
- Multiplication: `mul8`, `mul16`, `mul32`, `*`

### Logical Operations
- Bitwise AND: `and`
- Bitwise OR: `or`
- Bitwise XOR: `xor`
- Bitwise NOT: `not`

### Comparison Operations
- Equality: `==`
- Greater than: `>`
- Less than: `<`
- Min/Max: `vmin`, `vmax`

### Shift Operations
- Left shift: `shl`
- Right shift: `shr`

### Load/Store Operations
- Load: `vld1`, `vld1q`
- Store: `vst1`, `vst1q`

### Horizontal Operations
- Vector reduction: `vaddv`, `vaddvq`

## Type-Specific Operations

```nim
# 8-bit operations (16 elements in 128-bit vector)
let data8a = vec8(100)
let data8b = vec8(50)
let result8 = add8(data8a, data8b)

# 16-bit operations (8 elements in 128-bit vector)
let data16a = vec16(1000)
let data16b = vec16(500)
let result16 = add16(data16a, data16b)
let product16 = mul16(data16a, data16b)

# 64-bit operations (2 elements in 128-bit vector)
let data64a = vec64(1000000)
let data64b = vec64(500000)
let result64 = add64(data64a, data64b)
```

## Float Operations

```nim
# Single precision (4 elements)
let fa = vecf(3.14'f32)
let fb = vecf(2.71'f32)
let sum_f = fa + fb
let product_f = fa * fb
let quotient_f = fa / fb

# Double precision (2 elements)
let da = vecd(3.14159)
let db = vecd(2.71828)
let sum_d = da + db
let quotient_d = da / db
```

## 64-bit vs 128-bit Vectors

```nim
# 128-bit vectors (default - more parallel processing)
let big_vec = vec32(42)     # 4 x 32-bit integers
echo sum(big_vec)           # 168

# 64-bit vectors (sometimes faster for certain operations)
let small_vec = vec32_64(42) # 2 x 32-bit integers
# Note: 64-bit vectors have different function names
```

## Utility Functions

```nim
# Vector constructors (128-bit)
let eights = vec8(42)       # 16 x 42 (8-bit)
let sixteens = vec16(1000)  # 8 x 1000 (16-bit)
let thirtytwos = vec32(50)  # 4 x 50 (32-bit)
let sixtyfours = vec64(1M)  # 2 x 1M (64-bit)
let floats = vecf(3.14'f32) # 4 x 3.14 (float32)
let doubles = vecd(2.71)    # 2 x 2.71 (float64)

# Vector constructors (64-bit)
let small_eights = vec8_64(42)     # 8 x 42 (8-bit)
let small_floats = vecf_64(3.14'f32) # 2 x 3.14 (float32)

# Horizontal reductions
echo sum(thirtytwos)        # Sum all elements
echo avg(thirtytwos)        # Average of all elements
echo min(thirtytwos)        # Minimum element
echo max(thirtytwos)        # Maximum element
```

## Performance Notes

- All operations compile to direct NEON instructions
- Operator overloads have zero overhead
- 128-bit vectors generally provide better throughput
- 64-bit vectors may be faster for certain specific operations
- Use type-specific functions (`add8`, `add16`, etc.) when precision matters
- Horizontal operations may be slower than vertical operations

## Apple Silicon Specific

- Apple Silicon (M1/M2/M3/M4) fully supports all NEON instructions
- Additional Apple-specific optimizations are available but not covered here
- NEON provides the baseline SIMD that works across all ARM processors
- Excellent performance on current Mac hardware

## Requirements

- ARM64 processor (Apple Silicon Macs, modern ARM servers, Raspberry Pi 4+, etc.)
- Nim compiler with C backend
- GCC or Clang with ARM NEON support

## Compilation

### On Apple Silicon Macs (native):
```bash
nim c -d:release your_file.nim
```

### Cross-compilation to ARM64:
```bash
# For Apple Silicon Macs (from other platforms)
nim c -d:release --cpu:arm64 --os:macosx --passC:"-march=armv8-a+simd" your_file.nim

# For ARM64 Linux (from other platforms) 
nim c -d:release --cpu:arm64 --os:linux --passC:"-march=armv8-a+simd" your_file.nim
```

### For optimal Apple Silicon performance:
```bash
nim c -d:release --passC:"-mcpu=apple-m1 -march=armv8-a+simd" your_file.nim
```

**Note**: Cross-compilation from x86_64 to ARM64 may require proper ARM64 toolchain setup. The module is designed to work perfectly when compiled natively on ARM64 systems.

## Testing

Run the test suite:
```bash
nim c -d:release -r test_neon.nim
```

Run the example:
```bash
nim c -d:release -r example_neon.nim
```

## Differences from x86 AVX2

- NEON vectors are smaller (128-bit max vs 256-bit AVX2)
- NEON has both 64-bit and 128-bit variants
- Some operations have different naming conventions
- NEON is more consistent across different ARM implementations
- Generally lower power consumption than x86 SIMD