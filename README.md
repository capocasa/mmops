# AVX2 Nim Wrapper

A comprehensive Nim wrapper for all AVX2 instructions with convenient operator overloads and sequence-like operations.

## Features

- **Complete AVX2 coverage**: Wrappers for all major AVX2 instruction categories
- **Operator overloads**: Natural arithmetic (`+`, `-`, `*`, `/`) and logical (`and`, `or`, `xor`, `not`) operators
- **Comparison operators**: `==`, `<`, `>` for vector comparisons
- **Sequtils-style operations**: `sum`, `avg`, `min`, `max` for horizontal operations
- **Multi-precision support**: Operations for 8, 16, 32, and 64-bit integers
- **Float support**: Both single (`m256`) and double (`m256d`) precision floating point

## Types

- `m256i`: 256-bit integer vector (32 x 8-bit, 16 x 16-bit, 8 x 32-bit, or 4 x 64-bit)
- `m256`: 256-bit single-precision float vector (8 x float32)
- `m256d`: 256-bit double-precision float vector (4 x float64)

## Basic Usage

```nim
import avx2

# Create vectors
let a = vec32(10)  # [10, 10, 10, 10, 10, 10, 10, 10]
let b = vec32(5)   # [5, 5, 5, 5, 5, 5, 5, 5]

# Arithmetic operations
let sum_result = a + b     # [15, 15, 15, 15, 15, 15, 15, 15]
let diff = a - b           # [5, 5, 5, 5, 5, 5, 5, 5]
let product = a * b        # [50, 50, 50, 50, 50, 50, 50, 50]

# Horizontal operations
echo sum(a)                # 80 (10 * 8)
echo avg(a)                # 10.0
echo min(a)                # 10
echo max(a)                # 10
```

## Instruction Categories Covered

### Arithmetic Operations
- Addition: `add8`, `add16`, `add32`, `add64`, `+`
- Subtraction: `sub8`, `sub16`, `sub32`, `sub64`, `-`
- Multiplication: `mul16`, `mul32`, `*`
- Saturated arithmetic: `adds_epi8/16`, `subs_epi8/16`
- Multiply-accumulate: `madd_epi16`, `maddubs_epi16`

### Logical Operations
- Bitwise AND: `and`
- Bitwise OR: `or`
- Bitwise XOR: `xor`
- Bitwise NOT: `not`
- AND-NOT: `andnot`

### Comparison Operations
- Equality: `==`
- Greater than: `>`
- Less than: `<`
- Min/Max: `min`, `max`

### Shift Operations
- Left shift: `shl`
- Right shift: `shr`
- Arithmetic right shift: `sra`

### Shuffle and Permute
- `shuffle_epi8`, `shuffle_epi32`
- `unpack` operations for all data types
- `pack` operations with saturation

### Load/Store Operations
- Aligned: `load_si256`, `store_si256`
- Unaligned: `loadu_si256`, `storeu_si256`

### Conversion Operations
- Sign extension: `cvtepi8_epi16`, `cvtepi16_epi32`, etc.
- Zero extension: `cvtepu8_epi16`, `cvtepu16_epi32`, etc.

### Horizontal Operations
- Horizontal add/subtract: `hadd`, `hsub`
- Saturated horizontal operations: `hadds`, `hsubs`

### Advanced Operations
- Blend operations with masks
- Broadcast operations
- Gather operations for scattered memory access
- Extract/Insert for individual element access

## Type-Specific Operations

```nim
# 8-bit operations (32 elements)
let data8a = vec8(100)
let data8b = vec8(50)
let result8 = add8(data8a, data8b)

# 16-bit operations (16 elements)
let data16a = vec16(1000)
let data16b = vec16(500)
let result16 = add16(data16a, data16b)
let product16 = mul16(data16a, data16b)

# 64-bit operations (4 elements)
let data64a = vec64(1000000)
let data64b = vec64(500000)
let result64 = add64(data64a, data64b)
```

## Float Operations

```nim
# Single precision (8 elements)
let fa = mm256_set1_ps(3.14)
let fb = mm256_set1_ps(2.71)
let sum_f = fa + fb
let product_f = fa * fb

# Double precision (4 elements)
let da = mm256_set1_pd(3.14159)
let db = mm256_set1_pd(2.71828)
let sum_d = da + db
let quotient_d = da / db
```

## Utility Functions

```nim
# Vector constructors
let zeros = zero()          # All zeros
let eights = vec8(42)       # 32 x 42 (8-bit)
let sixteens = vec16(1000)  # 16 x 1000 (16-bit)
let thirtytwos = vec32(50)  # 8 x 50 (32-bit)
let sixtyfours = vec64(1M)  # 4 x 1M (64-bit)

# Horizontal reductions
echo sum(thirtytwos)        # Sum all elements
echo avg(thirtytwos)        # Average of all elements
echo min(thirtytwos)        # Minimum element
echo max(thirtytwos)        # Maximum element
```

## Performance Notes

- All operations compile to direct AVX2 instructions
- Operator overloads have zero overhead
- Use type-specific functions (`add8`, `add16`, etc.) when precision matters
- Horizontal operations may be slower than vertical operations
- Prefer aligned memory access when possible

## Requirements

- CPU with AVX2 support (Intel Haswell+ or AMD Excavator+)
- Nim compiler with C backend
- GCC or Clang with AVX2 support

## Compilation

AVX2 instructions require specific CPU features and compiler flags. Always compile with:

```bash
nim c -d:release --passC:"-mavx2" your_file.nim
```

## Testing

Run the test suite:
```bash
nim c -d:release --passC:"-mavx2" -r test_avx2.nim
```

Run the example:
```bash
nim c -d:release --passC:"-mavx2" -r example.nim
```