## N8: Universal SIMD Interface for Nim
## 
## Provides unified 128-bit SIMD operations that work on both x86_64 (AVX2) and ARM64 (NEON).
## The name "n8" refers to "nim eight floats" - the 128-bit register size.

when defined(arm) or defined(arm64) or defined(aarch64):
  import neon
  
  # Universal vector types - mapped to NEON 128-bit types
  type
    N8Vector32* = int32x4_t     ## 4x 32-bit integers
    N8VectorF32* = float32x4_t  ## 4x 32-bit floats
    N8VectorF64* = float64x2_t  ## 2x 64-bit floats
  
  # Vector creation
  proc n8_set1_i32*(val: int32): N8Vector32 {.inline.} = vec32(val)
  proc n8_set1_f32*(val: float32): N8VectorF32 {.inline.} = vecf(val)
  proc n8_set1_f64*(val: float64): N8VectorF64 {.inline.} = vecd(val)
  
  # Arithmetic operations
  proc n8_add*(a, b: N8Vector32): N8Vector32 {.inline.} = a + b
  proc n8_add*(a, b: N8VectorF32): N8VectorF32 {.inline.} = a + b
  proc n8_add*(a, b: N8VectorF64): N8VectorF64 {.inline.} = a + b
  
  proc n8_sub*(a, b: N8Vector32): N8Vector32 {.inline.} = a - b
  proc n8_sub*(a, b: N8VectorF32): N8VectorF32 {.inline.} = a - b
  proc n8_sub*(a, b: N8VectorF64): N8VectorF64 {.inline.} = a - b
  
  proc n8_mul*(a, b: N8Vector32): N8Vector32 {.inline.} = a * b
  proc n8_mul*(a, b: N8VectorF32): N8VectorF32 {.inline.} = a * b
  proc n8_mul*(a, b: N8VectorF64): N8VectorF64 {.inline.} = a * b
  
  # Logical operations
  proc n8_and*(a, b: N8Vector32): N8Vector32 {.inline.} = a and b
  proc n8_or*(a, b: N8Vector32): N8Vector32 {.inline.} = a or b
  proc n8_xor*(a, b: N8Vector32): N8Vector32 {.inline.} = a xor b
  proc n8_not*(a: N8Vector32): N8Vector32 {.inline.} = not a
  
  # Horizontal operations (reductions)
  proc n8_sum*(a: N8Vector32): int64 {.inline.} = sum(a)
  proc n8_sum*(a: N8VectorF32): float32 {.inline.} = sum(a)
  proc n8_sum*(a: N8VectorF64): float64 {.inline.} = sum(a)
  
  proc n8_min*(a: N8Vector32): int32 {.inline.} = min(a)
  proc n8_max*(a: N8Vector32): int32 {.inline.} = max(a)

elif defined(amd64) or defined(i386):
  import avx2
  
  # Universal vector types - mapped to AVX2 types (using lower 128 bits)
  type
    N8Vector32* = m256i         ## 4x 32-bit integers (using lower 128 bits)
    N8VectorF32* = m256         ## 4x 32-bit floats (using lower 128 bits)
    N8VectorF64* = m256d        ## 2x 64-bit floats (using lower 128 bits)
  
  # Vector creation
  proc n8_set1_i32*(val: int32): N8Vector32 {.inline.} = vec32(val)
  proc n8_set1_f32*(val: float32): N8VectorF32 {.inline.} = vecf(val)
  proc n8_set1_f64*(val: float64): N8VectorF64 {.inline.} = vecd(val)
  
  # Arithmetic operations
  proc n8_add*(a, b: N8Vector32): N8Vector32 {.inline.} = a + b
  proc n8_add*(a, b: N8VectorF32): N8VectorF32 {.inline.} = a + b
  proc n8_add*(a, b: N8VectorF64): N8VectorF64 {.inline.} = a + b
  
  proc n8_sub*(a, b: N8Vector32): N8Vector32 {.inline.} = a - b
  proc n8_sub*(a, b: N8VectorF32): N8VectorF32 {.inline.} = a - b
  proc n8_sub*(a, b: N8VectorF64): N8VectorF64 {.inline.} = a - b
  
  proc n8_mul*(a, b: N8Vector32): N8Vector32 {.inline.} = a * b
  proc n8_mul*(a, b: N8VectorF32): N8VectorF32 {.inline.} = a * b
  proc n8_mul*(a, b: N8VectorF64): N8VectorF64 {.inline.} = a * b
  
  # Logical operations
  proc n8_and*(a, b: N8Vector32): N8Vector32 {.inline.} = a and b
  proc n8_or*(a, b: N8Vector32): N8Vector32 {.inline.} = a or b
  proc n8_xor*(a, b: N8Vector32): N8Vector32 {.inline.} = a xor b
  proc n8_not*(a: N8Vector32): N8Vector32 {.inline.} = not a
  
  # Horizontal operations (reductions)
  proc n8_sum*(a: N8Vector32): int64 {.inline.} = sum(a)
  proc n8_sum*(a: N8VectorF32): float32 {.inline.} = sum(a)
  proc n8_sum*(a: N8VectorF64): float64 {.inline.} = sum(a)
  
  proc n8_min*(a: N8Vector32): int32 {.inline.} = min(a)
  proc n8_max*(a: N8Vector32): int32 {.inline.} = max(a)

else:
  {.error: "N8 requires either ARM64/NEON or x86_64/AVX2 support".}

# Convenience operator overloads for natural syntax
template `+`*(a, b: N8Vector32): N8Vector32 = n8_add(a, b)
template `+`*(a, b: N8VectorF32): N8VectorF32 = n8_add(a, b)
template `+`*(a, b: N8VectorF64): N8VectorF64 = n8_add(a, b)

template `-`*(a, b: N8Vector32): N8Vector32 = n8_sub(a, b)
template `-`*(a, b: N8VectorF32): N8VectorF32 = n8_sub(a, b)
template `-`*(a, b: N8VectorF64): N8VectorF64 = n8_sub(a, b)

template `*`*(a, b: N8Vector32): N8Vector32 = n8_mul(a, b)
template `*`*(a, b: N8VectorF32): N8VectorF32 = n8_mul(a, b)
template `*`*(a, b: N8VectorF64): N8VectorF64 = n8_mul(a, b)

# Architecture detection at compile time
const N8_ARCH* = when defined(arm) or defined(arm64) or defined(aarch64):
  "NEON"
elif defined(amd64) or defined(i386):
  "AVX2" 
else:
  "UNSUPPORTED"

# Vector size constants
const 
  N8_VECTOR32_SIZE* = 4  ## Number of 32-bit elements
  N8_VECTORF32_SIZE* = 4 ## Number of float32 elements  
  N8_VECTORF64_SIZE* = 2 ## Number of float64 elements