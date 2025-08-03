# mmops - Multimedia Operators

Zero-cost typed SIMD wrapper for Nim providing high-performance vector operations using AVX2 intrinsics.

## Overview

**mmops** is a comprehensive SIMD library that provides a safe, typed interface to AVX2 vector operations. It offers zero-overhead abstraction over Intel's AVX2 intrinsics while maintaining Nim's type safety and expressiveness.

### Key Features

- **Type-Safe SIMD**: Strongly typed vectors with compile-time width and element type checking
- **Zero-Cost Abstraction**: Direct mapping to AVX2 intrinsics with no runtime overhead  
- **Comprehensive Operations**: Full coverage of arithmetic, logical, comparison, and specialized SIMD operations
- **Array Integration**: Seamless conversion between Nim arrays and SIMD vectors
- **Multiple Data Types**: Support for float32/64, int8/16/32/64, and uint8/16/32/64
- **Advanced Features**: FMA, gather/scatter, saturated arithmetic, and horizontal operations

## Type System

The library centers around the `Mm[N, T]` type where:
- `N` specifies vector width: `w8` (32×8-bit), `w16` (16×16-bit), `w32` (8×32-bit), `w64` (4×64-bit)
- `T` specifies element type: `float32`, `float64`, `int8`-`int64`, `uint8`-`uint64`

## Quick Start

```nim
import mmops

# Create vectors from arrays
let a = load([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f])
let b = load([8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f])

# Perform SIMD operations
let sum = a + b
let product = a * b
let result = fma(a, b, sum)  # a * b + sum

# Array-style access
echo result[0]  # Access individual elements
result[0] = 42.0f  # Modify elements

# Convert back to array
let output = store(result)
```

## Core Operations

### Vector Creation
- `load()` - Load from arrays
- `splat()` - Broadcast single value
- `set()` - Set individual elements  
- `zero()` - All-zeros vector

### Arithmetic
- Standard operators: `+`, `-`, `*`, `/`
- `fma()` - Fused multiply-add
- `abs()`, `sqrt()`, `fastSqrt()`
- Rounding: `round()`, `floor()`, `ceil()`

### Comparisons
- Standard operators: `==`, `<`, `>`, `<=`, `>=`
- `min()`, `max()`

### Bitwise & Logical
- `and`, `or`, `xor`, `not`
- `andnot()` - AND with NOT
- Shifts: `shl`, `shr`, `vshl()`, `vshr()`

### Advanced Operations
- Horizontal: `sum()`, `hadd()`, `hsub()`
- Packing: `pack()`, `packus()`, `unpack*()`
- Blending: `blend()`, `blendv()`
- Gather/scatter operations for indirect memory access
- Saturated arithmetic for overflow protection

## Performance Notes

- Array-style access (`[]`) is provided for convenience but frequent use may impact performance
- For intensive single-element updates, consider maintaining local variables and using `load()`
- All operations compile directly to AVX2 instructions with no runtime overhead

## Documentation

For complete API reference, see: [https://capocasa.github.com/](https://example.com/mmops-docs)

## Requirements

- Nim compiler with AVX2 support
- CPU with AVX2 instruction set
- nimsimd library for low-level intrinsics

## License

Licensed under MIT licencse
