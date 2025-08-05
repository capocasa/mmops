# mmops - easy SIMD for Nim

mmops is a very thin (only template and aliases) type-safe wrapper for nimsimd.

The interface handles fixed-size array. To perform calculation, math operators (`+`, `-`, `*`, `/`, etc.) or very simple procedure names are used, which are applied to all elements.

The 

## Usage

```nim
import mmops

# Load data into register from array
let a = load([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f])

# load data into register from individual values
let b = load(8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f)

# Perform SIMD operations
let sum = a + b
let product = a * b
let result = fma(a, b, sum)  # a * b + sum

# Array-style access
# caution, poor performance
echo result[0]  # Access individual elements
result[0] = 42.0f  # Modify elements

# Convert back to array
let output = store(result)
```

## Basics

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

## Documentation

For complete API reference, see: [Full documentation](https://capocasa.github.com/mmops/mmops.html)

## Requirements

- Nim compiler with AVX2 support
- CPU with AVX2 instruction set
- nimsimd library for low-level intrinsics

## Internals

This library was created with a tightly controlled LLM.

## Limitations

This is currently AVX2 only, so no raspberry pi or native macs- yet.

## License

Licensed under MIT licencse
