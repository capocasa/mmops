# mmops - easy SIMD for Nim

mmops is a very thin type-safe convenience wrapper for nimsimd.

The YMM-registers used in AVX and AVX2 instructions are presented using the `Mm[N, T]` object with an interface similar to a fixed size array with additional vector math operators.

Only aliases and templates are used- the functionality and performance maps exactly to `nimsimsd`. It's just easier to read- and to translate normal arithmetic.

## Installation

```
$ nimble install mmops
```

## Usage

```nim
import mmops

# Load data into register from array
let a = load([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f])

# load data into register from individual values
let b = set(8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f)

# Perform SIMD operations
let sum = a + b
let product = a * b
var ma = fma(a, b, sum)  # a * b + sum

# Array-style access
# caution, poor performance
echo ma[0]  # Access individual elements
ma[0] = 42.0f  # Modify elements

# Convert back to array
let output = store(ma)

echo $output
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

This is an AI-assisted project- the design is human, the implementation is pretty much iterative AI-human teamwork.

## Limitations

This is currently AVX2 only, so no raspberry pi or native macs- yet.

## Future

I would very much like to add an 128-bit Neon version. One could emulate 256-bit Neon if feeling *really* fancy.

## License

mmops is under the MIT license. Please do get rich with it, if you can.

