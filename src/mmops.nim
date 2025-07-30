{.passC: "-mavx2 -mfma".}
import nimsimd/avx2
import nimsimd/fma
export avx2

## mmops - Multimedia Operators
## Zero-cost typed SIMD wrapper

# --- Comparison Predicates ---
const
  CMP_EQ_OQ* = 0x00
  CMP_LT_OS* = 0x01
  CMP_LE_OS* = 0x02
  CMP_UNORD_Q* = 0x03
  CMP_NEQ_UQ* = 0x04
  CMP_NLT_US* = 0x05
  CMP_NLE_US* = 0x06
  CMP_ORD_Q* = 0x07

# --- Type System ---
type 
  Width* = enum
    w64 = 4   ## 4 × 64-bit elements (256 bits total)
    w32 = 8   ## 8 × 32-bit elements (256 bits total)
    w16 = 16  ## 16 × 16-bit elements (256 bits total)
    w8 = 32   ## 32 × 8-bit elements (256 bits total)
  
  Mm*[N: static Width, T: SomeNumber] = distinct (
    when T is float32: M256 
    elif T is float64: M256d 
    else: M256i
  )


# --- Array Conversion ---
proc load*[T](p: array[32, T]): Mm[w8, T] =
  ## Load array into Mm vector (assumes 8-bit elements = 32 elements)
  Mm[w8, T](mm256_loadu_si256(cast[pointer](unsafeAddr p[0])))

proc load*[T](p: array[16, T]): Mm[w16, T] =
  ## Load array into Mm vector (assumes 16-bit elements = 16 elements)
  Mm[w16, T](mm256_loadu_si256(cast[pointer](unsafeAddr p[0])))

proc load*[T](p: array[8, T]): Mm[w32, T] =
  ## Load array into Mm vector (assumes 32-bit elements = 8 elements)
  when T is float32:
    Mm[w32, T](mm256_loadu_ps(cast[pointer](unsafeAddr p[0])))
  else:
    Mm[w32, T](mm256_loadu_si256(cast[pointer](unsafeAddr p[0])))

proc load*[T](p: array[4, T]): Mm[w64, T] =
  ## Load array into Mm vector (assumes 64-bit elements = 4 elements)
  when T is float64:
    Mm[w64, T](mm256_loadu_pd(cast[pointer](unsafeAddr p[0])))
  else: # 64-bit integers
    Mm[w64, T](mm256_loadu_si256(cast[pointer](unsafeAddr p[0])))

template store*[T](m: Mm[w32, T]): array[8, T] =
  ## Store Mm vector to array (32-bit elements)
  var result: array[8, T]
  when T is float32:
    mm256_storeu_ps(cast[pointer](addr result[0]), M256(m))
  elif T is float64:
    mm256_storeu_pd(cast[pointer](addr result[0]), M256d(m))
  else:
    mm256_storeu_si256(cast[pointer](addr result[0]), M256i(m))
  result

template store*[T](m: Mm[w64, T]): array[4, T] =
  ## Store Mm vector to array (64-bit elements)
  var result: array[4, T]
  when T is float64:
    mm256_storeu_pd(cast[pointer](addr result[0]), M256d(m))
  else: # 64-bit integers
    mm256_storeu_si256(cast[pointer](addr result[0]), M256i(m))
  result

template store*[T](m: Mm[w16, T]): array[16, T] =
  ## Store Mm vector to array (16-bit elements)
  var result: array[16, T]
  mm256_storeu_si256(cast[pointer](addr result[0]), M256i(m))
  result

template store*[T](m: Mm[w8, T]): array[32, T] =
  ## Store Mm vector to array (8-bit elements)
  var result: array[32, T]
  mm256_storeu_si256(cast[pointer](addr result[0]), M256i(m))
  result

# --- Vector Initialization ---
template splat*[N](val: float32): Mm[N, float32] = 
  ## Broadcast a single-precision floating-point value to all elements
  Mm[N, float32](mm256_set1_ps(val))

template splat*[N](val: float64): Mm[N, float64] = 
  ## Broadcast a double-precision floating-point value to all elements
  Mm[N, float64](mm256_set1_pd(val))

template splat*[N](val: int8): Mm[N, int8] = 
  ## Broadcast an 8-bit integer value to all elements
  Mm[N, int8](mm256_set1_epi8(val))

template splat*[N](val: int16): Mm[N, int16] = 
  ## Broadcast a 16-bit integer value to all elements
  Mm[N, int16](mm256_set1_epi16(val))

template splat*[N](val: int32): Mm[N, int32] = 
  ## Broadcast a 32-bit integer value to all elements
  Mm[N, int32](mm256_set1_epi32(val))

template splat*[N](val: int64): Mm[N, int64] = 
  ## Broadcast a 64-bit integer value to all elements
  Mm[N, int64](mm256_set1_epi64x(val))

template splat*[N](val: uint8): Mm[N, uint8] = 
  ## Broadcast an 8-bit unsigned integer value to all elements
  Mm[N, uint8](mm256_set1_epi8(cast[int8](val)))

template splat*[N](val: uint16): Mm[N, uint16] = 
  ## Broadcast a 16-bit unsigned integer value to all elements
  Mm[N, uint16](mm256_set1_epi16(cast[int16](val)))

template splat*[N](val: uint32): Mm[N, uint32] = 
  ## Broadcast a 32-bit unsigned integer value to all elements
  Mm[N, uint32](mm256_set1_epi32(cast[int32](val)))

template splat*[N](val: uint64): Mm[N, uint64] = 
  ## Broadcast a 64-bit unsigned integer value to all elements
  Mm[N, uint64](mm256_set1_epi64x(cast[int64](val)))

template set*(e7, e6, e5, e4, e3, e2, e1, e0: float32): Mm[w32, float32] = 
  ## Set packed single-precision floating-point elements with specified values (e7 is highest)
  Mm[w32, float32](mm256_set_ps(e7, e6, e5, e4, e3, e2, e1, e0))

template set*(e3, e2, e1, e0: float64): Mm[w64, float64] = 
  ## Set packed double-precision floating-point elements with specified values (e3 is highest)
  Mm[w64, float64](mm256_set_pd(e3, e2, e1, e0))

template set*(e31, e30, e29, e28, e27, e26, e25, e24, e23, e22, e21, e20, e19, e18, e17, e16, e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0: int8): Mm[w8, int8] = 
  ## Set packed 8-bit integer elements with specified values (e31 is highest)
  Mm[w8, int8](mm256_set_epi8(e31, e30, e29, e28, e27, e26, e25, e24, e23, e22, e21, e20, e19, e18, e17, e16, e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0))

template set*(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0: int16): Mm[w16, int16] = 
  ## Set packed 16-bit integer elements with specified values (e15 is highest)
  Mm[w16, int16](mm256_set_epi16(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0))

template set*(e7, e6, e5, e4, e3, e2, e1, e0: int32): Mm[w32, int32] = 
  ## Set packed 32-bit integer elements with specified values (e7 is highest)
  Mm[w32, int32](mm256_set_epi32(e7, e6, e5, e4, e3, e2, e1, e0))

template set*(e3, e2, e1, e0: int64): Mm[w64, int64] = 
  ## Set packed 64-bit integer elements with specified values (e3 is highest)
  Mm[w64, int64](mm256_set_epi64x(e3, e2, e1, e0))

# Unsigned integer versions (use same underlying intrinsics with casting)
template set*(e31, e30, e29, e28, e27, e26, e25, e24, e23, e22, e21, e20, e19, e18, e17, e16, e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0: uint8): Mm[w8, uint8] = 
  ## Set packed 8-bit unsigned integer elements with specified values (e31 is highest)
  Mm[w8, uint8](mm256_set_epi8(cast[int8](e31), cast[int8](e30), cast[int8](e29), cast[int8](e28), cast[int8](e27), cast[int8](e26), cast[int8](e25), cast[int8](e24), cast[int8](e23), cast[int8](e22), cast[int8](e21), cast[int8](e20), cast[int8](e19), cast[int8](e18), cast[int8](e17), cast[int8](e16), cast[int8](e15), cast[int8](e14), cast[int8](e13), cast[int8](e12), cast[int8](e11), cast[int8](e10), cast[int8](e9), cast[int8](e8), cast[int8](e7), cast[int8](e6), cast[int8](e5), cast[int8](e4), cast[int8](e3), cast[int8](e2), cast[int8](e1), cast[int8](e0)))

template set*(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0: uint16): Mm[w16, uint16] = 
  ## Set packed 16-bit unsigned integer elements with specified values (e15 is highest)
  Mm[w16, uint16](mm256_set_epi16(cast[int16](e15), cast[int16](e14), cast[int16](e13), cast[int16](e12), cast[int16](e11), cast[int16](e10), cast[int16](e9), cast[int16](e8), cast[int16](e7), cast[int16](e6), cast[int16](e5), cast[int16](e4), cast[int16](e3), cast[int16](e2), cast[int16](e1), cast[int16](e0)))

template set*(e7, e6, e5, e4, e3, e2, e1, e0: uint32): Mm[w32, uint32] = 
  ## Set packed 32-bit unsigned integer elements with specified values (e7 is highest)
  Mm[w32, uint32](mm256_set_epi32(cast[int32](e7), cast[int32](e6), cast[int32](e5), cast[int32](e4), cast[int32](e3), cast[int32](e2), cast[int32](e1), cast[int32](e0)))

template set*(e3, e2, e1, e0: uint64): Mm[w64, uint64] = 
  ## Set packed 64-bit unsigned integer elements with specified values (e3 is highest)
  Mm[w64, uint64](mm256_set_epi64x(cast[int64](e3), cast[int64](e2), cast[int64](e1), cast[int64](e0)))

template zero*[N, T](t: typedesc[Mm[N, T]]): Mm[N, T] = 
  ## Return vector with all elements set to zero
  when T is float32:
    Mm[N, T](mm256_setzero_ps())
  elif T is float64:
    Mm[N, T](mm256_setzero_pd())
  else:
    Mm[N, T](mm256_setzero_si256())

# --- Arithmetic Operators ---
template `+`*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Add packed elements in a and b
  when T is float32:
    Mm[N, T](mm256_add_ps(M256(a), M256(b)))
  elif T is float64:
    Mm[N, T](mm256_add_pd(M256d(a), M256d(b)))
  elif T is int64 | uint64:
    Mm[N, T](mm256_add_epi64(M256i(a), M256i(b)))
  elif T is int32 | uint32:
    Mm[N, T](mm256_add_epi32(M256i(a), M256i(b)))
  elif T is int16 | uint16:
    Mm[N, T](mm256_add_epi16(M256i(a), M256i(b)))
  elif T is int8 | uint8:
    Mm[N, T](mm256_add_epi8(M256i(a), M256i(b)))
  else:
    {.error: "Addition not supported for this integer type"}

template `-`*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Subtract packed elements in b from a
  when T is float32:
    Mm[N, T](mm256_sub_ps(M256(a), M256(b)))
  elif T is float64:
    Mm[N, T](mm256_sub_pd(M256d(a), M256d(b)))
  elif T is int64 | uint64:
    Mm[N, T](mm256_sub_epi64(M256i(a), M256i(b)))
  elif T is int32 | uint32:
    Mm[N, T](mm256_sub_epi32(M256i(a), M256i(b)))
  elif T is int16 | uint16:
    Mm[N, T](mm256_sub_epi16(M256i(a), M256i(b)))
  elif T is int8 | uint8:
    Mm[N, T](mm256_sub_epi8(M256i(a), M256i(b)))
  else:
    {.error: "Subtraction not supported for this integer type"}

template `*`*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Multiply packed elements in a and b
  when T is float32:
    Mm[N, T](mm256_mul_ps(M256(a), M256(b)))
  elif T is float64:
    Mm[N, T](mm256_mul_pd(M256d(a), M256d(b)))
  elif T is int32 | uint32:
    Mm[N, T](mm256_mullo_epi32(M256i(a), M256i(b)))
  elif T is int16 | uint16:
    Mm[N, T](mm256_mullo_epi16(M256i(a), M256i(b)))
  else:
    {.error: "Multiplication not supported for this integer type"}

template `/`*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Divide packed elements in a by b
  when T is float32:
    Mm[N, T](mm256_div_ps(M256(a), M256(b)))
  elif T is float64:
    Mm[N, T](mm256_div_pd(M256d(a), M256d(b)))
  else:
    {.error: "Integer division not supported for SIMD".}

template fma*[N, T](a, b, c: Mm[N, T]): Mm[N, T] = 
  ## Fused multiply-add (a * b + c)
  when T is float32:
    Mm[N, T](mm256_fmadd_ps(M256(a), M256(b), M256(c)))
  elif T is float64:
    Mm[N, T](mm256_fmadd_pd(M256d(a), M256d(b), M256d(c)))
  else:
    {.error: "FMA not supported for integers".}

# --- Mathematical Functions ---
template abs*[N, T](a: Mm[N, T]): Mm[N, T] = 
  ## Compute the absolute value of packed elements
  when T is int32:
    Mm[N, T](mm256_abs_epi32(M256i(a)))
  elif T is int16:
    Mm[N, T](mm256_abs_epi16(M256i(a)))
  elif T is int8:
    Mm[N, T](mm256_abs_epi8(M256i(a)))
  else:
    {.error: "abs() only supported for signed integers"}

template sqrt*[N, T](a: Mm[N, T]): Mm[N, T] = 
  ## Compute the square root of packed elements
  when T is float32:
    Mm[N, T](mm256_sqrt_ps(M256(a)))
  elif T is float64:
    Mm[N, T](mm256_sqrt_pd(M256d(a)))
  else:
    {.error: "sqrt not supported for integers".}

template fastSqrt*[N, T](a: Mm[N, T]): Mm[N, T] = 
  ## Compute the approximate reciprocal square root
  when T is float32:
    Mm[N, T](mm256_rsqrt_ps(M256(a)))
  else:
    {.error: "rsqrt only supported for float32".}

# --- Rounding Functions ---
const
  ROUND_NEAREST* = 0x00
  ROUND_DOWN* = 0x01
  ROUND_UP* = 0x02
  ROUND_TRUNC* = 0x03

template round*[N, T](a: Mm[N, T]): Mm[N, T] = 
  ## Round packed elements to nearest integer
  when T is float32:
    Mm[N, T](mm256_round_ps(M256(a), ROUND_NEAREST))
  elif T is float64:
    Mm[N, T](mm256_round_pd(M256d(a), ROUND_NEAREST))
  else:
    {.error: "Rounding not applicable to integers".}

template floor*[N, T](a: Mm[N, T]): Mm[N, T] = 
  ## Round packed elements down to integer
  when T is float32:
    Mm[N, T](mm256_floor_ps(M256(a)))
  elif T is float64:
    Mm[N, T](mm256_floor_pd(M256d(a)))
  else:
    {.error: "Floor not applicable to integers".}

template ceil*[N, T](a: Mm[N, T]): Mm[N, T] = 
  ## Round packed elements up to integer
  when T is float32:
    Mm[N, T](mm256_ceil_ps(M256(a)))
  elif T is float64:
    Mm[N, T](mm256_ceil_pd(M256d(a)))
  else:
    {.error: "Ceil not applicable to integers".}

# --- Comparison Operators ---
template `==`*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Compare packed elements for equality
  when T is float32:
    Mm[N, T](mm256_cmp_ps(M256(a), M256(b), CMP_EQ_OQ))
  elif T is float64:
    Mm[N, T](mm256_cmp_pd(M256d(a), M256d(b), CMP_EQ_OQ))
  elif T is int64 | uint64:
    Mm[N, T](mm256_cmpeq_epi64(M256i(a), M256i(b)))
  elif T is int32 | uint32:
    Mm[N, T](mm256_cmpeq_epi32(M256i(a), M256i(b)))
  elif T is int16 | uint16:
    Mm[N, T](mm256_cmpeq_epi16(M256i(a), M256i(b)))
  elif T is int8 | uint8:
    Mm[N, T](mm256_cmpeq_epi8(M256i(a), M256i(b)))
  else:
    {.error: "Equality comparison not supported for this integer type"}

template `<`*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Compare packed elements for less-than
  when T is float32:
    Mm[N, T](mm256_cmp_ps(M256(a), M256(b), CMP_LT_OS))
  elif T is float64:
    Mm[N, T](mm256_cmp_pd(M256d(a), M256d(b), CMP_LT_OS))
  elif T is int64:
    Mm[N, T](mm256_cmpgt_epi64(M256i(b), M256i(a)))
  elif T is int32:
    Mm[N, T](mm256_cmpgt_epi32(M256i(b), M256i(a)))
  elif T is int16:
    Mm[N, T](mm256_cmpgt_epi16(M256i(b), M256i(a)))
  elif T is int8:
    Mm[N, T](mm256_cmpgt_epi8(M256i(b), M256i(a)))
  else:
    {.error: "Less-than comparison not supported for this integer type (unsigned integers need special handling)"}

template `>`*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Compare packed elements for greater-than
  when T is float32:
    Mm[N, T](mm256_cmp_ps(M256(b), M256(a), CMP_LT_OS))
  elif T is float64:
    Mm[N, T](mm256_cmp_pd(M256d(b), M256d(a), CMP_LT_OS))
  elif T is int64:
    Mm[N, T](mm256_cmpgt_epi64(M256i(a), M256i(b)))
  elif T is int32:
    Mm[N, T](mm256_cmpgt_epi32(M256i(a), M256i(b)))
  elif T is int16:
    Mm[N, T](mm256_cmpgt_epi16(M256i(a), M256i(b)))
  elif T is int8:
    Mm[N, T](mm256_cmpgt_epi8(M256i(a), M256i(b)))
  else:
    {.error: "Greater-than comparison not supported for this integer type (unsigned integers need special handling)"}

template `<=`*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Compare packed elements for less-than-or-equal
  when T is float32:
    Mm[N, T](mm256_cmp_ps(M256(a), M256(b), CMP_LE_OS))
  elif T is float64:
    Mm[N, T](mm256_cmp_pd(M256d(a), M256d(b), CMP_LE_OS))
  else: # integers - use not greater-than
    not (a > b)

template `>=`*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Compare packed elements for greater-than-or-equal
  when T is float32:
    Mm[N, T](mm256_cmp_ps(M256(a), M256(b), CMP_NLT_US))
  elif T is float64:
    Mm[N, T](mm256_cmp_pd(M256d(a), M256d(b), CMP_NLT_US))
  else: # integers - use not less-than
    not (a < b)

# --- Logical and Bitwise Operators ---
template `and`*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Compute the bitwise AND
  Mm[N, T](mm256_and_si256(M256i(a), M256i(b)))

template `or`*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Compute the bitwise OR
  Mm[N, T](mm256_or_si256(M256i(a), M256i(b)))

template `xor`*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Compute the bitwise XOR
  Mm[N, T](mm256_xor_si256(M256i(a), M256i(b)))

template `not`*[N, T](a: Mm[N, T]): Mm[N, T] = 
  ## Compute the bitwise NOT
  Mm[N, T](mm256_xor_si256(M256i(a), mm256_cmpeq_epi32(M256i(a), M256i(a))))

# --- Min/Max Operations ---
template min*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Return packed minimum values
  when T is float32:
    Mm[N, T](mm256_min_ps(M256(a), M256(b)))
  elif T is float64:
    Mm[N, T](mm256_min_pd(M256d(a), M256d(b)))
  elif T is int32:
    Mm[N, T](mm256_min_epi32(M256i(a), M256i(b)))
  elif T is int16:
    Mm[N, T](mm256_min_epi16(M256i(a), M256i(b)))
  elif T is int8:
    Mm[N, T](mm256_min_epi8(M256i(a), M256i(b)))
  elif T is uint32:
    Mm[N, T](mm256_min_epu32(M256i(a), M256i(b)))
  elif T is uint16:
    Mm[N, T](mm256_min_epu16(M256i(a), M256i(b)))
  elif T is uint8:
    Mm[N, T](mm256_min_epu8(M256i(a), M256i(b)))
  else:
    {.error: "min() not implemented for this type"}

template max*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Return packed maximum values
  when T is float32:
    Mm[N, T](mm256_max_ps(M256(a), M256(b)))
  elif T is float64:
    Mm[N, T](mm256_max_pd(M256d(a), M256d(b)))
  elif T is int32:
    Mm[N, T](mm256_max_epi32(M256i(a), M256i(b)))
  elif T is int16:
    Mm[N, T](mm256_max_epi16(M256i(a), M256i(b)))
  elif T is int8:
    Mm[N, T](mm256_max_epi8(M256i(a), M256i(b)))
  elif T is uint32:
    Mm[N, T](mm256_max_epu32(M256i(a), M256i(b)))
  elif T is uint16:
    Mm[N, T](mm256_max_epu16(M256i(a), M256i(b)))
  elif T is uint8:
    Mm[N, T](mm256_max_epu8(M256i(a), M256i(b)))
  else:
    {.error: "max() not implemented for this type"}

# --- Horizontal Operations ---
template sum*[N, T](a: Mm[N, T]): T =
  ## Sum all elements, returning scalar result
  when T is float32:
    # Swaps the high and low 128-bit lanes and adds them.
    let v1 = mm256_permute2f128_ps(M256(a), M256(a), 1)
    let v2 = mm256_add_ps(M256(a), v1)
    # Horizontal adds within the 128-bit lane.
    let v3 = mm256_hadd_ps(v2, v2)
    let v4 = mm256_hadd_ps(v3, v3)
    # Extract the final scalar result.
    mm_cvtss_f32(mm256_castps256_ps128(v4))
  elif T is float64:
    let v1 = mm256_permute2f128_pd(M256d(a), M256d(a), 1)
    let v2 = mm256_add_pd(M256d(a), v1)
    let v3 = mm256_hadd_pd(v2, v2)
    mm256_cvtsd_f64(v3)
  else:
    {.error: "hsum() not implemented for this integer type".}

# --- Nim-style aliases ---
template toMm*[T](p: array[32, T]): Mm[w8, T] = load(p)
template toMm*[T](p: array[16, T]): Mm[w16, T] = load(p)
template toMm*[T](p: array[8, T]): Mm[w32, T] = load(p)
template toMm*[T](p: array[4, T]): Mm[w64, T] = load(p)
template toArray*[T](m: Mm[w8, T]): array[32, T] = store(m)
template toArray*[T](m: Mm[w16, T]): array[16, T] = store(m)
template toArray*[T](m: Mm[w32, T]): array[8, T] = store(m)
template toArray*[T](m: Mm[w64, T]): array[4, T] = store(m)

template initMm*(e7, e6, e5, e4, e3, e2, e1, e0: float32): Mm[w32, float32] = set(e7, e6, e5, e4, e3, e2, e1, e0)
template initMm*(e3, e2, e1, e0: float64): Mm[w64, float64] = set(e3, e2, e1, e0)
template initMm*(e31, e30, e29, e28, e27, e26, e25, e24, e23, e22, e21, e20, e19, e18, e17, e16, e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0: int8): Mm[w8, int8] = set(e31, e30, e29, e28, e27, e26, e25, e24, e23, e22, e21, e20, e19, e18, e17, e16, e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0)
template initMm*(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0: int16): Mm[w16, int16] = set(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0)
template initMm*(e7, e6, e5, e4, e3, e2, e1, e0: int32): Mm[w32, int32] = set(e7, e6, e5, e4, e3, e2, e1, e0)
template initMm*(e3, e2, e1, e0: int64): Mm[w64, int64] = set(e3, e2, e1, e0)
template initMm*(e31, e30, e29, e28, e27, e26, e25, e24, e23, e22, e21, e20, e19, e18, e17, e16, e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0: uint8): Mm[w8, uint8] = set(e31, e30, e29, e28, e27, e26, e25, e24, e23, e22, e21, e20, e19, e18, e17, e16, e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0)
template initMm*(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0: uint16): Mm[w16, uint16] = set(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0)
template initMm*(e7, e6, e5, e4, e3, e2, e1, e0: uint32): Mm[w32, uint32] = set(e7, e6, e5, e4, e3, e2, e1, e0)
template initMm*(e3, e2, e1, e0: uint64): Mm[w64, uint64] = set(e3, e2, e1, e0)
template initMm*[N, T](t: typedesc[Mm[N, T]]): Mm[N, T] = zero(t)

template mask*[N, T](a: Mm[N, T]): int32 = 
  ## Create mask from the most significant bit of each element
  when T is float32:
    mm256_movemask_ps(M256(a))
  elif T is float64:
    mm256_movemask_pd(M256d(a))
  else: # integers (treats as 8-bit elements)
    mm256_movemask_epi8(M256i(a))

# --- Advanced SIMD Operations ---

# --- Saturated Arithmetic ---
template saturatedAdd*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Add packed elements using saturation
  when T is int16 | uint16:
    Mm[N, T](mm256_adds_epi16(M256i(a), M256i(b)))
  elif T is uint8:
    Mm[N, T](mm256_adds_epu8(M256i(a), M256i(b)))
  elif T is int8:
    Mm[N, T](mm256_adds_epi8(M256i(a), M256i(b)))
  else:
    {.error: "Saturated addition not supported for this type"}

template saturatedSub*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Subtract packed elements using saturation
  when T is int16 | uint16:
    Mm[N, T](mm256_subs_epi16(M256i(a), M256i(b)))
  elif T is uint8:
    Mm[N, T](mm256_subs_epu8(M256i(a), M256i(b)))
  elif T is int8:
    Mm[N, T](mm256_subs_epi8(M256i(a), M256i(b)))
  else:
    {.error: "Saturated subtraction not supported for this type"}

# --- Blend Operations ---
template blend*[N, T](a, b: Mm[N, T], imm8: int32): Mm[N, T] = 
  ## Blend packed elements from a and b using control mask imm8
  when T is float32:
    Mm[N, T](mm256_blend_ps(M256(a), M256(b), imm8))
  elif T is float64:
    Mm[N, T](mm256_blend_pd(M256d(a), M256d(b), imm8))
  elif T is int16 | uint16:
    Mm[N, T](mm256_blend_epi16(M256i(a), M256i(b), imm8))
  elif T is int32 | uint32:
    Mm[N, T](mm256_blend_epi32(M256i(a), M256i(b), imm8))
  else:
    {.error: "Blend not supported for this type"}

template blendv*[N, T](a, b, mask: Mm[N, T]): Mm[N, T] = 
  ## Blend packed elements from a and b using mask
  when T is float32:
    Mm[N, T](mm256_blendv_ps(M256(a), M256(b), M256(mask)))
  elif T is float64:
    Mm[N, T](mm256_blendv_pd(M256d(a), M256d(b), M256d(mask)))
  elif T is int8 | uint8:
    Mm[N, T](mm256_blendv_epi8(M256i(a), M256i(b), M256i(mask)))
  else:
    {.error: "Variable blend not supported for this type"}

# --- Gather Operations ---
template gather*[T](p: openArray[T], vindex: Mm[w32, int32], scale: int32): Mm[w32, T] = 
  ## Gather elements from memory using 32-bit indices
  when T is int32:
    Mm[w32, T](mm256_i32gather_epi32(cast[pointer](unsafeAddr p[0]), M256i(vindex), scale))
  elif T is uint32:
    Mm[w32, T](mm256_i32gather_epi32(cast[pointer](unsafeAddr p[0]), M256i(vindex), scale))
  elif T is float32:
    Mm[w32, T](mm256_i32gather_ps(cast[pointer](unsafeAddr p[0]), M256i(vindex), scale))
  else:
    {.error: "Gather not supported for this type"}

template gather*[T](p: openArray[T], vindex: Mm[w64, int64], scale: int32): Mm[w64, T] = 
  ## Gather elements from memory using 64-bit indices
  when T is int64:
    Mm[w64, T](mm256_i64gather_epi64(cast[pointer](unsafeAddr p[0]), M256i(vindex), scale))
  elif T is uint64:
    Mm[w64, T](mm256_i64gather_epi64(cast[pointer](unsafeAddr p[0]), M256i(vindex), scale))
  elif T is float64:
    Mm[w64, T](mm256_i64gather_pd(cast[pointer](unsafeAddr p[0]), M256i(vindex), scale))
  else:
    {.error: "Gather not supported for this type"}

# --- Pack/Unpack Operations ---
template pack*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Convert packed elements to smaller type using signed saturation
  when T is int16:
    Mm[N, int8](mm256_packs_epi16(M256i(a), M256i(b)))
  elif T is int32:
    Mm[N, int16](mm256_packs_epi32(M256i(a), M256i(b)))
  else:
    {.error: "Pack not supported for this type"}

template packus*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Convert packed elements to smaller unsigned type using unsigned saturation
  when T is int16:
    Mm[N, uint8](mm256_packus_epi16(M256i(a), M256i(b)))
  elif T is int32:
    Mm[N, uint16](mm256_packus_epi32(M256i(a), M256i(b)))
  else:
    {.error: "Pack unsigned not supported for this type"}

template unpackLo*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Unpack and interleave elements from the low half
  when T is float32:
    Mm[N, T](mm256_unpacklo_ps(M256(a), M256(b)))
  elif T is float64:
    Mm[N, T](mm256_unpacklo_pd(M256d(a), M256d(b)))
  elif T is int32 | uint32:
    Mm[N, T](mm256_unpacklo_epi32(M256i(a), M256i(b)))
  elif T is int16 | uint16:
    Mm[N, T](mm256_unpacklo_epi16(M256i(a), M256i(b)))
  elif T is int8 | uint8:
    Mm[N, T](mm256_unpacklo_epi8(M256i(a), M256i(b)))
  elif T is int64 | uint64:
    Mm[N, T](mm256_unpacklo_epi64(M256i(a), M256i(b)))
  else:
    {.error: "Unpack low not supported for this type"}

template unpackHi*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Unpack and interleave elements from the high half
  when T is float32:
    Mm[N, T](mm256_unpackhi_ps(M256(a), M256(b)))
  elif T is float64:
    Mm[N, T](mm256_unpackhi_pd(M256d(a), M256d(b)))
  elif T is int32 | uint32:
    Mm[N, T](mm256_unpackhi_epi32(M256i(a), M256i(b)))
  elif T is int16 | uint16:
    Mm[N, T](mm256_unpackhi_epi16(M256i(a), M256i(b)))
  elif T is int8 | uint8:
    Mm[N, T](mm256_unpackhi_epi8(M256i(a), M256i(b)))
  elif T is int64 | uint64:
    Mm[N, T](mm256_unpackhi_epi64(M256i(a), M256i(b)))
  else:
    {.error: "Unpack high not supported for this type"}

# --- Horizontal Operations ---
template hadd*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Horizontally add adjacent pairs of elements, packing results
  when T is int32 | uint32:
    Mm[N, T](mm256_hadd_epi32(M256i(a), M256i(b)))
  elif T is int16 | uint16:
    Mm[N, T](mm256_hadd_epi16(M256i(a), M256i(b)))
  elif T is float32:
    Mm[N, T](mm256_hadd_ps(M256(a), M256(b)))
  elif T is float64:
    Mm[N, T](mm256_hadd_pd(M256d(a), M256d(b)))
  else:
    {.error: "Horizontal add not supported for this type"}

template hsub*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Horizontally subtract adjacent pairs of elements, packing results
  when T is int32 | uint32:
    Mm[N, T](mm256_hsub_epi32(M256i(a), M256i(b)))
  elif T is int16 | uint16:
    Mm[N, T](mm256_hsub_epi16(M256i(a), M256i(b)))
  elif T is float32:
    Mm[N, T](mm256_hsub_ps(M256(a), M256(b)))
  elif T is float64:
    Mm[N, T](mm256_hsub_pd(M256d(a), M256d(b)))
  else:
    {.error: "Horizontal subtract not supported for this type"}

# --- Average Operations ---
template avg*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Average packed unsigned elements
  when T is uint16:
    Mm[N, T](mm256_avg_epu16(M256i(a), M256i(b)))
  elif T is uint8:
    Mm[N, T](mm256_avg_epu8(M256i(a), M256i(b)))
  else:
    {.error: "Average only supported for unsigned 8-bit and 16-bit integers"}

# --- Shuffle Operations ---
template shuffle*[N, T](a: Mm[N, T], imm8: int32): Mm[N, T] = 
  ## Shuffle elements using control in imm8
  when T is float32:
    Mm[N, T](mm256_shuffle_ps(M256(a), M256(a), imm8))
  elif T is float64:
    Mm[N, T](mm256_shuffle_pd(M256d(a), M256d(a), imm8))
  elif T is int32 | uint32:
    Mm[N, T](mm256_shuffle_epi32(M256i(a), imm8))
  else:
    {.error: "Shuffle not supported for this type"}

# --- Shift Operations ---
template `shl`*[N, T](a: Mm[N, T], imm: int32): Mm[N, T] = 
  ## Shift packed elements left by imm bits
  when T is int32 | uint32:
    Mm[N, T](mm256_slli_epi32(M256i(a), imm))
  elif T is int64 | uint64:
    Mm[N, T](mm256_slli_epi64(M256i(a), imm))
  elif T is int16 | uint16:
    Mm[N, T](mm256_slli_epi16(M256i(a), imm))
  else:
    {.error: "Left shift not supported for this type"}

template `shr`*[N, T](a: Mm[N, T], imm: int32): Mm[N, T] = 
  ## Shift packed elements right by imm bits
  when T is int32 | uint32:
    Mm[N, T](mm256_srli_epi32(M256i(a), imm))
  elif T is int64 | uint64:
    Mm[N, T](mm256_srli_epi64(M256i(a), imm))
  elif T is int16 | uint16:
    Mm[N, T](mm256_srli_epi16(M256i(a), imm))
  else:
    {.error: "Right shift not supported for this type"}

# --- Variable Shift Operations ---
template vshl*[N, T](a, count: Mm[N, T]): Mm[N, T] = 
  ## Shift packed elements left by amounts specified in count
  when T is int32 | uint32:
    Mm[N, T](mm256_sllv_epi32(M256i(a), M256i(count)))
  elif T is int64 | uint64:
    Mm[N, T](mm256_sllv_epi64(M256i(a), M256i(count)))
  else:
    {.error: "Variable left shift not supported for this type"}

template vshr*[N, T](a, count: Mm[N, T]): Mm[N, T] = 
  ## Shift packed elements right by amounts specified in count
  when T is int32 | uint32:
    Mm[N, T](mm256_srlv_epi32(M256i(a), M256i(count)))
  elif T is int64 | uint64:
    Mm[N, T](mm256_srlv_epi64(M256i(a), M256i(count)))
  else:
    {.error: "Variable right shift not supported for this type"}

# --- Multiply Variants ---
template mulHi*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Multiply packed elements, returning high bits of intermediate results
  when T is int16:
    Mm[N, T](mm256_mulhi_epi16(M256i(a), M256i(b)))
  elif T is uint16:
    Mm[N, T](mm256_mulhi_epu16(M256i(a), M256i(b)))
  else:
    {.error: "High multiply only supported for 16-bit integers"}

template mulLo*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Multiply packed elements, returning low bits of intermediate results
  when T is int16 | uint16:
    Mm[N, T](mm256_mullo_epi16(M256i(a), M256i(b)))
  else:
    {.error: "Low multiply only supported for 16-bit integers"}

# --- Permute Operations ---
template permute*[N, T](a: Mm[N, T], imm8: int32): Mm[N, T] = 
  ## Shuffle elements using the control in imm8
  when T is float64:
    Mm[N, T](mm256_permute4x64_pd(M256d(a), imm8))
  elif T is int64 | uint64:
    Mm[N, T](mm256_permute4x64_epi64(M256i(a), imm8))
  else:
    {.error: "Permute not supported for this type"}

template permuteVar*[N, T](a: Mm[N, T], idx: Mm[N, int32]): Mm[N, T] = 
  ## Shuffle elements using control indices in idx
  when T is float32:
    Mm[N, T](mm256_permutevar8x32_ps(M256(a), M256i(idx)))
  elif T is int32 | uint32:
    Mm[N, T](mm256_permutevar8x32_epi32(M256i(a), M256i(idx)))
  else:
    {.error: "Variable permute not supported for this type"}

# --- Masked Load/Store Operations ---
template maskLoad*[N, T](p: openArray[T], mask: Mm[N, T]): Mm[N, T] = 
  ## Conditionally load elements from memory using mask
  when T is float32:
    Mm[N, T](mm256_maskload_ps(cast[pointer](unsafeAddr p[0]), M256i(mask)))
  elif T is float64:
    Mm[N, T](mm256_maskload_pd(cast[pointer](unsafeAddr p[0]), M256i(mask)))
  else:
    {.error: "Mask load only supported for floating-point types"}

template maskStore*[N, T](p: var openArray[T], mask: Mm[N, T], a: Mm[N, T]) = 
  ## Conditionally store elements to memory using mask
  when T is float32:
    mm256_maskstore_ps(cast[pointer](addr p[0]), M256i(mask), M256(a))
  elif T is float64:
    mm256_maskstore_pd(cast[pointer](addr p[0]), M256i(mask), M256d(a))
  else:
    {.error: "Mask store only supported for floating-point types"}

# --- Stream Operations ---
template streamLoad*[T](p: array[8, T]): Mm[w32, T] = 
  ## Load data from memory using non-temporal memory hint
  when T is int32 | uint32:
    Mm[w32, T](mm256_stream_load_si256(cast[pointer](unsafeAddr p[0])))
  else:
    {.error: "Stream load only supported for 32-bit integers"}

# --- Additional Multiply Operations ---
template madd*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Multiply packed 16-bit integers, horizontally add adjacent 32-bit results
  when T is int16 | uint16:
    Mm[N, T](mm256_madd_epi16(M256i(a), M256i(b)))
  else:
    {.error: "Multiply-add only supported for 16-bit integers"}

template mulHrs*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Multiply packed signed 16-bit integers, truncate to 18 MSBs, round by adding 1
  when T is int16:
    Mm[N, T](mm256_mulhrs_epi16(M256i(a), M256i(b)))
  else:
    {.error: "High rounded multiply only supported for signed 16-bit integers"}

template mul32*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Multiply low 32-bit integers from each 64-bit element, returning 64-bit results
  when T is int32:
    Mm[N, T](mm256_mul_epi32(M256i(a), M256i(b)))
  elif T is uint32:
    Mm[N, T](mm256_mul_epu32(M256i(a), M256i(b)))
  else:
    {.error: "32-bit multiply only supported for 32-bit integers"}

# --- Sign Operations ---
template sign*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Negate elements in a when corresponding element in b is negative
  when T is int32:
    Mm[N, T](mm256_sign_epi32(M256i(a), M256i(b)))
  elif T is int16:
    Mm[N, T](mm256_sign_epi16(M256i(a), M256i(b)))
  elif T is int8:
    Mm[N, T](mm256_sign_epi8(M256i(a), M256i(b)))
  else:
    {.error: "Sign only supported for signed integers"}

# --- Sum of Absolute Differences ---
template sad*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Compute absolute differences, horizontally sum each consecutive 8 differences
  when T is uint8:
    Mm[N, T](mm256_sad_epu8(M256i(a), M256i(b)))
  else:
    {.error: "SAD only supported for unsigned 8-bit integers"}

# --- Additional Bitwise Operations ---
template andnot*[N, T](a, b: Mm[N, T]): Mm[N, T] = 
  ## Compute bitwise NOT of a and then AND with b
  Mm[N, T](mm256_andnot_si256(M256i(a), M256i(b)))

# --- Alignment Operations ---
template alignr*[N, T](a, b: Mm[N, T], imm8: int32): Mm[N, T] = 
  ## Concatenate 16-byte blocks, shift right by imm8 bytes, return low 16 bytes
  when T is int8 | uint8:
    Mm[N, T](mm256_alignr_epi8(M256i(a), M256i(b), imm8))
  else:
    {.error: "Align right only supported for 8-bit integers"}

# --- Extract/Insert Operations ---
template extract*[N, T](a: Mm[N, T], index: int32): T = 
  ## Extract element from vector at specified index
  when T is int16 | uint16:
    cast[T](mm256_extract_epi16(M256i(a), index))
  elif T is int8 | uint8:
    cast[T](mm256_extract_epi8(M256i(a), index))
  else:
    {.error: "Extract only supported for 8-bit and 16-bit integers"}

# --- Type Conversion Operations ---
template bitcast*[N, T, U](a: Mm[N, T], targetType: typedesc[Mm[N, U]]): Mm[N, U] = 
  ## Cast vector to different type (zero latency, compilation-only operation)
  when T is float32 and U is int32:
    Mm[N, U](mm256_castps_si256(M256(a)))
  elif T is int32 and U is float32:
    Mm[N, U](mm256_castsi256_ps(M256i(a)))
  elif T is float64 and U is float32:
    Mm[N, U](mm256_castpd_ps(M256d(a)))
  elif T is float32 and U is float64:
    Mm[N, U](mm256_castps_pd(M256(a)))
  else:
    {.error: "Bitcast not supported for this type combination"}

template convert*[N, T](a: Mm[N, T]): auto = 
  ## Convert between integer and floating-point types
  when T is int32:
    Mm[N, float32](mm256_cvtepi32_ps(M256i(a)))
  elif T is float32:
    Mm[N, int32](mm256_cvttps_epi32(M256(a)))
  else:
    {.error: "Convert only supported between int32 and float32"}
