{.passC: "-mavx2 -mfma".}
import nimsimd/avx2
import nimsimd/fma
export avx2

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

# --- Vector Initialization ---
template splat*(val: float32): M256 = mm256_set1_ps(val)
template splat*(val: float64): M256d = mm256_set1_pd(val)
template splat*(val: int32): M256i = mm256_set1_epi32(val)
template splat*(val: int64): M256i = mm256_set1_epi64x(val)

template set*(e7, e6, e5, e4, e3, e2, e1, e0: float32): M256 = mm256_set_ps(e7, e6, e5, e4, e3, e2, e1, e0)
template set*(e3, e2, e1, e0: float64): M256d = mm256_set_pd(e3, e2, e1, e0)
template set*(e7, e6, e5, e4, e3, e2, e1, e0: int32): M256i = mm256_set_epi32(e7, e6, e5, e4, e3, e2, e1, e0)

template setzero*(`type`: typedesc[M256]): M256 = mm256_setzero_ps()
template setzero*(`type`: typedesc[M256d]): M256d = mm256_setzero_pd()
template setzero*(`type`: typedesc[M256i]): M256i = mm256_setzero_si256()

# --- Memory Operations ---
template load*(p: pointer, T: typedesc[M256]): M256 = mm256_load_ps(p)
template load*(p: pointer, T: typedesc[M256d]): M256d = mm256_load_pd(p)
template load*(p: pointer, T: typedesc[M256i]): M256i = mm256_load_si256(p)
template loadu*(p: pointer, T: typedesc[M256]): M256 = mm256_loadu_ps(p)
template loadu*(p: pointer, T: typedesc[M256d]): M256d = mm256_loadu_pd(p)
template loadu*(p: pointer, T: typedesc[M256i]): M256i = mm256_loadu_si256(p)

template store*(p: pointer, a: M256) = mm256_store_ps(p, a)
template store*(p: pointer, a: M256d) = mm256_store_pd(p, a)
template store*(p: pointer, a: M256i) = mm256_store_si256(p, a)
template storeu*(p: pointer, a: M256) = mm256_storeu_ps(p, a)
template storeu*(p: pointer, a: M256d) = mm256_storeu_pd(p, a)
template storeu*(p: pointer, a: M256i) = mm256_storeu_si256(p, a)

template maskload*(p: pointer, mask: M256i, T: typedesc[M256]): M256 = mm256_maskload_ps(p, mask)
template maskload*(p: pointer, mask: M256i, T: typedesc[M256d]): M256d = mm256_maskload_pd(p, mask)
template maskstore*(p: pointer, mask: M256i, a: M256) = mm256_maskstore_ps(p, mask, a)
template maskstore*(p: pointer, mask: M256i, a: M256d) = mm256_maskstore_pd(p, mask, a)

# --- Arithmetic and Mathematical Functions ---
template `+`*(a, b: M256): M256 = mm256_add_ps(a, b)
template `+`*(a, b: M256d): M256d = mm256_add_pd(a, b)
template `+`*(a, b: M256i): M256i = mm256_add_epi32(a, b)

template `-`*(a, b: M256): M256 = mm256_sub_ps(a, b)
template `-`*(a, b: M256d): M256d = mm256_sub_pd(a, b)
template `-`*(a, b: M256i): M256i = mm256_sub_epi32(a, b)

template `*`*(a, b: M256): M256 = mm256_mul_ps(a, b)
template `*`*(a, b: M256d): M256d = mm256_mul_pd(a, b)
template `*`*(a, b: M256i): M256i = mm256_mullo_epi32(a, b)

template `/`*(a, b: M256): M256 = mm256_div_ps(a, b)
template `/`*(a, b: M256d): M256d = mm256_div_pd(a, b)

template fmadd*(a, b, c: M256): M256 = mm256_fmadd_ps(a, b, c)
template fmadd*(a, b, c: M256d): M256d = mm256_fmadd_pd(a, b, c)

template abs*(a: M256i): M256i = mm256_abs_epi32(a)

template saturatedAdd*(a, b: M256i): M256i = mm256_adds_epi16(a, b)
template saturatedSub*(a, b: M256i): M256i = mm256_subs_epi16(a, b)

template sqrt*(a: M256): M256 = mm256_sqrt_ps(a)
template sqrt*(a: M256d): M256d = mm256_sqrt_pd(a)
template rsqrt*(a: M256): M256 = mm256_rsqrt_ps(a)

# --- Rounding Functions ---
const
  ROUND_NEAREST* = 0x00
  ROUND_DOWN* = 0x01
  ROUND_UP* = 0x02
  ROUND_TRUNC* = 0x03

template round*(a: M256): M256 = mm256_round_ps(a, ROUND_NEAREST)
template round*(a: M256d): M256d = mm256_round_pd(a, ROUND_NEAREST)
template floor*(a: M256): M256 = mm256_floor_ps(a)
template floor*(a: M256d): M256d = mm256_floor_pd(a)
template ceil*(a: M256): M256 = mm256_ceil_ps(a)
template ceil*(a: M256d): M256d = mm256_ceil_pd(a)

# --- Comparison Operators ---
template `==`*(a, b: M256): M256 = mm256_cmp_ps(a, b, CMP_EQ_OQ)
template `==`*(a, b: M256d): M256d = mm256_cmp_pd(a, b, CMP_EQ_OQ)
template `==`*(a, b: M256i): M256i = mm256_cmpeq_epi32(a, b)

template `<`*(a, b: M256): M256 = mm256_cmp_ps(a, b, CMP_LT_OS)
template `<`*(a, b: M256d): M256d = mm256_cmp_pd(a, b, CMP_LT_OS)
template `<`*(a, b: M256i): M256i = mm256_cmpgt_epi32(b, a)

template `>`*(a, b: M256): M256 = mm256_cmp_ps(b, a, CMP_LT_OS)
template `>`*(a, b: M256d): M256d = mm256_cmp_pd(b, a, CMP_LT_OS)
template `>`*(a, b: M256i): M256i = mm256_cmpgt_epi32(a, b)

template `<=`*(a, b: M256): M256 = mm256_cmp_ps(a, b, CMP_LE_OS)
template `<=`*(a, b: M256d): M256d = mm256_cmp_pd(a, b, CMP_LE_OS)
template `<=`*(a, b: M256i): M256i = not `>`(a,b)

template `>=`*(a, b: M256): M256 = mm256_cmp_ps(a, b, CMP_NLT_US)
template `>=`*(a, b: M256d): M256d = mm256_cmp_pd(a, b, CMP_NLT_US)
template `>=`*(a, b: M256i): M256i = not `<`(a,b)

# --- Logical and Bitwise Operators ---
template `and`*(a, b: M256i): M256i = mm256_and_si256(a, b)
template `or`*(a, b: M256i): M256i = mm256_or_si256(a, b)
template `xor`*(a, b: M256i): M256i = mm256_xor_si256(a, b)
template `not`*(a: M256i): M256i = mm256_xor_si256(a, mm256_cmpeq_epi32(a, a))

template `shl`*(a: M256i, imm8: int32): M256i = mm256_slli_epi32(a, imm8)
template `shr`*(a: M256i, imm8: int32): M256i = mm256_srli_epi32(a, imm8)
template `shl`*(a: M256i, imm64: int64): M256i = mm256_slli_epi64(a, cint(imm64))
template `shr`*(a: M256i, imm64: int64): M256i = mm256_srli_epi64(a, cint(imm64))

# --- Data Manipulation ---
template shuffle*(a: M256, imm8: int32): M256 = mm256_shuffle_ps(a, a, imm8)
template shuffle*(a: M256d, imm8: int32): M256d = mm256_shuffle_pd(a, a, imm8)

template blend*(a, b: M256, imm8: int32): M256 = mm256_blend_ps(a, b, imm8)
template blend*(a, b: M256d, imm8: int32): M256d = mm256_blend_pd(a, b, imm8)
template blendv*(a, b, mask: M256): M256 = mm256_blendv_ps(a, b, mask)
template blendv*(a, b, mask: M256d): M256d = mm256_blendv_pd(a, b, mask)

template unpackLo*(a, b: M256): M256 = mm256_unpacklo_ps(a, b)
template unpackLo*(a, b: M256d): M256d = mm256_unpacklo_pd(a, b)
template unpackHi*(a, b: M256): M256 = mm256_unpackhi_ps(a, b)
template unpackHi*(a, b: M256d): M256d = mm256_unpackhi_pd(a, b)

# --- Type Conversions ---
template bitcast*(a: M256i, T: typedesc[M256]): M256 = mm256_castsi256_ps(a)
template bitcast*(a: M256, T: typedesc[M256i]): M256i = mm256_castps_si256(a)
template bitcast*(a: M256d, T: typedesc[M256]): M256 = mm256_castpd_ps(a)
template bitcast*(a: M256, T: typedesc[M256d]): M256d = mm256_castps_pd(a)
template convert*(a: M256i): M256 = mm256_cvtepi32_ps(a)
template convert*(a: M256): M256i = mm256_cvttps_epi32(a)

# --- Horizontal (Reduction) Operations & Masks ---
template hsum*(a: M256): float32 =
  # Swaps the high and low 128-bit lanes and adds them.
  let v1 = mm256_permute2f128_ps(a, a, 1)
  let v2 = mm256_add_ps(a, v1)
  # Horizontal adds within the 128-bit lane.
  let v3 = mm256_hadd_ps(v2, v2)
  let v4 = mm256_hadd_ps(v3, v3)
  # Extract the final scalar result.
  mm_cvtss_f32(mm256_castps256_ps128(v4))

template hsum*(a: M256d): float64 =
  let v1 = mm256_permute2f128_pd(a, a, 1)
  let v2 = mm256_add_pd(a, v1)
  let v3 = mm256_hadd_pd(v2, v2)
  mm256_cvtsd_f64(v3)

template hmin*(a: M256): float32 =
  # Reduce across 128-bit lanes
  let v1 = mm256_permute2f128_ps(a, a, 1)
  let m1 = mm256_min_ps(a, v1)
  # Reduce within the 128-bit lane
  let v2 = mm256_permute_ps(m1, 0b01_00_11_10) # Shuffle [3,2,1,0] -> [2,3,0,1]
  let m2 = mm256_min_ps(m1, v2)
  let v3 = mm256_permute_ps(m2, 0b10_11_00_01) # Shuffle [3,2,1,0] -> [1,0,3,2]
  let m3 = mm256_min_ps(m2, v3)
  # Extract the final scalar result
  mm_cvtss_f32(mm256_castps256_ps128(m3))

# template hmax*(a: M256): float32 = ... (To be implemented)

template movemask*(a: M256): int32 = mm256_movemask_ps(a)
template movemask*(a: M256d): int32 = mm256_movemask_pd(a)
template movemask*(a: M256i): int32 = mm256_movemask_epi8(a)
