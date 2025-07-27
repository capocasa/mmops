## AVX2 Wrapper for Nim
## 
## This module provides comprehensive wrappers for all AVX2 instructions
## along with convenient operator overloads and sequence-like operations.

import std/math

{.push header: "immintrin.h".}

type
  m256i* {.importc: "__m256i", header: "immintrin.h".} = object
  m256* {.importc: "__m256", header: "immintrin.h".} = object
  m256d* {.importc: "__m256d", header: "immintrin.h".} = object

# Core AVX2 integer types
proc mm256_setzero_si256*(): m256i {.importc: "_mm256_setzero_si256".}
proc mm256_set1_epi8*(a: int8): m256i {.importc: "_mm256_set1_epi8".}
proc mm256_set1_epi16*(a: int16): m256i {.importc: "_mm256_set1_epi16".}
proc mm256_set1_epi32*(a: int32): m256i {.importc: "_mm256_set1_epi32".}
proc mm256_set1_epi64x*(a: int64): m256i {.importc: "_mm256_set1_epi64x".}

# Load/Store operations
proc mm256_load_si256*(mem_addr: ptr m256i): m256i {.importc: "_mm256_load_si256".}
proc mm256_loadu_si256*(mem_addr: ptr m256i): m256i {.importc: "_mm256_loadu_si256".}
proc mm256_store_si256*(mem_addr: ptr m256i, a: m256i) {.importc: "_mm256_store_si256".}
proc mm256_storeu_si256*(mem_addr: ptr m256i, a: m256i) {.importc: "_mm256_storeu_si256".}

# Arithmetic operations - 8-bit integers
proc mm256_add_epi8*(a, b: m256i): m256i {.importc: "_mm256_add_epi8".}
proc mm256_sub_epi8*(a, b: m256i): m256i {.importc: "_mm256_sub_epi8".}
proc mm256_adds_epi8*(a, b: m256i): m256i {.importc: "_mm256_adds_epi8".}
proc mm256_adds_epu8*(a, b: m256i): m256i {.importc: "_mm256_adds_epu8".}
proc mm256_subs_epi8*(a, b: m256i): m256i {.importc: "_mm256_subs_epi8".}
proc mm256_subs_epu8*(a, b: m256i): m256i {.importc: "_mm256_subs_epu8".}
proc mm256_abs_epi8*(a: m256i): m256i {.importc: "_mm256_abs_epi8".}
proc mm256_sign_epi8*(a, b: m256i): m256i {.importc: "_mm256_sign_epi8".}

# Arithmetic operations - 16-bit integers
proc mm256_add_epi16*(a, b: m256i): m256i {.importc: "_mm256_add_epi16".}
proc mm256_sub_epi16*(a, b: m256i): m256i {.importc: "_mm256_sub_epi16".}
proc mm256_mullo_epi16*(a, b: m256i): m256i {.importc: "_mm256_mullo_epi16".}
proc mm256_mulhi_epi16*(a, b: m256i): m256i {.importc: "_mm256_mulhi_epi16".}
proc mm256_mulhi_epu16*(a, b: m256i): m256i {.importc: "_mm256_mulhi_epu16".}
proc mm256_adds_epi16*(a, b: m256i): m256i {.importc: "_mm256_adds_epi16".}
proc mm256_adds_epu16*(a, b: m256i): m256i {.importc: "_mm256_adds_epu16".}
proc mm256_subs_epi16*(a, b: m256i): m256i {.importc: "_mm256_subs_epi16".}
proc mm256_subs_epu16*(a, b: m256i): m256i {.importc: "_mm256_subs_epu16".}
proc mm256_abs_epi16*(a: m256i): m256i {.importc: "_mm256_abs_epi16".}
proc mm256_sign_epi16*(a, b: m256i): m256i {.importc: "_mm256_sign_epi16".}
proc mm256_madd_epi16*(a, b: m256i): m256i {.importc: "_mm256_madd_epi16".}
proc mm256_maddubs_epi16*(a, b: m256i): m256i {.importc: "_mm256_maddubs_epi16".}

# Arithmetic operations - 32-bit integers
proc mm256_add_epi32*(a, b: m256i): m256i {.importc: "_mm256_add_epi32".}
proc mm256_sub_epi32*(a, b: m256i): m256i {.importc: "_mm256_sub_epi32".}
proc mm256_mullo_epi32*(a, b: m256i): m256i {.importc: "_mm256_mullo_epi32".}
proc mm256_mul_epi32*(a, b: m256i): m256i {.importc: "_mm256_mul_epi32".}
proc mm256_mul_epu32*(a, b: m256i): m256i {.importc: "_mm256_mul_epu32".}
proc mm256_abs_epi32*(a: m256i): m256i {.importc: "_mm256_abs_epi32".}
proc mm256_sign_epi32*(a, b: m256i): m256i {.importc: "_mm256_sign_epi32".}

# Arithmetic operations - 64-bit integers
proc mm256_add_epi64*(a, b: m256i): m256i {.importc: "_mm256_add_epi64".}
proc mm256_sub_epi64*(a, b: m256i): m256i {.importc: "_mm256_sub_epi64".}
proc mm256_mul_epi64*(a, b: m256i): m256i {.importc: "_mm256_mul_epi64".}

# Logical operations
proc mm256_and_si256*(a, b: m256i): m256i {.importc: "_mm256_and_si256".}
proc mm256_or_si256*(a, b: m256i): m256i {.importc: "_mm256_or_si256".}
proc mm256_xor_si256*(a, b: m256i): m256i {.importc: "_mm256_xor_si256".}
proc mm256_andnot_si256*(a, b: m256i): m256i {.importc: "_mm256_andnot_si256".}

# Comparison operations - 8-bit
proc mm256_cmpeq_epi8*(a, b: m256i): m256i {.importc: "_mm256_cmpeq_epi8".}
proc mm256_cmpgt_epi8*(a, b: m256i): m256i {.importc: "_mm256_cmpgt_epi8".}
proc mm256_min_epi8*(a, b: m256i): m256i {.importc: "_mm256_min_epi8".}
proc mm256_min_epu8*(a, b: m256i): m256i {.importc: "_mm256_min_epu8".}
proc mm256_max_epi8*(a, b: m256i): m256i {.importc: "_mm256_max_epi8".}
proc mm256_max_epu8*(a, b: m256i): m256i {.importc: "_mm256_max_epu8".}

# Comparison operations - 16-bit
proc mm256_cmpeq_epi16*(a, b: m256i): m256i {.importc: "_mm256_cmpeq_epi16".}
proc mm256_cmpgt_epi16*(a, b: m256i): m256i {.importc: "_mm256_cmpgt_epi16".}
proc mm256_min_epi16*(a, b: m256i): m256i {.importc: "_mm256_min_epi16".}
proc mm256_min_epu16*(a, b: m256i): m256i {.importc: "_mm256_min_epu16".}
proc mm256_max_epi16*(a, b: m256i): m256i {.importc: "_mm256_max_epi16".}
proc mm256_max_epu16*(a, b: m256i): m256i {.importc: "_mm256_max_epu16".}

# Comparison operations - 32-bit
proc mm256_cmpeq_epi32*(a, b: m256i): m256i {.importc: "_mm256_cmpeq_epi32".}
proc mm256_cmpgt_epi32*(a, b: m256i): m256i {.importc: "_mm256_cmpgt_epi32".}
proc mm256_min_epi32*(a, b: m256i): m256i {.importc: "_mm256_min_epi32".}
proc mm256_min_epu32*(a, b: m256i): m256i {.importc: "_mm256_min_epu32".}
proc mm256_max_epi32*(a, b: m256i): m256i {.importc: "_mm256_max_epi32".}
proc mm256_max_epu32*(a, b: m256i): m256i {.importc: "_mm256_max_epu32".}

# Comparison operations - 64-bit
proc mm256_cmpeq_epi64*(a, b: m256i): m256i {.importc: "_mm256_cmpeq_epi64".}
proc mm256_cmpgt_epi64*(a, b: m256i): m256i {.importc: "_mm256_cmpgt_epi64".}

# Shift operations
proc mm256_sll_epi16*(a: m256i, count: m256i): m256i {.importc: "_mm256_sll_epi16".}
proc mm256_sll_epi32*(a: m256i, count: m256i): m256i {.importc: "_mm256_sll_epi32".}
proc mm256_sll_epi64*(a: m256i, count: m256i): m256i {.importc: "_mm256_sll_epi64".}
proc mm256_slli_epi16*(a: m256i, imm8: int32): m256i {.importc: "_mm256_slli_epi16".}
proc mm256_slli_epi32*(a: m256i, imm8: int32): m256i {.importc: "_mm256_slli_epi32".}
proc mm256_slli_epi64*(a: m256i, imm8: int32): m256i {.importc: "_mm256_slli_epi64".}

proc mm256_srl_epi16*(a: m256i, count: m256i): m256i {.importc: "_mm256_srl_epi16".}
proc mm256_srl_epi32*(a: m256i, count: m256i): m256i {.importc: "_mm256_srl_epi32".}
proc mm256_srl_epi64*(a: m256i, count: m256i): m256i {.importc: "_mm256_srl_epi64".}
proc mm256_srli_epi16*(a: m256i, imm8: int32): m256i {.importc: "_mm256_srli_epi16".}
proc mm256_srli_epi32*(a: m256i, imm8: int32): m256i {.importc: "_mm256_srli_epi32".}
proc mm256_srli_epi64*(a: m256i, imm8: int32): m256i {.importc: "_mm256_srli_epi64".}

proc mm256_sra_epi16*(a: m256i, count: m256i): m256i {.importc: "_mm256_sra_epi16".}
proc mm256_sra_epi32*(a: m256i, count: m256i): m256i {.importc: "_mm256_sra_epi32".}
proc mm256_srai_epi16*(a: m256i, imm8: int32): m256i {.importc: "_mm256_srai_epi16".}
proc mm256_srai_epi32*(a: m256i, imm8: int32): m256i {.importc: "_mm256_srai_epi32".}

# Shuffle and permute operations
proc mm256_shuffle_epi8*(a, b: m256i): m256i {.importc: "_mm256_shuffle_epi8".}
proc mm256_shuffle_epi32*(a: m256i, imm8: int32): m256i {.importc: "_mm256_shuffle_epi32".}
proc mm256_shufflehi_epi16*(a: m256i, imm8: int32): m256i {.importc: "_mm256_shufflehi_epi16".}
proc mm256_shufflelo_epi16*(a: m256i, imm8: int32): m256i {.importc: "_mm256_shufflelo_epi16".}

proc mm256_unpackhi_epi8*(a, b: m256i): m256i {.importc: "_mm256_unpackhi_epi8".}
proc mm256_unpackhi_epi16*(a, b: m256i): m256i {.importc: "_mm256_unpackhi_epi16".}
proc mm256_unpackhi_epi32*(a, b: m256i): m256i {.importc: "_mm256_unpackhi_epi32".}
proc mm256_unpackhi_epi64*(a, b: m256i): m256i {.importc: "_mm256_unpackhi_epi64".}
proc mm256_unpacklo_epi8*(a, b: m256i): m256i {.importc: "_mm256_unpacklo_epi8".}
proc mm256_unpacklo_epi16*(a, b: m256i): m256i {.importc: "_mm256_unpacklo_epi16".}
proc mm256_unpacklo_epi32*(a, b: m256i): m256i {.importc: "_mm256_unpacklo_epi32".}
proc mm256_unpacklo_epi64*(a, b: m256i): m256i {.importc: "_mm256_unpacklo_epi64".}

# Pack operations
proc mm256_packus_epi16*(a, b: m256i): m256i {.importc: "_mm256_packus_epi16".}
proc mm256_packus_epi32*(a, b: m256i): m256i {.importc: "_mm256_packus_epi32".}
proc mm256_packs_epi16*(a, b: m256i): m256i {.importc: "_mm256_packs_epi16".}
proc mm256_packs_epi32*(a, b: m256i): m256i {.importc: "_mm256_packs_epi32".}

# Conversion operations
proc mm256_cvtepi8_epi16*(a: m256i): m256i {.importc: "_mm256_cvtepi8_epi16".}
proc mm256_cvtepi8_epi32*(a: m256i): m256i {.importc: "_mm256_cvtepi8_epi32".}
proc mm256_cvtepi8_epi64*(a: m256i): m256i {.importc: "_mm256_cvtepi8_epi64".}
proc mm256_cvtepi16_epi32*(a: m256i): m256i {.importc: "_mm256_cvtepi16_epi32".}
proc mm256_cvtepi16_epi64*(a: m256i): m256i {.importc: "_mm256_cvtepi16_epi64".}
proc mm256_cvtepi32_epi64*(a: m256i): m256i {.importc: "_mm256_cvtepi32_epi64".}

proc mm256_cvtepu8_epi16*(a: m256i): m256i {.importc: "_mm256_cvtepu8_epi16".}
proc mm256_cvtepu8_epi32*(a: m256i): m256i {.importc: "_mm256_cvtepu8_epi32".}
proc mm256_cvtepu8_epi64*(a: m256i): m256i {.importc: "_mm256_cvtepu8_epi64".}
proc mm256_cvtepu16_epi32*(a: m256i): m256i {.importc: "_mm256_cvtepu16_epi32".}
proc mm256_cvtepu16_epi64*(a: m256i): m256i {.importc: "_mm256_cvtepu16_epi64".}
proc mm256_cvtepu32_epi64*(a: m256i): m256i {.importc: "_mm256_cvtepu32_epi64".}

# Extract and insert operations
proc mm256_extract_epi8*(a: m256i, index: int32): int32 {.importc: "_mm256_extract_epi8".}
proc mm256_extract_epi16*(a: m256i, index: int32): int32 {.importc: "_mm256_extract_epi16".}
proc mm256_extract_epi32*(a: m256i, index: int32): int32 {.importc: "_mm256_extract_epi32".}
proc mm256_extract_epi64*(a: m256i, index: int32): int64 {.importc: "_mm256_extract_epi64".}

proc mm256_insert_epi8*(a: m256i, i: int32, index: int32): m256i {.importc: "_mm256_insert_epi8".}
proc mm256_insert_epi16*(a: m256i, i: int32, index: int32): m256i {.importc: "_mm256_insert_epi16".}
proc mm256_insert_epi32*(a: m256i, i: int32, index: int32): m256i {.importc: "_mm256_insert_epi32".}
proc mm256_insert_epi64*(a: m256i, i: int64, index: int32): m256i {.importc: "_mm256_insert_epi64".}

# Horizontal operations
proc mm256_hadd_epi16*(a, b: m256i): m256i {.importc: "_mm256_hadd_epi16".}
proc mm256_hadd_epi32*(a, b: m256i): m256i {.importc: "_mm256_hadd_epi32".}
proc mm256_hsub_epi16*(a, b: m256i): m256i {.importc: "_mm256_hsub_epi16".}
proc mm256_hsub_epi32*(a, b: m256i): m256i {.importc: "_mm256_hsub_epi32".}
proc mm256_hadds_epi16*(a, b: m256i): m256i {.importc: "_mm256_hadds_epi16".}
proc mm256_hsubs_epi16*(a, b: m256i): m256i {.importc: "_mm256_hsubs_epi16".}

# Blend operations
proc mm256_blend_epi16*(a, b: m256i, imm8: int32): m256i {.importc: "_mm256_blend_epi16".}
proc mm256_blend_epi32*(a, b: m256i, imm8: int32): m256i {.importc: "_mm256_blend_epi32".}
proc mm256_blendv_epi8*(a, b, mask: m256i): m256i {.importc: "_mm256_blendv_epi8".}

# Broadcast operations
proc mm256_broadcastb_epi8*(a: m256i): m256i {.importc: "_mm256_broadcastb_epi8".}
proc mm256_broadcastw_epi16*(a: m256i): m256i {.importc: "_mm256_broadcastw_epi16".}
proc mm256_broadcastd_epi32*(a: m256i): m256i {.importc: "_mm256_broadcastd_epi32".}
proc mm256_broadcastq_epi64*(a: m256i): m256i {.importc: "_mm256_broadcastq_epi64".}

# Gather operations
proc mm256_i32gather_epi32*(base_addr: ptr int32, vindex: m256i, scale: int32): m256i {.importc: "_mm256_i32gather_epi32".}
proc mm256_i32gather_epi64*(base_addr: ptr int64, vindex: m256i, scale: int32): m256i {.importc: "_mm256_i32gather_epi64".}
proc mm256_i64gather_epi32*(base_addr: ptr int32, vindex: m256i, scale: int32): m256i {.importc: "_mm256_i64gather_epi32".}
proc mm256_i64gather_epi64*(base_addr: ptr int64, vindex: m256i, scale: int32): m256i {.importc: "_mm256_i64gather_epi64".}

# Float operations
proc mm256_setzero_ps*(): m256 {.importc: "_mm256_setzero_ps".}
proc mm256_setzero_pd*(): m256d {.importc: "_mm256_setzero_pd".}
proc mm256_set1_ps*(a: float32): m256 {.importc: "_mm256_set1_ps".}
proc mm256_set1_pd*(a: float64): m256d {.importc: "_mm256_set1_pd".}

proc mm256_add_ps*(a, b: m256): m256 {.importc: "_mm256_add_ps".}
proc mm256_sub_ps*(a, b: m256): m256 {.importc: "_mm256_sub_ps".}
proc mm256_mul_ps*(a, b: m256): m256 {.importc: "_mm256_mul_ps".}
proc mm256_div_ps*(a, b: m256): m256 {.importc: "_mm256_div_ps".}
proc mm256_fmadd_ps*(a, b, c: m256): m256 {.importc: "_mm256_fmadd_ps".}
proc mm256_fmsub_ps*(a, b, c: m256): m256 {.importc: "_mm256_fmsub_ps".}

proc mm256_add_pd*(a, b: m256d): m256d {.importc: "_mm256_add_pd".}
proc mm256_sub_pd*(a, b: m256d): m256d {.importc: "_mm256_sub_pd".}
proc mm256_mul_pd*(a, b: m256d): m256d {.importc: "_mm256_mul_pd".}
proc mm256_div_pd*(a, b: m256d): m256d {.importc: "_mm256_div_pd".}
proc mm256_fmadd_pd*(a, b, c: m256d): m256d {.importc: "_mm256_fmadd_pd".}
proc mm256_fmsub_pd*(a, b, c: m256d): m256d {.importc: "_mm256_fmsub_pd".}

# Store operations for floats
proc mm256_storeu_ps*(mem_addr: ptr float32, a: m256) {.importc: "_mm256_storeu_ps".}
proc mm256_storeu_pd*(mem_addr: ptr float64, a: m256d) {.importc: "_mm256_storeu_pd".}

{.pop.}

# Arithmetic operator overloads for integers
proc `+`*(a, b: m256i): m256i = mm256_add_epi32(a, b)
proc `-`*(a, b: m256i): m256i = mm256_sub_epi32(a, b)
proc `*`*(a, b: m256i): m256i = mm256_mullo_epi32(a, b)

# Arithmetic operator overloads for floats
proc `+`*(a, b: m256): m256 = mm256_add_ps(a, b)
proc `-`*(a, b: m256): m256 = mm256_sub_ps(a, b)
proc `*`*(a, b: m256): m256 = mm256_mul_ps(a, b)
proc `/`*(a, b: m256): m256 = mm256_div_ps(a, b)

proc `+`*(a, b: m256d): m256d = mm256_add_pd(a, b)
proc `-`*(a, b: m256d): m256d = mm256_sub_pd(a, b)
proc `*`*(a, b: m256d): m256d = mm256_mul_pd(a, b)
proc `/`*(a, b: m256d): m256d = mm256_div_pd(a, b)

# Logical operator overloads
proc `and`*(a, b: m256i): m256i = mm256_and_si256(a, b)
proc `or`*(a, b: m256i): m256i = mm256_or_si256(a, b)
proc `xor`*(a, b: m256i): m256i = mm256_xor_si256(a, b)
proc `not`*(a: m256i): m256i = mm256_xor_si256(a, mm256_set1_epi32(-1))

# Comparison operator overloads
proc `==`*(a, b: m256i): m256i = mm256_cmpeq_epi32(a, b)
proc `>`*(a, b: m256i): m256i = mm256_cmpgt_epi32(a, b)
proc `<`*(a, b: m256i): m256i = mm256_cmpgt_epi32(b, a)

# Sequtils-style operations
proc sum*(a: m256i): int64 =
  let val0 = mm256_extract_epi32(a, 0).int64
  let val1 = mm256_extract_epi32(a, 1).int64
  let val2 = mm256_extract_epi32(a, 2).int64
  let val3 = mm256_extract_epi32(a, 3).int64
  let val4 = mm256_extract_epi32(a, 4).int64
  let val5 = mm256_extract_epi32(a, 5).int64
  let val6 = mm256_extract_epi32(a, 6).int64
  let val7 = mm256_extract_epi32(a, 7).int64
  result = val0 + val1 + val2 + val3 + val4 + val5 + val6 + val7

proc sum*(a: m256): float32 =
  var temp: array[8, float32]
  mm256_storeu_ps(addr temp[0], a)
  result = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7]

proc sum*(a: m256d): float64 =
  var temp: array[4, float64]
  mm256_storeu_pd(addr temp[0], a)
  result = temp[0] + temp[1] + temp[2] + temp[3]

proc avg*(a: m256i): float64 = sum(a).float64 / 8.0
proc avg*(a: m256): float32 = sum(a) / 8.0
proc avg*(a: m256d): float64 = sum(a) / 4.0

proc min*(a: m256i): int32 =
  let val0 = mm256_extract_epi32(a, 0)
  let val1 = mm256_extract_epi32(a, 1)
  let val2 = mm256_extract_epi32(a, 2)
  let val3 = mm256_extract_epi32(a, 3)
  let val4 = mm256_extract_epi32(a, 4)
  let val5 = mm256_extract_epi32(a, 5)
  let val6 = mm256_extract_epi32(a, 6)
  let val7 = mm256_extract_epi32(a, 7)
  result = min(min(min(val0, val1), min(val2, val3)), 
              min(min(val4, val5), min(val6, val7)))

proc max*(a: m256i): int32 =
  let val0 = mm256_extract_epi32(a, 0)
  let val1 = mm256_extract_epi32(a, 1)
  let val2 = mm256_extract_epi32(a, 2)
  let val3 = mm256_extract_epi32(a, 3)
  let val4 = mm256_extract_epi32(a, 4)
  let val5 = mm256_extract_epi32(a, 5)
  let val6 = mm256_extract_epi32(a, 6)
  let val7 = mm256_extract_epi32(a, 7)
  result = max(max(max(val0, val1), max(val2, val3)), 
              max(max(val4, val5), max(val6, val7)))

# Type-specific arithmetic operators for different bit widths
proc add8*(a, b: m256i): m256i = mm256_add_epi8(a, b)
proc sub8*(a, b: m256i): m256i = mm256_sub_epi8(a, b)

proc add16*(a, b: m256i): m256i = mm256_add_epi16(a, b)
proc sub16*(a, b: m256i): m256i = mm256_sub_epi16(a, b)
proc mul16*(a, b: m256i): m256i = mm256_mullo_epi16(a, b)

proc add32*(a, b: m256i): m256i = mm256_add_epi32(a, b)
proc sub32*(a, b: m256i): m256i = mm256_sub_epi32(a, b)
proc mul32*(a, b: m256i): m256i = mm256_mullo_epi32(a, b)

proc add64*(a, b: m256i): m256i = mm256_add_epi64(a, b)
proc sub64*(a, b: m256i): m256i = mm256_sub_epi64(a, b)

# Shift operators
proc `shl`*(a: m256i, count: int32): m256i = mm256_slli_epi32(a, count)
proc `shr`*(a: m256i, count: int32): m256i = mm256_srli_epi32(a, count)

# Utility constructors
proc vec8*(val: int8): m256i = mm256_set1_epi8(val)
proc vec16*(val: int16): m256i = mm256_set1_epi16(val)
proc vec32*(val: int32): m256i = mm256_set1_epi32(val)
proc vec64*(val: int64): m256i = mm256_set1_epi64x(val)
proc vecf*(val: float32): m256 = mm256_set1_ps(val)
proc vecd*(val: float64): m256d = mm256_set1_pd(val)

proc zero*(): m256i = mm256_setzero_si256()
proc zerof*(): m256 = mm256_setzero_ps()
proc zerod*(): m256d = mm256_setzero_pd()