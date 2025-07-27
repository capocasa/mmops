## ARM NEON Wrapper for Nim
## 
## This module provides comprehensive wrappers for all ARM NEON instructions
## along with convenient operator overloads and sequence-like operations.

{.push header: "arm_neon.h".}

type
  # 64-bit vector types
  int8x8_t* {.importc: "int8x8_t", header: "arm_neon.h".} = object
  int16x4_t* {.importc: "int16x4_t", header: "arm_neon.h".} = object
  int32x2_t* {.importc: "int32x2_t", header: "arm_neon.h".} = object
  int64x1_t* {.importc: "int64x1_t", header: "arm_neon.h".} = object
  
  uint8x8_t* {.importc: "uint8x8_t", header: "arm_neon.h".} = object
  uint16x4_t* {.importc: "uint16x4_t", header: "arm_neon.h".} = object
  uint32x2_t* {.importc: "uint32x2_t", header: "arm_neon.h".} = object
  uint64x1_t* {.importc: "uint64x1_t", header: "arm_neon.h".} = object
  
  float32x2_t* {.importc: "float32x2_t", header: "arm_neon.h".} = object
  float64x1_t* {.importc: "float64x1_t", header: "arm_neon.h".} = object
  
  # 128-bit vector types (quad)
  int8x16_t* {.importc: "int8x16_t", header: "arm_neon.h".} = object
  int16x8_t* {.importc: "int16x8_t", header: "arm_neon.h".} = object
  int32x4_t* {.importc: "int32x4_t", header: "arm_neon.h".} = object
  int64x2_t* {.importc: "int64x2_t", header: "arm_neon.h".} = object
  
  uint8x16_t* {.importc: "uint8x16_t", header: "arm_neon.h".} = object
  uint16x8_t* {.importc: "uint16x8_t", header: "arm_neon.h".} = object
  uint32x4_t* {.importc: "uint32x4_t", header: "arm_neon.h".} = object
  uint64x2_t* {.importc: "uint64x2_t", header: "arm_neon.h".} = object
  
  float32x4_t* {.importc: "float32x4_t", header: "arm_neon.h".} = object
  float64x2_t* {.importc: "float64x2_t", header: "arm_neon.h".} = object

# Vector creation functions - 64-bit
proc vdup_n_s8*(value: int8): int8x8_t {.importc: "vdup_n_s8".}
proc vdup_n_s16*(value: int16): int16x4_t {.importc: "vdup_n_s16".}
proc vdup_n_s32*(value: int32): int32x2_t {.importc: "vdup_n_s32".}
proc vdup_n_s64*(value: int64): int64x1_t {.importc: "vdup_n_s64".}

proc vdup_n_u8*(value: uint8): uint8x8_t {.importc: "vdup_n_u8".}
proc vdup_n_u16*(value: uint16): uint16x4_t {.importc: "vdup_n_u16".}
proc vdup_n_u32*(value: uint32): uint32x2_t {.importc: "vdup_n_u32".}
proc vdup_n_u64*(value: uint64): uint64x1_t {.importc: "vdup_n_u64".}

proc vdup_n_f32*(value: float32): float32x2_t {.importc: "vdup_n_f32".}
proc vdup_n_f64*(value: float64): float64x1_t {.importc: "vdup_n_f64".}

# Vector creation functions - 128-bit (quad)
proc vdupq_n_s8*(value: int8): int8x16_t {.importc: "vdupq_n_s8".}
proc vdupq_n_s16*(value: int16): int16x8_t {.importc: "vdupq_n_s16".}
proc vdupq_n_s32*(value: int32): int32x4_t {.importc: "vdupq_n_s32".}
proc vdupq_n_s64*(value: int64): int64x2_t {.importc: "vdupq_n_s64".}

proc vdupq_n_u8*(value: uint8): uint8x16_t {.importc: "vdupq_n_u8".}
proc vdupq_n_u16*(value: uint16): uint16x8_t {.importc: "vdupq_n_u16".}
proc vdupq_n_u32*(value: uint32): uint32x4_t {.importc: "vdupq_n_u32".}
proc vdupq_n_u64*(value: uint64): uint64x2_t {.importc: "vdupq_n_u64".}

proc vdupq_n_f32*(value: float32): float32x4_t {.importc: "vdupq_n_f32".}
proc vdupq_n_f64*(value: float64): float64x2_t {.importc: "vdupq_n_f64".}

# Arithmetic operations - Addition (64-bit)
proc vadd_s8*(a, b: int8x8_t): int8x8_t {.importc: "vadd_s8".}
proc vadd_s16*(a, b: int16x4_t): int16x4_t {.importc: "vadd_s16".}
proc vadd_s32*(a, b: int32x2_t): int32x2_t {.importc: "vadd_s32".}
proc vadd_s64*(a, b: int64x1_t): int64x1_t {.importc: "vadd_s64".}

proc vadd_u8*(a, b: uint8x8_t): uint8x8_t {.importc: "vadd_u8".}
proc vadd_u16*(a, b: uint16x4_t): uint16x4_t {.importc: "vadd_u16".}
proc vadd_u32*(a, b: uint32x2_t): uint32x2_t {.importc: "vadd_u32".}
proc vadd_u64*(a, b: uint64x1_t): uint64x1_t {.importc: "vadd_u64".}

proc vadd_f32*(a, b: float32x2_t): float32x2_t {.importc: "vadd_f32".}
proc vadd_f64*(a, b: float64x1_t): float64x1_t {.importc: "vadd_f64".}

# Arithmetic operations - Addition (128-bit)
proc vaddq_s8*(a, b: int8x16_t): int8x16_t {.importc: "vaddq_s8".}
proc vaddq_s16*(a, b: int16x8_t): int16x8_t {.importc: "vaddq_s16".}
proc vaddq_s32*(a, b: int32x4_t): int32x4_t {.importc: "vaddq_s32".}
proc vaddq_s64*(a, b: int64x2_t): int64x2_t {.importc: "vaddq_s64".}

proc vaddq_u8*(a, b: uint8x16_t): uint8x16_t {.importc: "vaddq_u8".}
proc vaddq_u16*(a, b: uint16x8_t): uint16x8_t {.importc: "vaddq_u16".}
proc vaddq_u32*(a, b: uint32x4_t): uint32x4_t {.importc: "vaddq_u32".}
proc vaddq_u64*(a, b: uint64x2_t): uint64x2_t {.importc: "vaddq_u64".}

proc vaddq_f32*(a, b: float32x4_t): float32x4_t {.importc: "vaddq_f32".}
proc vaddq_f64*(a, b: float64x2_t): float64x2_t {.importc: "vaddq_f64".}

# Arithmetic operations - Subtraction (64-bit)
proc vsub_s8*(a, b: int8x8_t): int8x8_t {.importc: "vsub_s8".}
proc vsub_s16*(a, b: int16x4_t): int16x4_t {.importc: "vsub_s16".}
proc vsub_s32*(a, b: int32x2_t): int32x2_t {.importc: "vsub_s32".}
proc vsub_s64*(a, b: int64x1_t): int64x1_t {.importc: "vsub_s64".}

proc vsub_u8*(a, b: uint8x8_t): uint8x8_t {.importc: "vsub_u8".}
proc vsub_u16*(a, b: uint16x4_t): uint16x4_t {.importc: "vsub_u16".}
proc vsub_u32*(a, b: uint32x2_t): uint32x2_t {.importc: "vsub_u32".}
proc vsub_u64*(a, b: uint64x1_t): uint64x1_t {.importc: "vsub_u64".}

proc vsub_f32*(a, b: float32x2_t): float32x2_t {.importc: "vsub_f32".}
proc vsub_f64*(a, b: float64x1_t): float64x1_t {.importc: "vsub_f64".}

# Arithmetic operations - Subtraction (128-bit)
proc vsubq_s8*(a, b: int8x16_t): int8x16_t {.importc: "vsubq_s8".}
proc vsubq_s16*(a, b: int16x8_t): int16x8_t {.importc: "vsubq_s16".}
proc vsubq_s32*(a, b: int32x4_t): int32x4_t {.importc: "vsubq_s32".}
proc vsubq_s64*(a, b: int64x2_t): int64x2_t {.importc: "vsubq_s64".}

proc vsubq_u8*(a, b: uint8x16_t): uint8x16_t {.importc: "vsubq_u8".}
proc vsubq_u16*(a, b: uint16x8_t): uint16x8_t {.importc: "vsubq_u16".}
proc vsubq_u32*(a, b: uint32x4_t): uint32x4_t {.importc: "vsubq_u32".}
proc vsubq_u64*(a, b: uint64x2_t): uint64x2_t {.importc: "vsubq_u64".}

proc vsubq_f32*(a, b: float32x4_t): float32x4_t {.importc: "vsubq_f32".}
proc vsubq_f64*(a, b: float64x2_t): float64x2_t {.importc: "vsubq_f64".}

# Arithmetic operations - Multiplication (64-bit)
proc vmul_s8*(a, b: int8x8_t): int8x8_t {.importc: "vmul_s8".}
proc vmul_s16*(a, b: int16x4_t): int16x4_t {.importc: "vmul_s16".}
proc vmul_s32*(a, b: int32x2_t): int32x2_t {.importc: "vmul_s32".}

proc vmul_u8*(a, b: uint8x8_t): uint8x8_t {.importc: "vmul_u8".}
proc vmul_u16*(a, b: uint16x4_t): uint16x4_t {.importc: "vmul_u16".}
proc vmul_u32*(a, b: uint32x2_t): uint32x2_t {.importc: "vmul_u32".}

proc vmul_f32*(a, b: float32x2_t): float32x2_t {.importc: "vmul_f32".}
proc vmul_f64*(a, b: float64x1_t): float64x1_t {.importc: "vmul_f64".}

# Arithmetic operations - Multiplication (128-bit)
proc vmulq_s8*(a, b: int8x16_t): int8x16_t {.importc: "vmulq_s8".}
proc vmulq_s16*(a, b: int16x8_t): int16x8_t {.importc: "vmulq_s16".}
proc vmulq_s32*(a, b: int32x4_t): int32x4_t {.importc: "vmulq_s32".}

proc vmulq_u8*(a, b: uint8x16_t): uint8x16_t {.importc: "vmulq_u8".}
proc vmulq_u16*(a, b: uint16x8_t): uint16x8_t {.importc: "vmulq_u16".}
proc vmulq_u32*(a, b: uint32x4_t): uint32x4_t {.importc: "vmulq_u32".}

proc vmulq_f32*(a, b: float32x4_t): float32x4_t {.importc: "vmulq_f32".}
proc vmulq_f64*(a, b: float64x2_t): float64x2_t {.importc: "vmulq_f64".}

# Division (float only)
proc vdiv_f32*(a, b: float32x2_t): float32x2_t {.importc: "vdiv_f32".}
proc vdiv_f64*(a, b: float64x1_t): float64x1_t {.importc: "vdiv_f64".}
proc vdivq_f32*(a, b: float32x4_t): float32x4_t {.importc: "vdivq_f32".}
proc vdivq_f64*(a, b: float64x2_t): float64x2_t {.importc: "vdivq_f64".}

# Logical operations (64-bit)
proc vand_s8*(a, b: int8x8_t): int8x8_t {.importc: "vand_s8".}
proc vand_s16*(a, b: int16x4_t): int16x4_t {.importc: "vand_s16".}
proc vand_s32*(a, b: int32x2_t): int32x2_t {.importc: "vand_s32".}
proc vand_s64*(a, b: int64x1_t): int64x1_t {.importc: "vand_s64".}

proc vorr_s8*(a, b: int8x8_t): int8x8_t {.importc: "vorr_s8".}
proc vorr_s16*(a, b: int16x4_t): int16x4_t {.importc: "vorr_s16".}
proc vorr_s32*(a, b: int32x2_t): int32x2_t {.importc: "vorr_s32".}
proc vorr_s64*(a, b: int64x1_t): int64x1_t {.importc: "vorr_s64".}

proc veor_s8*(a, b: int8x8_t): int8x8_t {.importc: "veor_s8".}
proc veor_s16*(a, b: int16x4_t): int16x4_t {.importc: "veor_s16".}
proc veor_s32*(a, b: int32x2_t): int32x2_t {.importc: "veor_s32".}
proc veor_s64*(a, b: int64x1_t): int64x1_t {.importc: "veor_s64".}

proc vmvn_s8*(a: int8x8_t): int8x8_t {.importc: "vmvn_s8".}
proc vmvn_s16*(a: int16x4_t): int16x4_t {.importc: "vmvn_s16".}
proc vmvn_s32*(a: int32x2_t): int32x2_t {.importc: "vmvn_s32".}

# Logical operations (128-bit)
proc vandq_s8*(a, b: int8x16_t): int8x16_t {.importc: "vandq_s8".}
proc vandq_s16*(a, b: int16x8_t): int16x8_t {.importc: "vandq_s16".}
proc vandq_s32*(a, b: int32x4_t): int32x4_t {.importc: "vandq_s32".}
proc vandq_s64*(a, b: int64x2_t): int64x2_t {.importc: "vandq_s64".}

proc vorrq_s8*(a, b: int8x16_t): int8x16_t {.importc: "vorrq_s8".}
proc vorrq_s16*(a, b: int16x8_t): int16x8_t {.importc: "vorrq_s16".}
proc vorrq_s32*(a, b: int32x4_t): int32x4_t {.importc: "vorrq_s32".}
proc vorrq_s64*(a, b: int64x2_t): int64x2_t {.importc: "vorrq_s64".}

proc veorq_s8*(a, b: int8x16_t): int8x16_t {.importc: "veorq_s8".}
proc veorq_s16*(a, b: int16x8_t): int16x8_t {.importc: "veorq_s16".}
proc veorq_s32*(a, b: int32x4_t): int32x4_t {.importc: "veorq_s32".}
proc veorq_s64*(a, b: int64x2_t): int64x2_t {.importc: "veorq_s64".}

proc vmvnq_s8*(a: int8x16_t): int8x16_t {.importc: "vmvnq_s8".}
proc vmvnq_s16*(a: int16x8_t): int16x8_t {.importc: "vmvnq_s16".}
proc vmvnq_s32*(a: int32x4_t): int32x4_t {.importc: "vmvnq_s32".}

# Comparison operations (64-bit)
proc vceq_s8*(a, b: int8x8_t): uint8x8_t {.importc: "vceq_s8".}
proc vceq_s16*(a, b: int16x4_t): uint16x4_t {.importc: "vceq_s16".}
proc vceq_s32*(a, b: int32x2_t): uint32x2_t {.importc: "vceq_s32".}

proc vcgt_s8*(a, b: int8x8_t): uint8x8_t {.importc: "vcgt_s8".}
proc vcgt_s16*(a, b: int16x4_t): uint16x4_t {.importc: "vcgt_s16".}
proc vcgt_s32*(a, b: int32x2_t): uint32x2_t {.importc: "vcgt_s32".}

proc vclt_s8*(a, b: int8x8_t): uint8x8_t {.importc: "vclt_s8".}
proc vclt_s16*(a, b: int16x4_t): uint16x4_t {.importc: "vclt_s16".}
proc vclt_s32*(a, b: int32x2_t): uint32x2_t {.importc: "vclt_s32".}

# Comparison operations (128-bit)
proc vceqq_s8*(a, b: int8x16_t): uint8x16_t {.importc: "vceqq_s8".}
proc vceqq_s16*(a, b: int16x8_t): uint16x8_t {.importc: "vceqq_s16".}
proc vceqq_s32*(a, b: int32x4_t): uint32x4_t {.importc: "vceqq_s32".}

proc vcgtq_s8*(a, b: int8x16_t): uint8x16_t {.importc: "vcgtq_s8".}
proc vcgtq_s16*(a, b: int16x8_t): uint16x8_t {.importc: "vcgtq_s16".}
proc vcgtq_s32*(a, b: int32x4_t): uint32x4_t {.importc: "vcgtq_s32".}

proc vcltq_s8*(a, b: int8x16_t): uint8x16_t {.importc: "vcltq_s8".}
proc vcltq_s16*(a, b: int16x8_t): uint16x8_t {.importc: "vcltq_s16".}
proc vcltq_s32*(a, b: int32x4_t): uint32x4_t {.importc: "vcltq_s32".}

# Min/Max operations (64-bit)
proc vmin_s8*(a, b: int8x8_t): int8x8_t {.importc: "vmin_s8".}
proc vmin_s16*(a, b: int16x4_t): int16x4_t {.importc: "vmin_s16".}
proc vmin_s32*(a, b: int32x2_t): int32x2_t {.importc: "vmin_s32".}

proc vmax_s8*(a, b: int8x8_t): int8x8_t {.importc: "vmax_s8".}
proc vmax_s16*(a, b: int16x4_t): int16x4_t {.importc: "vmax_s16".}
proc vmax_s32*(a, b: int32x2_t): int32x2_t {.importc: "vmax_s32".}

# Min/Max operations (128-bit)
proc vminq_s8*(a, b: int8x16_t): int8x16_t {.importc: "vminq_s8".}
proc vminq_s16*(a, b: int16x8_t): int16x8_t {.importc: "vminq_s16".}
proc vminq_s32*(a, b: int32x4_t): int32x4_t {.importc: "vminq_s32".}

proc vmaxq_s8*(a, b: int8x16_t): int8x16_t {.importc: "vmaxq_s8".}
proc vmaxq_s16*(a, b: int16x8_t): int16x8_t {.importc: "vmaxq_s16".}
proc vmaxq_s32*(a, b: int32x4_t): int32x4_t {.importc: "vmaxq_s32".}

# Shift operations (64-bit)
proc vshl_s8*(a: int8x8_t, b: int8x8_t): int8x8_t {.importc: "vshl_s8".}
proc vshl_s16*(a: int16x4_t, b: int16x4_t): int16x4_t {.importc: "vshl_s16".}
proc vshl_s32*(a: int32x2_t, b: int32x2_t): int32x2_t {.importc: "vshl_s32".}
proc vshl_s64*(a: int64x1_t, b: int64x1_t): int64x1_t {.importc: "vshl_s64".}

proc vshl_n_s8*(a: int8x8_t, n: int32): int8x8_t {.importc: "vshl_n_s8".}
proc vshl_n_s16*(a: int16x4_t, n: int32): int16x4_t {.importc: "vshl_n_s16".}
proc vshl_n_s32*(a: int32x2_t, n: int32): int32x2_t {.importc: "vshl_n_s32".}
proc vshl_n_s64*(a: int64x1_t, n: int32): int64x1_t {.importc: "vshl_n_s64".}

proc vshr_n_s8*(a: int8x8_t, n: int32): int8x8_t {.importc: "vshr_n_s8".}
proc vshr_n_s16*(a: int16x4_t, n: int32): int16x4_t {.importc: "vshr_n_s16".}
proc vshr_n_s32*(a: int32x2_t, n: int32): int32x2_t {.importc: "vshr_n_s32".}
proc vshr_n_s64*(a: int64x1_t, n: int32): int64x1_t {.importc: "vshr_n_s64".}

# Shift operations (128-bit)
proc vshlq_s8*(a: int8x16_t, b: int8x16_t): int8x16_t {.importc: "vshlq_s8".}
proc vshlq_s16*(a: int16x8_t, b: int16x8_t): int16x8_t {.importc: "vshlq_s16".}
proc vshlq_s32*(a: int32x4_t, b: int32x4_t): int32x4_t {.importc: "vshlq_s32".}
proc vshlq_s64*(a: int64x2_t, b: int64x2_t): int64x2_t {.importc: "vshlq_s64".}

proc vshlq_n_s8*(a: int8x16_t, n: int32): int8x16_t {.importc: "vshlq_n_s8".}
proc vshlq_n_s16*(a: int16x8_t, n: int32): int16x8_t {.importc: "vshlq_n_s16".}
proc vshlq_n_s32*(a: int32x4_t, n: int32): int32x4_t {.importc: "vshlq_n_s32".}
proc vshlq_n_s64*(a: int64x2_t, n: int32): int64x2_t {.importc: "vshlq_n_s64".}

proc vshrq_n_s8*(a: int8x16_t, n: int32): int8x16_t {.importc: "vshrq_n_s8".}
proc vshrq_n_s16*(a: int16x8_t, n: int32): int16x8_t {.importc: "vshrq_n_s16".}
proc vshrq_n_s32*(a: int32x4_t, n: int32): int32x4_t {.importc: "vshrq_n_s32".}
proc vshrq_n_s64*(a: int64x2_t, n: int32): int64x2_t {.importc: "vshrq_n_s64".}

# Load/Store operations
proc vld1_s8*(mem_ptr: ptr int8): int8x8_t {.importc: "vld1_s8".}
proc vld1_s16*(mem_ptr: ptr int16): int16x4_t {.importc: "vld1_s16".}
proc vld1_s32*(mem_ptr: ptr int32): int32x2_t {.importc: "vld1_s32".}
proc vld1_s64*(mem_ptr: ptr int64): int64x1_t {.importc: "vld1_s64".}

proc vld1q_s8*(mem_ptr: ptr int8): int8x16_t {.importc: "vld1q_s8".}
proc vld1q_s16*(mem_ptr: ptr int16): int16x8_t {.importc: "vld1q_s16".}
proc vld1q_s32*(mem_ptr: ptr int32): int32x4_t {.importc: "vld1q_s32".}
proc vld1q_s64*(mem_ptr: ptr int64): int64x2_t {.importc: "vld1q_s64".}

proc vld1_f32*(mem_ptr: ptr float32): float32x2_t {.importc: "vld1_f32".}
proc vld1_f64*(mem_ptr: ptr float64): float64x1_t {.importc: "vld1_f64".}
proc vld1q_f32*(mem_ptr: ptr float32): float32x4_t {.importc: "vld1q_f32".}
proc vld1q_f64*(mem_ptr: ptr float64): float64x2_t {.importc: "vld1q_f64".}

proc vst1_s8*(mem_ptr: ptr int8, val: int8x8_t) {.importc: "vst1_s8".}
proc vst1_s16*(mem_ptr: ptr int16, val: int16x4_t) {.importc: "vst1_s16".}
proc vst1_s32*(mem_ptr: ptr int32, val: int32x2_t) {.importc: "vst1_s32".}
proc vst1_s64*(mem_ptr: ptr int64, val: int64x1_t) {.importc: "vst1_s64".}

proc vst1q_s8*(mem_ptr: ptr int8, val: int8x16_t) {.importc: "vst1q_s8".}
proc vst1q_s16*(mem_ptr: ptr int16, val: int16x8_t) {.importc: "vst1q_s16".}
proc vst1q_s32*(mem_ptr: ptr int32, val: int32x4_t) {.importc: "vst1q_s32".}
proc vst1q_s64*(mem_ptr: ptr int64, val: int64x2_t) {.importc: "vst1q_s64".}

proc vst1_f32*(mem_ptr: ptr float32, val: float32x2_t) {.importc: "vst1_f32".}
proc vst1_f64*(mem_ptr: ptr float64, val: float64x1_t) {.importc: "vst1_f64".}
proc vst1q_f32*(mem_ptr: ptr float32, val: float32x4_t) {.importc: "vst1q_f32".}
proc vst1q_f64*(mem_ptr: ptr float64, val: float64x2_t) {.importc: "vst1q_f64".}

# Horizontal operations
proc vaddv_s8*(a: int8x8_t): int8 {.importc: "vaddv_s8".}
proc vaddv_s16*(a: int16x4_t): int16 {.importc: "vaddv_s16".}
proc vaddv_s32*(a: int32x2_t): int32 {.importc: "vaddv_s32".}

proc vaddvq_s8*(a: int8x16_t): int8 {.importc: "vaddvq_s8".}
proc vaddvq_s16*(a: int16x8_t): int16 {.importc: "vaddvq_s16".}
proc vaddvq_s32*(a: int32x4_t): int32 {.importc: "vaddvq_s32".}
proc vaddvq_s64*(a: int64x2_t): int64 {.importc: "vaddvq_s64".}

proc vaddv_f32*(a: float32x2_t): float32 {.importc: "vaddv_f32".}
proc vaddvq_f32*(a: float32x4_t): float32 {.importc: "vaddvq_f32".}
proc vaddvq_f64*(a: float64x2_t): float64 {.importc: "vaddvq_f64".}

# Extract and insert operations
proc vget_lane_s8*(v: int8x8_t, lane: int32): int8 {.importc: "vget_lane_s8".}
proc vget_lane_s16*(v: int16x4_t, lane: int32): int16 {.importc: "vget_lane_s16".}
proc vget_lane_s32*(v: int32x2_t, lane: int32): int32 {.importc: "vget_lane_s32".}
proc vget_lane_s64*(v: int64x1_t, lane: int32): int64 {.importc: "vget_lane_s64".}

proc vgetq_lane_s8*(v: int8x16_t, lane: int32): int8 {.importc: "vgetq_lane_s8".}
proc vgetq_lane_s16*(v: int16x8_t, lane: int32): int16 {.importc: "vgetq_lane_s16".}
proc vgetq_lane_s32*(v: int32x4_t, lane: int32): int32 {.importc: "vgetq_lane_s32".}
proc vgetq_lane_s64*(v: int64x2_t, lane: int32): int64 {.importc: "vgetq_lane_s64".}

proc vget_lane_f32*(v: float32x2_t, lane: int32): float32 {.importc: "vget_lane_f32".}
proc vget_lane_f64*(v: float64x1_t, lane: int32): float64 {.importc: "vget_lane_f64".}
proc vgetq_lane_f32*(v: float32x4_t, lane: int32): float32 {.importc: "vgetq_lane_f32".}
proc vgetq_lane_f64*(v: float64x2_t, lane: int32): float64 {.importc: "vgetq_lane_f64".}

{.pop.}

# Arithmetic operator overloads for 128-bit vectors (most common)
proc `+`*(a, b: int32x4_t): int32x4_t = vaddq_s32(a, b)
proc `-`*(a, b: int32x4_t): int32x4_t = vsubq_s32(a, b)
proc `*`*(a, b: int32x4_t): int32x4_t = vmulq_s32(a, b)

proc `+`*(a, b: float32x4_t): float32x4_t = vaddq_f32(a, b)
proc `-`*(a, b: float32x4_t): float32x4_t = vsubq_f32(a, b)
proc `*`*(a, b: float32x4_t): float32x4_t = vmulq_f32(a, b)
proc `/`*(a, b: float32x4_t): float32x4_t = vdivq_f32(a, b)

proc `+`*(a, b: float64x2_t): float64x2_t = vaddq_f64(a, b)
proc `-`*(a, b: float64x2_t): float64x2_t = vsubq_f64(a, b)
proc `*`*(a, b: float64x2_t): float64x2_t = vmulq_f64(a, b)
proc `/`*(a, b: float64x2_t): float64x2_t = vdivq_f64(a, b)

# Logical operator overloads
proc `and`*(a, b: int32x4_t): int32x4_t = vandq_s32(a, b)
proc `or`*(a, b: int32x4_t): int32x4_t = vorrq_s32(a, b)
proc `xor`*(a, b: int32x4_t): int32x4_t = veorq_s32(a, b)
proc `not`*(a: int32x4_t): int32x4_t = vmvnq_s32(a)

# Comparison operator overloads (return masks)
proc `==`*(a, b: int32x4_t): uint32x4_t = vceqq_s32(a, b)
proc `>`*(a, b: int32x4_t): uint32x4_t = vcgtq_s32(a, b)
proc `<`*(a, b: int32x4_t): uint32x4_t = vcltq_s32(a, b)

# Sequtils-style operations for 128-bit vectors
proc sum*(a: int32x4_t): int64 = vaddvq_s32(a).int64

proc sum*(a: float32x4_t): float32 = vaddvq_f32(a)

proc sum*(a: float64x2_t): float64 = vaddvq_f64(a)

proc avg*(a: int32x4_t): float64 = sum(a).float64 / 4.0
proc avg*(a: float32x4_t): float32 = sum(a) / 4.0
proc avg*(a: float64x2_t): float64 = sum(a) / 2.0

proc min*(a: int32x4_t): int32 =
  let val0 = vgetq_lane_s32(a, 0)
  let val1 = vgetq_lane_s32(a, 1)
  let val2 = vgetq_lane_s32(a, 2)
  let val3 = vgetq_lane_s32(a, 3)
  result = min(min(val0, val1), min(val2, val3))

proc max*(a: int32x4_t): int32 =
  let val0 = vgetq_lane_s32(a, 0)
  let val1 = vgetq_lane_s32(a, 1)
  let val2 = vgetq_lane_s32(a, 2)
  let val3 = vgetq_lane_s32(a, 3)
  result = max(max(val0, val1), max(val2, val3))

# Type-specific arithmetic operators for different bit widths
proc add8*(a, b: int8x16_t): int8x16_t = vaddq_s8(a, b)
proc sub8*(a, b: int8x16_t): int8x16_t = vsubq_s8(a, b)
proc mul8*(a, b: int8x16_t): int8x16_t = vmulq_s8(a, b)

proc add16*(a, b: int16x8_t): int16x8_t = vaddq_s16(a, b)
proc sub16*(a, b: int16x8_t): int16x8_t = vsubq_s16(a, b)
proc mul16*(a, b: int16x8_t): int16x8_t = vmulq_s16(a, b)

proc add32*(a, b: int32x4_t): int32x4_t = vaddq_s32(a, b)
proc sub32*(a, b: int32x4_t): int32x4_t = vsubq_s32(a, b)
proc mul32*(a, b: int32x4_t): int32x4_t = vmulq_s32(a, b)

proc add64*(a, b: int64x2_t): int64x2_t = vaddq_s64(a, b)
proc sub64*(a, b: int64x2_t): int64x2_t = vsubq_s64(a, b)

# Shift operators
# Note: NEON shift intrinsics require compile-time constants
template `shl`*(a: int32x4_t, count: static int32): int32x4_t = vshlq_n_s32(a, count)  
template `shr`*(a: int32x4_t, count: static int32): int32x4_t = vshrq_n_s32(a, count)

# Utility constructors for 128-bit vectors (most common)
proc vec8*(val: int8): int8x16_t = vdupq_n_s8(val)
proc vec16*(val: int16): int16x8_t = vdupq_n_s16(val)
proc vec32*(val: int32): int32x4_t = vdupq_n_s32(val)
proc vec64*(val: int64): int64x2_t = vdupq_n_s64(val)
proc vecf*(val: float32): float32x4_t = vdupq_n_f32(val)
proc vecd*(val: float64): float64x2_t = vdupq_n_f64(val)

# Utility constructors for 64-bit vectors
proc vec8_64*(val: int8): int8x8_t = vdup_n_s8(val)
proc vec16_64*(val: int16): int16x4_t = vdup_n_s16(val)
proc vec32_64*(val: int32): int32x2_t = vdup_n_s32(val)
proc vec64_64*(val: int64): int64x1_t = vdup_n_s64(val)
proc vecf_64*(val: float32): float32x2_t = vdup_n_f32(val)
proc vecd_64*(val: float64): float64x1_t = vdup_n_f64(val)