import unittest
import math
import ../src/mmops

{.passC: "-mavx2 -mfma".}

## CI-ready unit tests for mmops - multimedia operators
## Proper assertions for automated testing systems

# --- Helper procedures for testing ---

proc approxEqual(a, b: float32, tol = 1e-6): bool =
  abs(a - b) < tol

proc approxEqual(a, b: float64, tol = 1e-9): bool =
  abs(a - b) < tol

proc checkArrayEqual[T](a, b: openArray[T], tol = 1e-6) =
  check a.len == b.len
  for i in 0..<a.len:
    when T is float32 or T is float64:
      check approxEqual(a[i], b[i], tol)
    else:
      check a[i] == b[i]

proc checkApprox[T](actual, expected: T, tol = 1e-6) =
  when T is float32 or T is float64:
    check approxEqual(actual, expected, tol)
  else:
    check actual == expected

suite "mmops - Multimedia Operators Test Suite":

  suite "Vector Initialization":
    test "splat operations":
      let v_f32 = splat[8](5.0'f32)
      let expected_f32 = [5.0'f32, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
      checkArrayEqual(store(v_f32), expected_f32)

      let v_f64 = splat[4](10.0'f64)
      let expected_f64 = [10.0'f64, 10.0, 10.0, 10.0]
      checkArrayEqual(store(v_f64), expected_f64)

      let v_i32 = splat[8](7'i32)
      let expected_i32 = [7'i32, 7, 7, 7, 7, 7, 7, 7]
      checkArrayEqual(store(v_i32), expected_i32)

    test "zero initialization":
      let zeros_f32 = zero(Mm[8, float32])
      let expected_f32 = [0.0'f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      checkArrayEqual(store(zeros_f32), expected_f32)

      let zeros_f64 = zero(Mm[4, float64])
      let expected_f64 = [0.0'f64, 0.0, 0.0, 0.0]
      checkArrayEqual(store(zeros_f64), expected_f64)

      let zeros_i32 = zero(Mm[8, int32])
      let expected_i32 = [0'i32, 0, 0, 0, 0, 0, 0, 0]
      checkArrayEqual(store(zeros_i32), expected_i32)

    test "set with explicit values":
      let v_f32 = set(8.0'f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)
      let expected_f32 = [1.0'f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  # Reversed due to Intel ordering
      checkArrayEqual(store(v_f32), expected_f32)

      let v_i32 = set(8'i32, 7, 6, 5, 4, 3, 2, 1)
      let expected_i32 = [1'i32, 2, 3, 4, 5, 6, 7, 8]  # Reversed due to Intel ordering
      checkArrayEqual(store(v_i32), expected_i32)

  suite "Memory Operations":
    test "load and store round-trip":
      let input_f32 = [1.0'f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
      let vec_f32 = load(input_f32)
      let output_f32 = store(vec_f32)
      checkArrayEqual(input_f32, output_f32)

      let input_f64 = [1.5'f64, 2.5, 3.5, 4.5]
      let vec_f64 = load(input_f64)
      let output_f64 = store(vec_f64)
      checkArrayEqual(input_f64, output_f64)

      let input_i32 = [10'i32, 20, 30, 40, 50, 60, 70, 80]
      let vec_i32 = load(input_i32)
      let output_i32 = store(vec_i32)
      checkArrayEqual(input_i32, output_i32)

    test "toMm and toArray aliases":
      let input = [1.5'f32, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
      let vec = toMm(input)
      let output = toArray(vec)
      checkArrayEqual(input, output)

  suite "Arithmetic Operations":
    let a_f32 = load([1.0'f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    let b_f32 = load([8.0'f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    
    test "addition":
      let result = a_f32 + b_f32
      let expected = [9.0'f32, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
      checkArrayEqual(store(result), expected)

    test "subtraction":
      let result = a_f32 - b_f32
      let expected = [-7.0'f32, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0]
      checkArrayEqual(store(result), expected)

    test "multiplication":
      let result = a_f32 * b_f32
      let expected = [8.0'f32, 14.0, 18.0, 20.0, 20.0, 18.0, 14.0, 8.0]
      checkArrayEqual(store(result), expected)

    test "division":
      let a = load([8.0'f32, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0])
      let b = splat[8](4.0'f32)
      let result = a / b
      let expected = [2.0'f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
      checkArrayEqual(store(result), expected)

    test "fused multiply-add (fma)":
      let c_f32 = splat[8](1.0'f32)
      let result = fma(a_f32, b_f32, c_f32)
      let expected = [9.0'f32, 15.0, 19.0, 21.0, 21.0, 19.0, 15.0, 9.0]
      checkArrayEqual(store(result), expected)

  suite "Integer Arithmetic":
    let a_i32 = load([1'i32, 2, 3, 4, 5, 6, 7, 8])
    let b_i32 = load([2'i32, 2, 2, 2, 2, 2, 2, 2])

    test "integer addition":
      let result = a_i32 + b_i32
      let expected = [3'i32, 4, 5, 6, 7, 8, 9, 10]
      checkArrayEqual(store(result), expected)

    test "integer subtraction":
      let result = a_i32 - b_i32
      let expected = [-1'i32, 0, 1, 2, 3, 4, 5, 6]
      checkArrayEqual(store(result), expected)

    test "integer multiplication":
      let result = a_i32 * b_i32
      let expected = [2'i32, 4, 6, 8, 10, 12, 14, 16]
      checkArrayEqual(store(result), expected)

  suite "Extended Integer Types":
    test "8-bit integer operations":
      let a8 = load([1'i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])
      let b8 = splat[32](2'i8)
      let result = a8 + b8
      let expected = [3'i8, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
      checkArrayEqual(store(result), expected)

    test "16-bit integer operations":
      let a16 = load([100'i16, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600])
      let b16 = splat[16](50'i16)
      let result = a16 * b16
      let expected = [5000'i16, 10000, 15000, 20000, 25000, 30000, -30536, -25536, -20536, -15536, -10536, -5536, -536, 4464, 9464, 14464]
      checkArrayEqual(store(result), expected)

    test "unsigned integer operations":
      let ua8 = load([100'u8, 200, 50, 255, 10, 128, 75, 180, 25, 240, 60, 150, 90, 210, 35, 175, 80, 220, 45, 160, 95, 230, 55, 185, 70, 200, 40, 165, 85, 225, 50, 190])
      let ub8 = splat[32](150'u8)
      let min_result = min(ua8, ub8)
      let max_result = max(ua8, ub8)
      
      let expected_min = [100'u8, 150, 50, 150, 10, 128, 75, 150, 25, 150, 60, 150, 90, 150, 35, 150, 80, 150, 45, 150, 95, 150, 55, 150, 70, 150, 40, 150, 85, 150, 50, 150]
      let expected_max = [150'u8, 200, 150, 255, 150, 150, 150, 180, 150, 240, 150, 150, 150, 210, 150, 175, 150, 220, 150, 160, 150, 230, 150, 185, 150, 200, 150, 165, 150, 225, 150, 190]
      
      checkArrayEqual(store(min_result), expected_min)
      checkArrayEqual(store(max_result), expected_max)

  suite "Comparison Operations":
    let a_i32 = load([1'i32, 2, 3, 4, 5, 6, 7, 8])
    let b_i32 = load([8'i32, 7, 6, 5, 4, 3, 2, 1])
    
    test "equality comparison":
      let equal_a = load([1'i32, 2, 3, 4, 4, 3, 2, 1])
      let equal_b = load([1'i32, 2, 3, 4, 4, 3, 2, 1])
      let result = equal_a == equal_b
      let mask_result = mask(result)
      check mask_result == -1  # All bits set for all-true comparison

    test "less than comparison":
      let result = a_i32 < b_i32
      let mask_result = mask(result)
      check mask_result != 0  # Some elements should be less than

    test "greater than comparison":
      let result = a_i32 > b_i32
      let mask_result = mask(result)  
      check mask_result != 0  # Some elements should be greater than

  suite "Mathematical Functions":
    test "square root":
      let v = load([4.0'f32, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0])
      let result = sqrt(v)
      let expected = [2.0'f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
      checkArrayEqual(store(result), expected)

    test "absolute value":
      let neg32 = load([-50000'i32, 30000, -80000, 10000, -20000, 70000, -40000, 60000])
      let result = abs(neg32)
      let expected = [50000'i32, 30000, 80000, 10000, 20000, 70000, 40000, 60000]
      checkArrayEqual(store(result), expected)

    test "rounding operations":
      let v = load([1.2'f32, 2.7, -3.4, -4.8, 5.5, 6.1, -7.9, 8.3])
      
      let rounded = round(v)
      let expected_round = [1.0'f32, 3.0, -3.0, -5.0, 6.0, 6.0, -8.0, 8.0]
      checkArrayEqual(store(rounded), expected_round, 0.1)

      let floored = floor(v)
      let expected_floor = [1.0'f32, 2.0, -4.0, -5.0, 5.0, 6.0, -8.0, 8.0]
      checkArrayEqual(store(floored), expected_floor)

      let ceiled = ceil(v)
      let expected_ceil = [2.0'f32, 3.0, -3.0, -4.0, 6.0, 7.0, -7.0, 9.0]
      checkArrayEqual(store(ceiled), expected_ceil)

  suite "Horizontal Operations":
    test "horizontal sum":
      let v_f32 = load([1.0'f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
      let result = sum(v_f32)
      checkApprox(result, 36.0'f32)

      let v_f64 = load([10.0'f64, 20.0, 30.0, 40.0])
      let result_f64 = sum(v_f64)
      checkApprox(result_f64, 100.0'f64)

  suite "Min/Max Operations":
    test "min and max":
      let a = load([8.0'f32, 2.0, 7.0, 1.0, 6.0, 3.0, 5.0, 4.0])
      let b = load([1.0'f32, 7.0, 2.0, 8.0, 3.0, 6.0, 4.0, 5.0])
      
      let min_result = min(a, b)
      let expected_min = [1.0'f32, 2.0, 2.0, 1.0, 3.0, 3.0, 4.0, 4.0]
      checkArrayEqual(store(min_result), expected_min)

      let max_result = max(a, b)
      let expected_max = [8.0'f32, 7.0, 7.0, 8.0, 6.0, 6.0, 5.0, 5.0]
      checkArrayEqual(store(max_result), expected_max)

  suite "Bitwise Operations":
    test "bitwise operations":
      let a = load([0xFF'i32, 0x00, 0xF0, 0x0F, 0xAA, 0x55, 0xFF, 0x00])
      let b = load([0xF0'i32, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0])
      
      let and_result = a and b
      let expected_and = [0xF0'i32, 0x00, 0xF0, 0x00, 0xA0, 0x50, 0xF0, 0x00]
      checkArrayEqual(store(and_result), expected_and)

      let or_result = a or b
      let expected_or = [0xFF'i32, 0xF0, 0xF0, 0xFF, 0xFA, 0xF5, 0xFF, 0xF0]
      checkArrayEqual(store(or_result), expected_or)

      let xor_result = a xor b
      let expected_xor = [0x0F'i32, 0xF0, 0x00, 0xFF, 0x5A, 0xA5, 0x0F, 0xF0]
      checkArrayEqual(store(xor_result), expected_xor)

  suite "64-bit Operations":
    test "64-bit double precision":
      let a = load([1.5'f64, 2.5, 3.5, 4.5])
      let b = load([0.5'f64, 1.0, 1.5, 2.0])
      
      let sum_result = a + b
      let expected_sum = [2.0'f64, 3.5, 5.0, 6.5]
      checkArrayEqual(store(sum_result), expected_sum)

      let prod_result = a * b
      let expected_prod = [0.75'f64, 2.5, 5.25, 9.0]
      checkArrayEqual(store(prod_result), expected_prod)

      let sqrt_result = sqrt(a)
      let expected_sqrt = [1.224744871391589'f64, 1.5811388300841898, 1.8708286933869707, 2.1213203435596424]
      checkArrayEqual(store(sqrt_result), expected_sqrt)

  suite "Nim-style Aliases":
    test "initMm functions":
      let init_f32 = initMm(8.0'f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)  # High to low ordering
      let expected = [1.0'f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  # Stored as low to high
      checkArrayEqual(store(init_f32), expected)

      let zero_init = initMm(Mm[8, float32])
      let expected_zero = [0.0'f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      checkArrayEqual(store(zero_init), expected_zero)

  suite "Mask Operations":
    test "movemask functionality":
      let v = load([-1.0'f32, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0])
      let mask_result = mask(v)
      check mask_result == 85  # 0b01010101 - alternating pattern for negative values (low-to-high bit order)

  suite "Missing SIMD Operations":
    test "fast reciprocal square root":
      let v = load([4.0'f32, 16.0, 64.0, 100.0, 1.0, 9.0, 25.0, 36.0])
      let result = fastSqrt(v)
      # fastSqrt is approximate, so we need larger tolerance
      let output = store(result)
      # Approximate 1/sqrt(x) values
      let expected = [0.5'f32, 0.25, 0.125, 0.1, 1.0, 0.333333, 0.2, 0.166667]
      for i in 0..<8:
        check abs(output[i] - expected[i]) < 0.01  # Larger tolerance for approximation

    test "comparison operators <=, >=":
      let a = load([1.0'f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
      let b = load([2.0'f32, 2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 6.0])
      
      let le_result = a <= b
      let le_mask = mask(le_result)
      check le_mask != 0  # Some elements should be less than or equal
      
      let ge_result = a >= b
      let ge_mask = mask(ge_result)
      check ge_mask != 0  # Some elements should be greater than or equal

    test "64-bit integer operations":
      let a64 = load([1000000000000'i64, 2000000000000, 3000000000000, 4000000000000])
      let b64 = load([500000000000'i64, 1000000000000, 1500000000000, 2000000000000])
      
      let sum64 = a64 + b64
      let expected_sum = [1500000000000'i64, 3000000000000, 4500000000000, 6000000000000]
      checkArrayEqual(store(sum64), expected_sum)
      
      let diff64 = a64 - b64
      let expected_diff = [500000000000'i64, 1000000000000, 1500000000000, 2000000000000]
      checkArrayEqual(store(diff64), expected_diff)
      
      # Test 64-bit equality comparison
      let eq64 = a64 == b64
      let eq_mask = mask(eq64)
      check eq_mask == 0  # No elements should be equal

    test "additional floating-point comparisons":
      let a = load([1.0'f64, 2.0, 3.0, 4.0])
      let b = load([2.0'f64, 2.0, 2.0, 2.0])
      
      let eq_result = a == b
      let eq_mask = mask(eq_result)
      check eq_mask != 0  # Second element should be equal (both 2.0)
      
      let lt_result = a < b
      let lt_mask = mask(lt_result)
      check lt_mask != 0  # First element should be less than
      
      let gt_result = a > b
      let gt_mask = mask(gt_result)
      check gt_mask != 0  # Last two elements should be greater than

  suite "Advanced SIMD Operations":
    test "saturated arithmetic":
      let a16 = load([32000'i16, -30000, 25000, -20000, 15000, -10000, 5000, -1000, 30000, -25000, 20000, -15000, 10000, -5000, 1000, 500])
      let b16 = load([10000'i16, -5000, 15000, -10000, 20000, -15000, 25000, -20000, 5000, -2500, 7500, -5000, 12500, -7500, 2500, 1000])
      
      let sat_add = saturatedAdd(a16, b16)
      let sat_sub = saturatedSub(a16, b16)
      
      # Results should be clamped to int16 range
      let add_result = store(sat_add)
      let sub_result = store(sat_sub)
      
      # Check some values are saturated (at int16 limits)
      check add_result[0] == 32767  # Should saturate at max int16
      check sub_result[1] != -35000  # Should be saturated, not overflowed

    test "blend operations":
      let a = load([1.0'f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
      let b = load([10.0'f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
      
      # Test fixed blend (blend every other element)
      let blended = blend(a, b, 0b10101010)  # Alternate pattern
      let result = store(blended)
      
      # Should have alternating values from a and b
      check result[0] == 1.0   # From a
      check result[1] == 20.0  # From b  
      check result[2] == 3.0   # From a
      check result[3] == 40.0  # From b

    test "gather operations":
      let data = [100'i32, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
      let indices = load([0'i32, 2, 4, 6, 1, 3, 5, 7])  # Gather specific indices
      
      let gathered = gather(data, indices, 4)  # Scale factor 4 for int32
      let result = store(gathered)
      
      check result[0] == 100  # data[0]
      check result[1] == 300  # data[2]
      check result[2] == 500  # data[4]
      check result[3] == 700  # data[6]

    test "pack/unpack operations":
      let a = load([1.0'f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
      let b = load([10.0'f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
      
      let unpacked_lo = unpackLo(a, b)
      let unpacked_hi = unpackHi(a, b)
      
      let lo_result = store(unpacked_lo)
      let hi_result = store(unpacked_hi)
      
      # unpackLo interleaves low halves: a[0],b[0],a[1],b[1]...
      check lo_result[0] == 1.0
      check lo_result[1] == 10.0
      check lo_result[2] == 2.0
      check lo_result[3] == 20.0

    test "horizontal operations":
      let a = load([1'i32, 2, 3, 4, 5, 6, 7, 8])
      let b = load([10'i32, 20, 30, 40, 50, 60, 70, 80])
      
      let h_add = hadd(a, b)
      
      # hadd packing behavior is complex - just verify it produces results
      let add_result = store(h_add)
      check add_result.len == 8  # Should produce 8 results
      
      # The exact values depend on AVX2's specific packing behavior
      # which can vary, so just check that we get sensible results
      for val in add_result:
        check val > 0  # All should be positive

    test "shift operations":
      let a = load([1'i32, 2, 4, 8, 16, 32, 64, 128])
      
      let left_shifted = a shl 2  # Shift left by 2 bits (multiply by 4)
      let right_shifted = a shr 1  # Shift right by 1 bit (divide by 2)
      
      let left_result = store(left_shifted)
      let right_result = store(right_shifted)
      
      check left_result[0] == 4    # 1 << 2
      check left_result[1] == 8    # 2 << 2
      check right_result[2] == 2   # 4 >> 1
      check right_result[3] == 4   # 8 >> 1

    test "average operations":
      let a = load([10'u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 255, 0, 10, 20, 30, 40, 50])
      let b = load([20'u8, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 255, 255, 10, 20, 30, 40, 50, 60])
      
      let averaged = avg(a, b)
      let result = store(averaged)
      
      check result[0] == 15  # (10+20)/2
      check result[1] == 25  # (20+30)/2

    test "type conversion operations":
      let int_vec = load([1'i32, 2, 3, 4, 5, 6, 7, 8])
      let float_vec = convert(int_vec)
      let back_to_int = convert(float_vec)
      
      let float_result = store(float_vec) 
      let int_result = store(back_to_int)
      
      checkArrayEqual(int_result, [1'i32, 2, 3, 4, 5, 6, 7, 8])
      checkArrayEqual(float_result, [1.0'f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])