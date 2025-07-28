import unittest
import sequtils
import sugar
import ../src/simdops

# --- Helper procs for testing ---

proc toSeq*(v: M256): array[8, float32] =
  storeu(result[0].addr, v)

proc toSeq*(v: M256d): array[4, float64] =
  storeu(result[0].addr, v)

proc toSeq*(v: M256i): array[8, int32] =
  storeu(result[0].addr, v)

proc checkVec*(a, b: M256; tol = 1e-6) =
  let arrA = toSeq(a)
  let arrB = toSeq(b)
  for i in 0..7:
    check abs(arrA[i] - arrB[i]) < tol

proc checkVec*(a, b: M256d; tol = 1e-9) =
  let arrA = toSeq(a)
  let arrB = toSeq(b)
  for i in 0..3:
    check abs(arrA[i] - arrB[i]) < tol

proc checkVec*(a, b: M256i) =
  check toSeq(a) == toSeq(b)

suite "N82 SIMD Library Tests":

  suite "Vector Initialization":
    test "splat":
      checkVec(splat(5.0'f32), set(5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0))
      checkVec(splat(10.0'f64), set(10.0, 10.0, 10.0, 10.0))
      checkVec(splat(7'i32), set(7, 7, 7, 7, 7, 7, 7, 7))

    test "setzero":
      checkVec(setzero(M256), splat(0.0'f32))
      checkVec(setzero(M256d), splat(0.0'f64))
      checkVec(setzero(M256i), splat(0'i32))

  suite "Memory Operations":
    test "load and store":
      var seq_f32: seq[float32] = @[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
      let v_f32 = loadu(seq_f32[0].addr, M256)
      var stored_f32: array[8, float32]
      storeu(stored_f32[0].addr, v_f32)
      check stored_f32 == toSeq(v_f32)

  suite "Arithmetic Operations":
    let a_f32 = set(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    let b_f32 = set(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)
    
    test "addition":
      let expected = splat(9.0'f32)
      checkVec(a_f32 + b_f32, expected)

    test "subtraction":
      let expected = set(-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0)
      checkVec(a_f32 - b_f32, expected)

    test "multiplication":
      let expected = set(8.0, 14.0, 18.0, 20.0, 20.0, 18.0, 14.0, 8.0)
      checkVec(a_f32 * b_f32, expected)

    test "division":
      let a = set(8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0)
      let b = splat(4.0'f32)
      let expected = set(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
      checkVec(a / b, expected)

    test "fmadd":
      let c_f32 = splat(1.0'f32)
      let expected = set(9.0, 15.0, 19.0, 21.0, 21.0, 19.0, 15.0, 9.0)
      checkVec(fmadd(a_f32, b_f32, c_f32), expected)

  suite "Comparison Operators":
    let a = set(1, 2, 3, 4, 5, 6, 7, 8)
    let b = set(8, 7, 6, 5, 4, 3, 2, 1)
    
    test "equality":
      let v_eq = set(1, 2, 3, 4, 4, 3, 2, 1)
      let v_eq2 = set(1, 2, 3, 4, 4, 3, 2, 1)
      let r = toSeq(v_eq == v_eq2)
      check r == [0xFFFFFFFF'u32, 0xFFFFFFFF'u32, 0xFFFFFFFF'u32, 0xFFFFFFFF'u32, 0xFFFFFFFF'u32, 0xFFFFFFFF'u32, 0xFFFFFFFF'u32, 0xFFFFFFFF'u32].map(it => cast[int32](it))

    test "less than":
       let r = toSeq(a < b)
       let expected: array[8, int32] = [0, 0, 0, 0, -1, -1, -1, -1]
       check r == expected

  suite "Math Functions":
    test "sqrt":
      let v = set(4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0)
      let expected = set(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
      checkVec(sqrt(v), expected)

    test "floor":
      let v = set(1.1, 2.9, 3.5, 4.0, 5.8, 6.2, 7.7, 8.4)
      let expected = set(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
      checkVec(floor(v), expected)

  suite "Horizontal Reductions":
    let v_f32 = set(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    let v_f64 = set(10.0, 20.0, 30.0, 40.0)

    test "hsum":
      check abs(hsum(v_f32) - 36.0) < 1e-6
      check abs(hsum(v_f64) - 100.0) < 1e-9

    test "hmin/hmax":
      check abs(hmin(v_f32) - 1.0) < 1e-6
      # check abs(hmax(v_f32) - 8.0) < 1e-6

    test "movemask":
      let v = set(-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0)
      check movemask(v) == 0b10101010

  suite "Data Manipulation":
    let a = set(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    let b = set(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)

    # test "unpackLo/unpackHi":
    #   let a = set(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    #   let b = set(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)
    #   let lo = unpackLo(a, b)
    #   let hi = unpackHi(a, b)
    #   checkVec(lo, set(1.0, 10.0, 2.0, 20.0, 5.0, 50.0, 6.0, 60.0))
    #   checkVec(hi, set(3.0, 30.0, 4.0, 40.0, 7.0, 70.0, 8.0, 80.0))
      
  suite "Type Conversions":
    test "convert value":
      let i_vec = set(1, 2, 3, 4, 5, 6, 7, 8)
      let f_vec = convert(i_vec)
      checkVec(f_vec, set(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0))
      let i_vec_back = convert(f_vec)
      checkVec(i_vec_back, i_vec)

    test "bitcast reinterpret":
      let i_vec = set(0x40000000, 0x40400000, 0x40800000, 0x40A00000, 0x40C00000, 0x40E00000, 0x41000000, 0x41100000) # Represents 2.0, 3.0, ...
      let f_vec = bitcast(i_vec, M256)
      let i_vec_roundtrip = bitcast(f_vec, M256i)
      checkVec(i_vec, i_vec_roundtrip)
      
      let f_vec_expected = set(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
      checkVec(f_vec, f_vec_expected)
