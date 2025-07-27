import avx2
import std/unittest

suite "AVX2 Wrapper Tests":
  
  test "Basic vector creation and arithmetic":
    let a = vec32(5)
    let b = vec32(3)
    let result = a + b
    check sum(result) == 64  # 8 * 8 = 64
    
  test "Vector multiplication":
    let a = vec32(4)
    let b = vec32(6)
    let result = a * b
    check sum(result) == 192  # 8 * 24 = 192
    
  test "Vector subtraction":
    let a = vec32(10)
    let b = vec32(3)
    let result = a - b
    check sum(result) == 56  # 8 * 7 = 56
    
  test "Logical operations":
    let a = vec32(0xFF)
    let b = vec32(0x0F)
    let and_result = a and b
    let or_result = a or b
    let xor_result = a xor b
    
    check sum(and_result) == 8 * 0x0F
    check sum(or_result) == 8 * 0xFF
    check sum(xor_result) == 8 * 0xF0
    
  test "Comparison operations":
    let a = vec32(5)
    let b = vec32(3)
    let eq_result = a == a
    let gt_result = a > b
    let lt_result = a < b
    
    # All elements should be equal to themselves
    check sum(eq_result) == -8  # All 1s in signed arithmetic
    
  test "Horizontal operations":
    let a = vec32(10)
    check sum(a) == 80
    check avg(a) == 10.0
    check min(a) == 10
    check max(a) == 10
    
  test "Different bit width operations":
    let a = vec16(100)
    let b = vec16(50)
    let result16 = add16(a, b)
    
    let c = vec8(20)
    let d = vec8(30)
    let result8 = add8(c, d)
    
    # These should compile and execute without error
    discard result16
    discard result8
    
  test "Shift operations":
    let a = vec32(8)
    let left_shifted = a shl 2
    let right_shifted = a shr 1
    
    check sum(left_shifted) == 256  # 8 * 32 = 256
    check sum(right_shifted) == 32   # 8 * 4 = 32
    
  test "Zero vector":
    let z = zero()
    check sum(z) == 0
    
when isMainModule:
  # Run the tests
  echo "Running AVX2 wrapper tests..."