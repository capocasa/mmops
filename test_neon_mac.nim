## ARM NEON Test for macOS
## 
## This file is designed to be compiled natively on Apple Silicon Macs
## Compile with: nim c -d:release test_neon_mac.nim

when defined(arm64) or defined(aarch64):
  import neon
  import std/unittest
  
  suite "ARM NEON Wrapper Tests - macOS":
    
    test "Basic vector creation and arithmetic":
      let a = vec32(5)
      let b = vec32(3)
      let result = a + b
      check sum(result) == 32  # 4 * 8 = 32
      
    test "Vector multiplication":
      let a = vec32(4)
      let b = vec32(6)
      let result = a * b
      check sum(result) == 96  # 4 * 24 = 96
      
    test "Vector subtraction":
      let a = vec32(10)
      let b = vec32(3)
      let result = a - b
      check sum(result) == 28  # 4 * 7 = 28
      
    test "Logical operations":
      let a = vec32(0xFF)
      let b = vec32(0x0F)
      let and_result = a and b
      let or_result = a or b
      let xor_result = a xor b
      
      check sum(and_result) == 4 * 0x0F
      check sum(or_result) == 4 * 0xFF
      check sum(xor_result) == 4 * 0xF0
      
    test "Horizontal operations":
      let a = vec32(10)
      check sum(a) == 40   # 4 * 10
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
      
      check sum(left_shifted) == 128  # 4 * 32 = 128
      check sum(right_shifted) == 16   # 4 * 4 = 16
      
    test "Float operations":
      let fa = vecf(3.14'f32)
      let fb = vecf(2.71'f32)
      let sum_result = fa + fb
      let product = fa * fb
      
      # Check approximate results due to float precision
      check abs(sum(sum_result) - 23.4'f32) < 0.01'f32
      check abs(sum(product) - 34.0'f32) < 0.1'f32
      
    test "Double precision operations":
      let da = vecd(3.14159)
      let db = vecd(2.71828)
      let sum_result = da + db
      let quotient = da / db
      
      # Check approximate results
      check abs(sum(sum_result) - 11.71974) < 0.0001
      check abs(sum(quotient) - 2.31054) < 0.001

  when isMainModule:
    echo "Running ARM NEON wrapper tests on macOS..."
    echo "CPU Architecture: ", when defined(arm64): "ARM64" else: "Unknown"
    echo "OS: ", when defined(macosx): "macOS" else: "Unknown"

else:
  # Fallback for non-ARM platforms
  echo "This test is designed for ARM64 processors (Apple Silicon Macs)"
  echo "Current architecture is not ARM64 - skipping NEON tests"
  echo "To run this test, use an ARM64 processor and compile with:"
  echo "nim c -d:release test_neon_mac.nim"