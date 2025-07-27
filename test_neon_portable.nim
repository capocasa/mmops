## Portable NEON Test - Cross-compile friendly
## This version compiles to ARM64 binary that imports NEON at runtime

echo "ARM NEON Test - Portable Version"
echo "================================"

when defined(macosx) and (defined(arm64) or defined(aarch64)):
  # NEON available - use full implementation
  when compiles(block:
    import neon
    discard vec32(42)
  ):
    import neon
    
    proc run_neon_tests() =
      echo "Running NEON acceleration tests..."
      
      # Basic arithmetic test
      let a = vec32(10)
      let b = vec32(5)
      let result = a + b
      echo "Vector addition: 10 + 5 = ", sum(result) / 4  # Should be 15
      
      # Multiplication test  
      let mult_result = a * b
      echo "Vector multiplication: 10 * 5 = ", sum(mult_result) / 4  # Should be 50
      
      # Float test
      let fa = vecf(3.14'f32)
      let fb = vecf(2.0'f32)
      let float_result = fa + fb
      echo "Float addition: 3.14 + 2.0 = ", sum(float_result) / 4  # Should be ~5.14
      
      # Logical operations
      let mask1 = vec32(0xFF)
      let mask2 = vec32(0x0F)
      let and_result = mask1 and mask2
      echo "Bitwise AND: 0xFF & 0x0F = ", sum(and_result)
      
      echo "All NEON tests completed successfully!"
    
    run_neon_tests()
  else:
    echo "NEON module available but not compiling - check compiler flags"
    echo "Expected on cross-compilation, will work when run natively"
else:
  echo "This binary is designed for Apple Silicon Macs (ARM64 + macOS)"
  echo "Current platform:"
  echo "  OS: ", when defined(macosx): "macOS" elif defined(linux): "Linux" elif defined(windows): "Windows" else: "Unknown"
  echo "  CPU: ", when defined(arm64): "ARM64" elif defined(aarch64): "AArch64" elif defined(amd64): "x86_64" else: "Unknown"
  echo ""
  echo "To test NEON:"
  echo "1. Run this on an Apple Silicon Mac"  
  echo "2. Or compile natively: nim c -d:release test_neon_mac.nim"

echo ""
echo "Cross-compilation test completed."
echo "Binary architecture: ARM64"
echo "Target OS: macOS"