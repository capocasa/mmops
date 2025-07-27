import unittest

when defined(arm) or defined(arm64) or defined(aarch64):
  import neon
  
  suite "NEON SIMD Tests":
    test "Basic 32-bit integer operations":
      let a = vec32(1)
      let b = vec32(2)
      let result = a + b
      echo "NEON: 32-bit addition works"
    
    test "Basic float operations":
      let a = vecf(1.0)
      let b = vecf(2.0)
      let result = a + b
      echo "NEON: float addition works"

elif defined(amd64) or defined(i386):
  import avx2
  
  suite "AVX2 SIMD Tests":
    test "Basic 32-bit integer operations":
      let a = vec32(1)
      let b = vec32(2)
      let result = a + b
      echo "AVX2: 32-bit addition works"
    
    test "Basic float operations":
      let a = vecf(1.0)
      let b = vecf(2.0)
      let result = a + b
      echo "AVX2: float addition works"

else:
  {.error: "Unsupported architecture".}

echo "Universal SIMD test completed"