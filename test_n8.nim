import unittest
import n8

suite "N8 Universal SIMD Tests":
  echo "Testing N8 on architecture: ", N8_ARCH
  
  test "32-bit integer operations":
    let a = n8_set1_i32(10)
    let b = n8_set1_i32(5)
    
    # Test arithmetic
    let add_result = n8_add(a, b)
    let sub_result = n8_sub(a, b)
    let mul_result = n8_mul(a, b)
    
    # Test operator overloads
    let add_op = a + b
    let sub_op = a - b
    let mul_op = a * b
    
    # Test logical operations
    let and_result = n8_and(a, b)
    let or_result = n8_or(a, b)
    let xor_result = n8_xor(a, b)
    let not_result = n8_not(a)
    
    # Test horizontal operations
    let sum_val = n8_sum(a)
    let min_val = n8_min(a)
    let max_val = n8_max(a)
    
    echo "N8: 32-bit integer operations completed"
    echo "  Sum of vector with all 10s: ", sum_val
    echo "  Min of vector with all 10s: ", min_val
    echo "  Max of vector with all 10s: ", max_val
  
  test "32-bit float operations":
    let a = n8_set1_f32(3.14'f32)
    let b = n8_set1_f32(2.0'f32)
    
    # Test arithmetic
    let add_result = n8_add(a, b)
    let sub_result = n8_sub(a, b)
    let mul_result = n8_mul(a, b)
    
    # Test operator overloads
    let add_op = a + b
    let sub_op = a - b
    let mul_op = a * b
    
    # Test horizontal operations
    let sum_val = n8_sum(a)
    
    echo "N8: 32-bit float operations completed"
    echo "  Sum of vector with all 3.14: ", sum_val
  
  test "64-bit double operations":
    let a = n8_set1_f64(2.718281828)
    let b = n8_set1_f64(1.414213562)
    
    # Test arithmetic
    let add_result = n8_add(a, b)
    let sub_result = n8_sub(a, b)
    let mul_result = n8_mul(a, b)
    
    # Test operator overloads  
    let add_op = a + b
    let sub_op = a - b
    let mul_op = a * b
    
    # Test horizontal operations
    let sum_val = n8_sum(a)
    
    echo "N8: 64-bit double operations completed"
    echo "  Sum of vector with all e: ", sum_val

echo "N8 Universal SIMD test completed on ", N8_ARCH, " architecture"
echo "Vector sizes: int32=", N8_VECTOR32_SIZE, ", float32=", N8_VECTORF32_SIZE, ", float64=", N8_VECTORF64_SIZE