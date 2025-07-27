import neon

echo "ARM NEON Nim Wrapper Example"
echo "============================="

# Basic arithmetic with operator overloads
let a = vec32(10)
let b = vec32(5)

echo "Vector arithmetic (128-bit vectors):"
echo "a + b = ", sum(a + b)  # 60 (4 * 15)
echo "a - b = ", sum(a - b)  # 20 (4 * 5)  
echo "a * b = ", sum(a * b)  # 200 (4 * 50)

echo "\nLogical operations:"
let mask1 = vec32(0xFF00)
let mask2 = vec32(0x00FF)

echo "mask1 and mask2 = ", sum(mask1 and mask2)
echo "mask1 or mask2 = ", sum(mask1 or mask2)
echo "mask1 xor mask2 = ", sum(mask1 xor mask2)

echo "\nHorizontal operations (sequtils-style):"
let data = vec32(42)
echo "sum = ", sum(data)      # 168 (4 * 42)
echo "avg = ", avg(data)      # 42.0
echo "min = ", min(data)      # 42
echo "max = ", max(data)      # 42

echo "\nDifferent bit widths:"
let data8 = vec8(100)
let data16 = vec16(1000)  
let data64 = vec64(1000000)

echo "8-bit sum = ", sum(add8(data8, data8))   # 16 elements * 200
echo "16-bit sum = ", sum(add16(data16, data16)) # 8 elements * 2000
echo "64-bit sum = ", sum(add64(data64, data64)) # 2 elements * 2000000

echo "\nShift operations:"
let shift_data = vec32(16)
echo "left shift by 2 = ", sum(shift_data shl 2)   # 4 * 64 = 256
echo "right shift by 1 = ", sum(shift_data shr 1)  # 4 * 8 = 32

echo "\nFloat operations:"
let fa = vecf(3.14'f32)
let fb = vecf(2.0'f32)
echo "float add = ", sum(fa + fb)     # 4 * 5.14 ≈ 20.56
echo "float mul = ", sum(fa * fb)     # 4 * 6.28 ≈ 25.12
echo "float div = ", sum(fa / fb)     # 4 * 1.57 ≈ 6.28

echo "\nDouble precision:"
let da = vecd(3.14159)
let db = vecd(2.71828)
echo "double add = ", sum(da + db)    # 2 * 5.85987 ≈ 11.72
echo "double mul = ", sum(da * db)    # 2 * 8.539734 ≈ 17.08

echo "\n64-bit vectors (smaller, but faster on some operations):"
let small_a = vec32_64(100)
let small_b = vec32_64(200)
echo "64-bit vector sum = ", sum(add32(vdupq_n_s32(vget_lane_s32(small_a, 0)), 
                                      vdupq_n_s32(vget_lane_s32(small_b, 0))))  # 2 * 300 = 1200