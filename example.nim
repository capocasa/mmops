import avx2

echo "AVX2 Nim Wrapper Example"
echo "========================"

# Basic arithmetic with operator overloads
let a = vec32(10)
let b = vec32(5)

echo "Vector arithmetic:"
echo "a + b = ", sum(a + b)  # 120 (8 * 15)
echo "a - b = ", sum(a - b)  # 40 (8 * 5)  
echo "a * b = ", sum(a * b)  # 400 (8 * 50)

echo "\nLogical operations:"
let mask1 = vec32(0xFF00)
let mask2 = vec32(0x00FF)

echo "mask1 and mask2 = ", sum(mask1 and mask2)
echo "mask1 or mask2 = ", sum(mask1 or mask2)
echo "mask1 xor mask2 = ", sum(mask1 xor mask2)

echo "\nHorizontal operations (sequtils-style):"
let data = vec32(42)
echo "sum = ", sum(data)      # 336 (8 * 42)
echo "avg = ", avg(data)      # 42.0
echo "min = ", min(data)      # 42
echo "max = ", max(data)      # 42

echo "\nDifferent bit widths:"
let data8 = vec8(100)
let data16 = vec16(1000)  
let data64 = vec64(1000000)

echo "8-bit sum = ", sum(add8(data8, data8))   # 32 elements * 200
echo "16-bit sum = ", sum(add16(data16, data16)) # 16 elements * 2000
echo "64-bit sum = ", sum(add64(data64, data64)) # 4 elements * 2000000

echo "\nShift operations:"
let shift_data = vec32(16)
echo "left shift by 2 = ", sum(shift_data shl 2)   # 8 * 64 = 512
echo "right shift by 1 = ", sum(shift_data shr 1)  # 8 * 8 = 64

echo "\nComparison operations:"
let x = vec32(10)
let y = vec32(5)
echo "x == x: ", sum(x == x)  # All -1s (true)
echo "x > y: ", sum(x > y)    # All -1s (true)
echo "x < y: ", sum(x < y)    # All 0s (false)