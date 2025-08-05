import unittest
import ../src/mmops

{.passC: "-mavx2 -mfma".}

## Tests for comparison operations (signed integers only - unsigned not supported)

proc checkArrayEqual[T](a, b: openArray[T]) =
  check a.len == b.len
  for i in 0..<a.len:
    check a[i] == b[i]

suite "Comparison Operations":
  test "signed 32-bit integer comparisons":
    let a_i32 = load([1'i32, 2, 3, 4, 5, 6, 7, 8])
    let b_i32 = load([8'i32, 7, 6, 5, 4, 3, 2, 1])
    let lt_result = a_i32 < b_i32
    let gt_result = a_i32 > b_i32
    let lt_mask = mask(lt_result)
    let gt_mask = mask(gt_result)
    check lt_mask != 0  # Some elements should be less than
    check gt_mask != 0  # Some elements should be greater than

    # Test with negative values
    let c_i32 = load([-100'i32, -50, -10, -1, 1, 10, 50, 100])
    let d_i32 = load([100'i32, 50, 10, 1, -1, -10, -50, -100])
    let lt_result2 = c_i32 < d_i32
    let gt_result2 = c_i32 > d_i32
    let lt_mask2 = mask(lt_result2)
    let gt_mask2 = mask(gt_result2)
    check lt_mask2 != 0  # Some should be less than
    check gt_mask2 != 0  # Some should be greater than
    
  test "signed 16-bit comparisons":
    let a_i16 = load([100'i16, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600])
    let b_i16 = load([1600'i16, 1500, 1400, 1300, 1200, 1100, 1000, 900, 800, 700, 600, 500, 400, 300, 200, 100])
    
    let lt_result = a_i16 < b_i16
    let gt_result = a_i16 > b_i16
    let lt_mask = mask(lt_result)
    let gt_mask = mask(gt_result)
    
    check lt_mask != 0  # Some should be less than
    check gt_mask != 0  # Some should be greater than
    
  test "signed 8-bit comparisons":
    let a_i8 = load([10'i8, 20, 30, 40, 50, 60, 70, 80, -80, -70, -60, -50, -40, -30, -20, -10, 
                     90, 100, 110, 120, -120, -110, -100, -90, 126, 125, 124, 123, -123, -124, -125, -126])
    let b_i8 = load([20'i8, 10, 40, 30, 60, 50, 80, 70, -70, -80, -50, -60, -30, -40, -10, -20,
                     100, 90, 120, 110, -110, -120, -90, -100, 125, 126, 123, 124, -124, -123, -126, -125])
                     
    let lt_result = a_i8 < b_i8
    let gt_result = a_i8 > b_i8
    let lt_mask = mask(lt_result)
    let gt_mask = mask(gt_result)
    
    check lt_mask != 0  # Some should be less than
    check gt_mask != 0  # Some should be greater than