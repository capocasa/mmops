version     = "0.1.0"
author      = "Carlo Capocasa"
description = "Multimedia Operators- Safe, zero-cost nim-like SIMD"
license     = "MIT"

srcDir = "src"

requires "nim >= 2.0.0"
requires "nimsimd"

task docs, "Generate docs":
  exec "nim doc -o:docs/mmops.html src/mmops.nim"

