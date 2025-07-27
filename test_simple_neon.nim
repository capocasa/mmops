import neon

# Simple compilation test - just check syntax and basic types
proc test_compilation() =
  discard vec32(42)
  discard vecf(3.14'f32)
  discard vecd(2.71)

when isMainModule:
  echo "NEON wrapper compiles successfully!"
  test_compilation()