echo "Hello from cross-compiled ARM64 binary!"
echo "This should run on Apple Silicon Macs"

when defined(macosx):
  echo "Compiled for macOS"
else:
  echo "Compiled for other OS"

when defined(arm64) or defined(aarch64):
  echo "Compiled for ARM64"
else:
  echo "Compiled for other architecture"