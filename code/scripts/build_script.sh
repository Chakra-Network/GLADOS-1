#!/bin/bash

args=("$@")
SCRIPT_NAME=${args[0]}

echo "Building $SCRIPT_NAME for multiple platforms..."

rm -rf builds/${SCRIPT_NAME}
mkdir -p builds/${SCRIPT_NAME}

echo "Building for Linux AMD64..."
GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -o builds/${SCRIPT_NAME}/${SCRIPT_NAME}_linux_amd64 ${SCRIPT_NAME}/main.go

echo "Building for Linux ARM64..."
GOOS=linux GOARCH=arm64 go build -ldflags="-s -w" -o builds/${SCRIPT_NAME}/${SCRIPT_NAME}_linux_arm64 ${SCRIPT_NAME}/main.go

echo "Building for macOS AMD64..."
GOOS=darwin GOARCH=amd64 go build -ldflags="-s -w" -o builds/${SCRIPT_NAME}/${SCRIPT_NAME}_darwin_amd64 ${SCRIPT_NAME}/main.go

echo "Building for macOS ARM64..."
GOOS=darwin GOARCH=arm64 go build -ldflags="-s -w" -o builds/${SCRIPT_NAME}/${SCRIPT_NAME}_darwin_arm64 ${SCRIPT_NAME}/main.go

echo "Building for Windows AMD64..."
GOOS=windows GOARCH=amd64 go build -ldflags="-s -w" -o builds/${SCRIPT_NAME}/${SCRIPT_NAME}_windows_amd64.exe ${SCRIPT_NAME}/main.go

chmod +x builds/${SCRIPT_NAME}/${SCRIPT_NAME}_linux_*
chmod +x builds/${SCRIPT_NAME}/${SCRIPT_NAME}_darwin_*

echo "Build complete! Binaries are in the 'builds' directory:"
ls -la builds/${SCRIPT_NAME}

echo ""
echo "Testing binaries..."
for binary in builds/${SCRIPT_NAME}/*; do
    if [[ "$binary" == *.exe ]]; then
        echo "Skipping Windows binary test on Unix system: $binary"
    else
        echo -n "Testing $binary: "
        if file "$binary" | grep -q "executable"; then
            echo "✓ Valid executable"
        else
            echo "✗ Invalid executable"
        fi
    fi
done