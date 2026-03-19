#!/bin/bash
# Usage: ./build_image.sh

echo "===================================="
echo "Building bfcl image"
echo "===================================="

docker buildx build --platform=linux/amd64 --file Dockerfile.oss --load -t bfclv4 . && git_hash=$(git rev-parse --short=6 HEAD)
beaker image delete shashankg/bfclv4-latest
beaker image create bfclv4 -n bfclv4-latest -w ai2/general-tool-use --description "BFCLv4 image corresponding to git hash ${git_hash}"
beaker image create bfclv4 -n bfclv4-${git_hash} -w ai2/general-tool-use
