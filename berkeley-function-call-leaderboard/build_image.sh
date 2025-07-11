#!/bin/bash
# Usage: ./scripts/build_image.sh [--handler <handler>]

# Parse optional --handler argument
HANDLER=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --handler)
      HANDLER="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "===================================="
echo "Building bfcl image"
echo "===================================="

# Choose Dockerfile based on handler
DOCKERFILE_ARG=""
IMAGE_PREFIX="bfcl"
if [ "$HANDLER" = "oss" ]; then
  DOCKERFILE_ARG="--file Dockerfile.oss"
  IMAGE_PREFIX="bfcl-oss"
  echo "Using Dockerfile for the OSS handler"
fi


docker buildx build --platform=linux/amd64 $DOCKERFILE_ARG --load -t bfcl . && git_hash=$(git rev-parse --short=6 HEAD)
#beaker image delete shashankg/${IMAGE_PREFIX}-latest
#beaker image create bfcl -n ${IMAGE_PREFIX}-latest -w ai2/general-tool-use
beaker image create bfcl -n ${IMAGE_PREFIX}-${git_hash} -w ai2/general-tool-use
