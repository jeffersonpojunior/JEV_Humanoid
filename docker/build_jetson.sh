#!/bin/bash
# Build the CLM image for this Jetson.
#
# Usage: ./docker/build_jetson.sh                 # base: the Dockerfile default (JetPack 6.2)
#        AUTOTAG=1 ./docker/build_jetson.sh       # base picked by `autotag vllm` (other JetPacks)
#        BASE=dustynv/vllm:<tag> ./docker/build_jetson.sh
#        IMAGE=registry.local/clm-jetson:1.0 ./docker/build_jetson.sh
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${IMAGE:-clm-jetson:latest}"
if [ -z "${BASE:-}" ] && [ "${AUTOTAG:-0}" = 1 ]; then
    command -v autotag >/dev/null || { echo "autotag not found: install jetson-containers or set BASE=" >&2; exit 1; }
    BASE="$(autotag vllm)"
fi
docker info >/dev/null 2>&1 || {
    echo "cannot reach docker: add yourself to the docker group (sudo usermod -aG docker \$USER) and log in again" >&2
    exit 1
}
echo "building $IMAGE from ${BASE:-the Dockerfile default base}"
docker build --network=host -f docker/Dockerfile.jetson ${BASE:+--build-arg BASE="$BASE"} -t "$IMAGE" .
