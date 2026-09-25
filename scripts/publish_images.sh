#!/bin/sh
# Build and push the Synthetic Hospital images to GitHub Container Registry (maintainers).
#
#   export GITHUB_TOKEN=<token with write:packages>
#   echo "$GITHUB_TOKEN" | docker login ghcr.io -u <github-user> --password-stdin
#   scripts/publish_images.sh [version]        # default 1.3
#
# Afterwards set each package to public in the repository's Packages settings.
set -eu
REG="${SH_REGISTRY:-ghcr.io/sparkcpark/synthetic-hospital}"
VER="${1:-1.3}"
cd "$(dirname "$0")/.."
docker build --target app       -t "$REG:$VER"           .
docker build --target with-data -t "$REG:$VER-data"      .
docker build --target allinone  -t "$REG:$VER-allinone"  .
for tag in "$VER" "$VER-data" "$VER-allinone"; do
    docker push "$REG:$tag"
done
echo "pushed $REG:{$VER,$VER-data,$VER-allinone}"
