#!/usr/bin/env bash

# Stop and remove potentially orphaned containers from previous runs
echo "Stopping potentially orphaned mech containers..."
docker stop $(docker ps -q --filter "name=mech.*_abci_0") 2>/dev/null || true
docker stop $(docker ps -q --filter "name=mech.*_tm_0") 2>/dev/null || true
echo "Removing potentially orphaned mech containers..."
docker rm $(docker ps -aq --filter "name=mech.*_abci_0") 2>/dev/null || true
docker rm $(docker ps -aq --filter "name=mech.*_tm_0") 2>/dev/null || true

# Remove previous mech directory (contains build artifacts, not running containers)
echo "Removing previous mech directory..."
rm -rf mech

# Load env vars
set -o allexport; source .env; set +o allexport

# Remove previous builds
# if [ -d "mech" ]; then
#     echo $PASSWORD | sudo -S sudo rm -Rf mech;
# fi

# Push packages and fetch service
# make formatters
# make generators
make clean

autonomy push-all

autonomy fetch --local --service valory/mech && cd mech

# Clean up previous docker-compose environment if it exists
previous_build_dir=$(find . -maxdepth 1 -type d -name 'abci_build_*' -print -quit)
if [ -n "$previous_build_dir" ]; then
    echo "Found previous build directory: $previous_build_dir. Cleaning up..."
    (cd "$previous_build_dir" && docker-compose down --remove-orphans) || echo "docker-compose down failed, continuing..."
    echo "Removing previous build directory: $previous_build_dir"
    rm -rf "$previous_build_dir"
fi

# Build the image
autonomy build-image

# Copy keys and build the deployment
cp $PWD/../keys.json .

autonomy deploy build -ltm --n ${NUM_AGENTS:-4}

# Find the build directory (assumes only one matching directory exists)
build_dir=$(find . -maxdepth 1 -type d -name 'abci_build_*' -print -quit)

# Check if build_dir was found
if [ -z "$build_dir" ]; then
    echo "Failed to find build directory starting with abci_build_ in $(pwd)"
    exit 1
fi

echo "Found deployment build directory: $build_dir"

# Run the deployment using the found build directory
autonomy deploy run --detach --build-dir "$build_dir"