#!/usr/bin/env bash
# PGO (Profile-Guided Optimization) build script for VecBoost
#
# PGO improves runtime performance by 5-15% through compiler feedback:
#   - Register allocation optimized by execution frequency
#   - Hot/cold code partitioning for better instruction cache usage
#   - Function reordering to reduce jump overhead
#   - Branch reordering based on actual execution paths
#
# Usage:
#   ./scripts/pgo-build.sh [features]
#
# Examples:
#   ./scripts/pgo-build.sh                    # default features
#   ./scripts/pgo-build.sh "http,cuda"        # with specific features
#
# Requirements: Rust nightly or Rust 1.75+ with profiling support

set -euo pipefail

FEATURES="${1:-http}"
PROFILE_DIR="$(pwd)/target/pgo-data"
BINARY="./target/release/vecboost"

echo "=== VecBoost PGO Build ==="
echo "Features: ${FEATURES}"
echo "Profile data dir: ${PROFILE_DIR}"
echo ""

# Clean previous profile data
rm -rf "${PROFILE_DIR}"
mkdir -p "${PROFILE_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Compile with profile generation
# ─────────────────────────────────────────────────────────────────────────────
echo "[Phase 1/3] Compiling with profile-generate..."
RUSTFLAGS="-C profile-generate=${PROFILE_DIR}" \
    cargo build --release --features "${FEATURES}"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Run representative workload to collect profile data
# ─────────────────────────────────────────────────────────────────────────────
echo "[Phase 2/3] Running workload to collect profile data..."
echo "  (Starting server briefly for profile collection)"

# Start the server in background, wait for it to initialize, then stop it.
# This collects profile data from startup, config loading, and model init paths.
# For better profile coverage, replace this with a real benchmark workload.
${BINARY} &
SERVER_PID=$!

# Wait for server to be ready (health check)
MAX_WAIT=60
WAITED=0
while ! curl -sf http://localhost:8080/health > /dev/null 2>&1; do
    sleep 1
    WAITED=$((WAITED + 1))
    if [ "${WAITED}" -ge "${MAX_WAIT}" ]; then
        echo "  Warning: Server did not start within ${MAX_WAIT}s, continuing with partial profile..."
        break
    fi
done

# Send a few representative requests if server is ready
if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
    echo "  Server ready, sending test requests..."
    # Embedding request
    curl -sf -X POST http://localhost:8080/api/v1/embed \
        -H "Content-Type: application/json" \
        -d '{"text": "profile guided optimization test"}' > /dev/null 2>&1 || true
    # Similarity request
    curl -sf -X POST http://localhost:8080/api/v1/similarity \
        -H "Content-Type: application/json" \
        -d '{"source": "hello world", "target": "hello rust"}' > /dev/null 2>&1 || true
    # Batch embedding
    curl -sf -X POST http://localhost:8080/api/v1/embed/batch \
        -H "Content-Type: application/json" \
        -d '{"texts": ["text one", "text two", "text three"]}' > /dev/null 2>&1 || true
    sleep 1
fi

# Graceful shutdown
kill -SIGTERM "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true

echo "  Profile data collected in ${PROFILE_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Recompile with profile data
# ─────────────────────────────────────────────────────────────────────────────
echo "[Phase 3/3] Recompiling with profile-use..."
RUSTFLAGS="-C profile-use=${PROFILE_DIR}" \
    cargo build --release --features "${FEATURES}"

echo ""
echo "=== PGO build complete ==="
echo "Optimized binary: ${BINARY}"
echo ""
echo "To verify: ls -lh ${BINARY}"
