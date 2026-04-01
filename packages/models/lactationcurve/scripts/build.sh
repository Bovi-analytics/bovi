#!/usr/bin/env bash
# Build script for the lactationcurve package.
# Must be run from the repo root.
set -euo pipefail

bash scripts/workflows/sync-version.sh
cd packages/models/lactationcurve && uv build
