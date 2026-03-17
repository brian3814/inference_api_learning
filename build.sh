#!/usr/bin/env bash
set -euo pipefail
cd frontend && npm install && npm run build && cd ..
rm -rf backend/static
cp -r frontend/dist backend/static
echo "Build complete. Run: cd backend && uv run python run.py"
