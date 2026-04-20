#!/usr/bin/env bash
set -euo pipefail

uvicorn mvp_app.main:app --host 0.0.0.0 --port 8090 --reload
