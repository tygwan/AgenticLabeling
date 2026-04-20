#!/usr/bin/env bash
set -euo pipefail

docker compose -f docker-compose.legacy.yml up --build
