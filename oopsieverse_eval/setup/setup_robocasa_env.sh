#!/usr/bin/env bash
# Creates the `oopsieverse_robocasa` conda env and installs RoboCasa/RoboSuite
# into it, using OopsieVerse's own install.py (unmodified). Safe to re-run:
# install.py skips steps that already exist (env, clones).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OOPSIEVERSE_DIR="$SCRIPT_DIR/../external/oopsieverse"

cd "$OOPSIEVERSE_DIR"
python install.py --new_env --robocasa
