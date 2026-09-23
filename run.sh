#!/usr/bin/env bash
# Launcher script for AI Factuality Comparison Tool

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

if [ -f ".venv/bin/python" ]; then
    .venv/bin/python app.py
else
    python3 app.py
fi
