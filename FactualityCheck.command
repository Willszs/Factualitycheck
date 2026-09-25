#!/bin/bash
# 双击即可在 macOS 上直接启动 Factuality Check 桌面软件
cd "$(dirname "$0")"
.venv/bin/python app.py
