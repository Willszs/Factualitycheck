#!/bin/bash
# 双击即可在 macOS 上直接启动 Factuality Check 桌面软件
cd "$(dirname "$0")"
.venv/bin/python app.py
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo ""
    echo "⚠️ 程序异常退出 (退出码: $STATUS)，请检查上述错误信息。"
    read -p "按回车键关闭此窗口..."
fi
