@echo off
chcp 65001 >nul
echo ========================================
echo AI近视风险预测系统 - 启动脚本
echo ========================================
echo.

cd /d "%~dp0"

echo [1/2] 正在启动服务器...
echo 请稍候，服务器正在初始化模型...
echo.

python standalone_server.py

pause
