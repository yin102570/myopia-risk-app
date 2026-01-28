@echo off
chcp 65001 >nul
echo ========================================
echo AI近视风险预测系统 - 启动脚本
echo ========================================

echo.
echo [1/3] 准备前端文件...
if not exist "backend\public" mkdir backend\public
xcopy /Y /E /I "frontend\dist\*" "backend\public\" >nul 2>&1
echo 前端文件已复制完成

echo.
echo [2/3] 启动后端API服务器...
cd backend
python simple_server.py

pause
