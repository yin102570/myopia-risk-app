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
start /B python simple_server.py

echo.
echo [3/3] 等待服务器启动...
timeout /t 3 /nobreak >nul

echo.
echo ========================================
echo 系统启动完成！
echo 访问地址: http://localhost:3000
echo 按 Ctrl+C 可停止服务器
echo ========================================

python -m http.server 8080 --directory backend\public
