#!/bin/bash
# 部署近视风险预测系统的脚本

cd /root/myopia-risk-ai_20260127150102/backend

# 杀掉旧的进程
pkill -f "python3 app.py" 2>/dev/null

# 停止旧的HTTP服务器
pkill -f "python3 -m http.server" 2>/dev/null

# 等待进程结束
sleep 2

# 启动新的后端服务器（在后台）
nohup python3 app.py > /tmp/myopia-backend.log 2>&1 &
echo $! > /tmp/myopia-backend.pid

# 等待后端启动
sleep 3

# 检查后端是否启动成功
if ps aux | grep -v grep | grep "python3 app.py" > /dev/null; then
    echo "后端服务器启动成功"
    echo "日志: /tmp/myopia-backend.log"
else
    echo "后端服务器启动失败，请检查日志"
    tail -20 /tmp/myopia-backend.log
    exit 1
fi

# 启动前端服务器（在3001端口）
cd /root/myopia-risk-ai_20260127150102/frontend/dist
nohup python3 -m http.server 3001 > /tmp/myopia-frontend.log 2>&1 &
echo $! > /tmp/myopia-frontend.pid

# 等待前端启动
sleep 2

# 检查前端是否启动成功
if ps aux | grep -v grep | grep "python3 -m http.server 3001" > /dev/null; then
    echo "前端服务器启动成功"
    echo "访问地址: http://154.8.203.14:3001"
else
    echo "前端服务器启动失败，请检查日志"
    tail -20 /tmp/myopia-frontend.log
    exit 1
fi

echo "部署完成！"
echo "后端: http://154.8.203.14:3000"
echo "前端: http://154.8.203.14:3001"
