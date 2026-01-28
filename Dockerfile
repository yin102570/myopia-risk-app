FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

FROM python:3.9-slim AS backend

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 复制Python依赖
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制后端代码
COPY backend/ .

# 创建models目录
RUN mkdir -p models

# 安装Node.js和npm用于运行API服务器
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# 安装后端Node依赖
COPY backend/package*.json ./
RUN npm install

# 复制前端构建产物
COPY --from=frontend-builder /app/frontend/dist ./public

EXPOSE 3000

CMD ["npm", "start"]
