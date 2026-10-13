FROM python:3.12-slim

WORKDIR /app

# 先装依赖（利用镜像层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码与数据
COPY src ./src
COPY data ./data
COPY eval ./eval
COPY .env.example ./.env.example

# 首次索引 + 健康检查
EXPOSE 8000

CMD ["sh", "-c", "python -m src.selfcheck && uvicorn src.app:app --host 0.0.0.0 --port 8000"]
