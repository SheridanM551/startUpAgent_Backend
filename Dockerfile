FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 關鍵：把整個 repo 複製到 /app/startUpAgent_Backend 底下
RUN mkdir -p /app/startUpAgent_Backend
COPY . /app/startUpAgent_Backend

# 讓 /app 成為 Python 的模組搜尋路徑父層
ENV PYTHONPATH=/app

CMD ["uvicorn", "startUpAgent_Backend.main:app", "--host", "0.0.0.0", "--port", "8000"]