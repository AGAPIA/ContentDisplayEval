FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 50051

ENTRYPOINT ["python", "server.py", "--config", "/app/config/server_config.yaml"]
