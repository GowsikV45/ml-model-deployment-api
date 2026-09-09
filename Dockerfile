FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# 0.0.0.0 makes Uvicorn listen on all network interfaces inside
# the container, allowing Docker's published port to reach the API.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]