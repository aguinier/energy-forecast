FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (libgomp needed by LightGBM/XGBoost, cron for scheduling)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 cron \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY config.py ./
COPY src/ src/
COPY scripts/ scripts/

# Install crontab: 07:00, 14:00, 15:30, 19:00
COPY docker/crontab /etc/cron.d/energy-forecast
RUN chmod 0644 /etc/cron.d/energy-forecast && crontab /etc/cron.d/energy-forecast

COPY docker/entrypoint.sh /entrypoint.sh

# Database lives on a mounted volume
ENV ENERGY_DB_PATH=/data/energy_dashboard.db

RUN mkdir -p /app/logs /app/models /data

ENTRYPOINT ["/entrypoint.sh"]
