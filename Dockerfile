FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for cron-based scheduling
RUN apt-get update \
  && apt-get install -y --no-install-recommends cron ca-certificates tzdata procps \
  && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install -U pip && pip install -r /app/requirements.txt

# App code
COPY . /app

# Cron config + entrypoint
RUN chmod +x /app/scripts/run_trade.sh /app/scripts/run_calibrate.sh /app/scripts/run_settle.sh
RUN chmod +x /app/scripts/docker_entrypoint.sh

ENTRYPOINT ["/app/scripts/docker_entrypoint.sh"]
