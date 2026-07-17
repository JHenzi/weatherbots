FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Add this before your RUN apt-get command
# RUN sed -i 's/deb.debian.org/ftp.us.debian.org/g' /etc/apt/sources.list

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
RUN chmod +x /app/scripts/run_trade.sh /app/scripts/run_calibrate.sh /app/scripts/run_settle.sh /app/scripts/run_intraday_pulse.sh
RUN chmod +x /app/scripts/docker_entrypoint.sh

# Surface a hung dashboard in `docker ps` (unhealthy) instead of a silent 000.
# The in-container supervisor still self-heals; this is for visibility/alerting.
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8080/', timeout=8)" || exit 1

ENTRYPOINT ["/app/scripts/docker_entrypoint.sh"]
