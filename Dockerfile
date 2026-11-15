# SPHEREx Blink: one container serves the API and the web page.
# Developed with AI assistance (Claude, Anthropic). See README "Use of AI".
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SPX_CACHE_DIR=/data/cache

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY scripts ./scripts

# The cache holds downloaded cutouts. Mount a volume at /data to keep it across restarts.
RUN mkdir -p /data/cache && useradd --create-home app && chown -R app /data /app
USER app

EXPOSE 8000
CMD ["sh", "-c", "python -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-8000}"]
