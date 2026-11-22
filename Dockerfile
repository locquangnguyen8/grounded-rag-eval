FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \n    GRAG_CORPUS_DIR=/app/data/corpus GRAG_TRACE_LOG=/app/logs/traces.jsonl GRAG_FEEDBACK_LOG=/app/logs/feedback.jsonl
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY eval ./eval
RUN pip install ".[claude]" && useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["grounded-rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
