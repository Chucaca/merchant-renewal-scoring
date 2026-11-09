FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt prometheus_client

COPY renewal_scoring/ renewal_scoring/
COPY pyproject.toml README.md ./

# Build the synthetic snapshot at image build time (a real deployment would
# instead read the daily scoring snapshot from storage).
RUN python -m renewal_scoring.cli generate --n 5000 && python -m renewal_scoring.cli score

ENV RENEWAL_SCORES_PATH=/app/data/scores.csv RENEWAL_CACHE_TTL=300
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "renewal_scoring.api:app", "--host", "0.0.0.0", "--port", "8000"]
