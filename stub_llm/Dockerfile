# Tiny image on purpose: the stub must never be the heavy part of a load test.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv
RUN pip install --no-cache-dir fastapi==0.156.4 "uvicorn[standard]==0.53.0"
COPY stub_llm/ stub_llm/
EXPOSE 9000
CMD ["uvicorn", "stub_llm.app:app", "--host", "0.0.0.0", "--port", "9000"]
