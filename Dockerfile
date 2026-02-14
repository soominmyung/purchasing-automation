# Hugging Face Spaces: port 7860, Framer frontend calls this API
FROM python:3.11-slim

RUN useradd -m -u 1000 appuser
USER appuser
ENV PATH="/home/appuser/.local/bin:$PATH"
WORKDIR /app

COPY --chown=appuser requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=appuser main.py config.py schemas.py ./
COPY --chown=appuser routers ./routers
COPY --chown=appuser services ./services
COPY --chown=appuser utils ./utils

# Cloud Run: port 8080, Framer frontend calls this API
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
