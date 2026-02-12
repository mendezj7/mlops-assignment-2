FROM python:3.10-slim-bullseye

WORKDIR /app

# Set proxy environment variables
ENV HTTP_PROXY=http://fastweb.bell.ca:80
ENV HTTPS_PROXY=http://fastweb.bell.ca:80
ENV http_proxy=http://fastweb.bell.ca:80
ENV https_proxy=http://fastweb.bell.ca:80

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy and install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy and install requirements with proxy
#COPY requirements.txt ./
#RUN pip install --no-cache-dir --upgrade pip \
#    --proxy http://fastweb.bell.ca:80 \
#    --trusted-host pypi.python.org \
#    --trusted-host pypi.org \
#    --trusted-host files.pythonhosted.org && \
#    pip install --no-cache-dir -r requirements.txt \
    --proxy http://fastweb.bell.ca:80 \
    --trusted-host pypi.python.org \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org

# Copy application code and artifacts
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser artifacts ./artifacts

# Switch to non-root user
USER appuser

EXPOSE 8000

ENV MODEL_PATH=/app/artifacts/models/baseline_cnn.pt

# Health check using Python instead of curl
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
