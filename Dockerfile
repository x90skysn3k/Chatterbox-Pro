FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps + Python 3.12 from deadsnakes PPA (not in Ubuntu 22.04 default repos)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default + bootstrap pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

WORKDIR /app

# Create venv and install deps (cached layer)
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps auto-editor==27.1.1

# Copy application code
COPY Chatter.py server.py test-models.py test-corpus.json ./
COPY chatterbox/ ./chatterbox/
COPY install-patches.sh .

# Apply our patches over pip-installed chatterbox
RUN chmod +x install-patches.sh && ./install-patches.sh

# Create runtime dirs
RUN mkdir -p voices temp output

# NLTK data (sentence tokenizer)
RUN python3 -c "import nltk; nltk.download('punkt_tab', quiet=True)"

EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

ENV DEFAULT_VOICE=default.wav

CMD ["python3", "server.py"]
