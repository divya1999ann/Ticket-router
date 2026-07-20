FROM python:3.12-slim

WORKDIR /app

# libgomp1: faiss-cpu needs OpenMP at runtime, not just build time
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch build first (the default PyPI wheel pulls CUDA and is ~10x larger);
# pip then sees torch==2.5.1 already satisfied and skips reinstalling it from requirements.txt
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Only what app2.py actually imports at runtime — notebooks/data/scripts aren't needed
# to serve the app, and skipping them keeps the image and build context small.
COPY pipeline/ ./pipeline/
COPY app2.py .

EXPOSE 7860

# Gradio defaults to binding 127.0.0.1, which host port-mapping (-p) can't reach from
# outside the container. This makes it bind on all interfaces without touching app2.py.
ENV GRADIO_SERVER_NAME=0.0.0.0

# No secrets.env copied into the image — pass HF_TOKEN / GROQ_API_KEY at `docker run`
# time (--env-file secrets.env or -e), since os.getenv() reads real env vars regardless
# of whether python-dotenv finds a file.
CMD ["python", "app2.py"]
