# FactoryFlow — Hugging Face Space image (CPU)
# The live AMD MI300X demo runs on a separate cloud box; this image
# is the prize-track Space and runs MOMENT on CPU for browsability.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/user/.cache/huggingface \
    DEMO_MODE=true \
    AMD_DEVICE=cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

COPY --chown=user:user requirements.txt .

# CPU-only torch from PyPI (HF Spaces free tier has no GPU).
RUN pip install --user --upgrade pip && \
    pip install --user torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --user -r requirements.txt && \
    pip install --user --no-deps momentfm==0.1.4

ENV PATH="/home/user/.local/bin:${PATH}"

COPY --chown=user:user . .

EXPOSE 7860
CMD ["python", "-m", "src.demo.app"]
