FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev && \
    rm -rf /var/lib/apt/lists/*

# Install uv and python 3.12
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN uv python install 3.12

RUN git clone https://github.com/groundlight/r1_vlm.git && \
    git clone --branch release_2025_03_06 --single-branch https://github.com/groundlight/verifiers.git && \
    git clone --branch release_2025_03_06 --single-branch https://github.com/groundlight/trl.git

WORKDIR /app/r1_vlm

RUN uv venv && \
    uv pip install hatchling editables psutil torch==2.5.1 && \
    uv sync --no-build-isolation

CMD ["bash"]
