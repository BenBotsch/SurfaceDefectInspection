# Build and test the C++ classifier as CI does: docker build -f cpp/Dockerfile .
FROM ubuntu:24.04
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake libopencv-dev curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
WORKDIR /src
COPY . .
RUN uv sync --frozen && uv run sdi parity-fixture /tmp/fixture
RUN cmake -S cpp -B build -DSDI_FIXTURE=/tmp/fixture && cmake --build build -j \
    && ctest --test-dir build --output-on-failure
