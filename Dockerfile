FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        libsecp256k1-dev \
        autoconf \
        automake \
        libtool \
    && rm -rf /var/lib/apt/lists/*

COPY uv.lock pyproject.toml ./
RUN mkdir -p /routstr

RUN uv sync --frozen --no-dev --no-install-project

WORKDIR /app

COPY . .
COPY docker/routstr-cap-entrypoint /usr/local/bin/routstr-cap-entrypoint

RUN mkdir -p /app/data/logs /app/data/tmp /app/data/.cache /app/data/.config /app/data/.local/share \
    && chmod 0755 /usr/local/bin/routstr-cap-entrypoint \
    && chown -R 10001:10001 /app/data

ARG GIT_COMMIT=""
ARG GIT_TAG=""
ENV GIT_COMMIT=${GIT_COMMIT}
ENV GIT_TAG=${GIT_TAG}
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ROUTSTR_DATA_DIR=/app/data
ENV DATABASE_URL=sqlite+aiosqlite:////app/data/keys.db
ENV ROUTSTR_LOG_DIR=/app/data/logs
ENV HOME=/app/data
ENV TMPDIR=/app/data/tmp
ENV XDG_CACHE_HOME=/app/data/.cache
ENV XDG_CONFIG_HOME=/app/data/.config
ENV XDG_DATA_HOME=/app/data/.local/share
ENV ENABLE_ANALYTICS_SHARING=false
ENV CONFIDENTIAL_ROUTING_MODE=required
ENV ROUTSTR_TEE_ATTESTATION_REQUIRED=true

EXPOSE 8000

USER 10001:10001

CMD ["/usr/local/bin/routstr-cap-entrypoint"]
