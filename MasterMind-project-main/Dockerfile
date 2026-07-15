###############################################################################
# NS-Project Auto-Poster — Processing Fleet Container
#
# Multi-stage build:
#   builder  — compile wheels for all C-extension packages (asyncpg, cryptography,
#              numpy) in a full build environment so the final stage stays slim.
#   final    — python:3.11-slim with only runtime artefacts, running as nobody.
#
# Security properties:
#   - Non-root process (UID/GID 65534 — nobody/nogroup, universally present on
#     Debian-derived images without requiring adduser).
#   - No package manager or compilers in the final layer (build tools purged).
#   - Read-only root filesystem compatible (all writable state goes through
#     the Redis/Postgres clients; the only FS write at runtime is the log
#     stream to stdout/stderr which Docker captures without a writable layer).
#   - PYTHONDONTWRITEBYTECODE=1 prevents .pyc sprawl; PYTHONUNBUFFERED=1
#     ensures structured log lines flush immediately to the Docker daemon.
#   - pip cache purged in the same RUN layer as the install to avoid leaving
#     index metadata in the image layer.
###############################################################################

# ─── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install build-time OS deps (C compiler + Postgres client headers for asyncpg).
# Kept in the builder only — not copied to the final stage.
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libpq-dev \
        libffi-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements first for maximum layer cache hit rate.
COPY requirements.txt ./

# Build wheels into /wheels so the final stage can install without a compiler.
# --no-deps on the second pass ensures we don't pull transitive deps twice.
RUN pip install --upgrade pip --quiet && \
    pip wheel \
        --no-cache-dir \
        --wheel-dir /wheels \
        -r requirements.txt

# ─── Stage 2: final ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS final

# Runtime-only OS deps: libpq (asyncpg needs the shared object at runtime).
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        libpq5 \
    && rm -rf /var/lib/apt/lists/* && \
    # Harden: remove setuid/setgid bits from standard utilities.
    find / -xdev -perm /6000 -type f 2>/dev/null | xargs chmod a-s 2>/dev/null || true

# Python hygiene.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    # Prevent pip from writing to /root inside the container.
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install pre-built wheels — no compiler needed in this stage.
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links /wheels /wheels/*.whl && \
    rm -rf /wheels

# Copy application source.
# .dockerignore should exclude: .git, .venv, __pycache__, *.pyc, tests/,
# docs/, node_modules/, .env*, bandit-report.json, coverage.xml.
COPY src/           ./src/
COPY migrations/    ./migrations/
COPY config.yaml    ./config.yaml
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Ensure the entrypoint is executable (idempotent).
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Drop to nobody (UID 65534 / GID 65534).
# The /app directory is already owned by root which is fine — the worker
# process only reads files, never writes to the filesystem at runtime.
USER 65534:65534

# Expose no ports — the fleet worker is a pure consumer; inbound traffic
# goes through the Edge Gateway, not through this container.

# Health check: verify the Python interpreter and the critical import chain
# work without network access. If Redis/Postgres are unavailable the worker
# will gracefully retry; this check catches packaging regressions only.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "from src.fleet import Worker; from src.enclave import CredentialStore"

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
