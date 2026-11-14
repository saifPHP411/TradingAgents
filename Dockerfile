FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY . .
RUN pip install --no-cache-dir .

FROM python:3.13-slim

# TRADINGAGENTS_IN_DOCKER lets an update notice name the rebuild command.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRADINGAGENTS_IN_DOCKER=1

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# A login shell (`bash -lc`, `sh -lc`) sources /etc/profile, which on Debian
# *overwrites* PATH (see its `PATH="..."` line) and discards the ENV prepend
# above. `python` then resolves to the base image's interpreter, which has none
# of the project's dependencies, and any `python -m ...` command fails with
# `ModuleNotFoundError` (e.g. `No module named 'dotenv'`). Re-prepend the venv
# from /etc/profile.d, which /etc/profile sources *after* it resets PATH, so the
# app's interpreter is the one a manual `report`/`verify` in a shell also gets.
RUN printf '%s\n' 'export PATH="/opt/venv/bin:$PATH"' > /etc/profile.d/10-venv-path.sh \
 && chmod 0644 /etc/profile.d/10-venv-path.sh

RUN useradd --create-home appuser \
 && install -d -m 0755 -o appuser -g appuser /home/appuser/.tradingagents \
 && install -d -m 0755 -o appuser -g appuser /home/appuser/app/data
USER appuser
WORKDIR /home/appuser/app

COPY --from=builder --chown=appuser:appuser /build .

# Health check for Railway - uses the dashboard's /healthz endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/healthz', timeout=5)" || exit 1

# The image's default process is the paper dashboard, which is the only program
# in the image that serves /healthz (the HEALTHCHECK above) and the only one a
# platform deployment wants to run.
#
# It is deliberately NOT the `tradingagents` analysis CLI. A Dockerfile
# ENTRYPOINT is *prepended* to whatever command the platform runs, so an
# entrypoint of `tradingagents` would turn a platform start command into
# `tradingagents <start command>` — the analysis CLI is a Typer app, so it
# would exit with "No such command" and the service would never come up.
# Starting from the dashboard module also means the image's default is the
# same program the deployment's start command names, so the container behaves
# the same whether the platform replaces the command or appends to it.
#
# The analysis CLI is still reachable: `docker run ... python -m cli.main` or
# the `tradingagents` console script, and docker-compose gives the analysis
# services their own explicit entrypoint.
ENTRYPOINT ["python", "-m", "tradingagents.paper_experiment.dashboard.server"]
