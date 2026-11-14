# Imagen de servicio (C3, C6): la API con un modelo entrenado durante el build.
FROM python:3.13-slim

# uv fijado a una versión concreta (no `latest`) para builds reproducibles.
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

# Primero solo dependencias: esta capa se cachea mientras uv.lock no cambie.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY configs ./configs
RUN uv sync --locked --no-dev
ENV PATH="/app/.venv/bin:$PATH"

# Entrena con datos simulados (semilla fija = mismo modelo en cada build).
# En el proyecto real, aquí se descargaría el modelo aprobado en vez de entrenarlo.
RUN gasolina simulate && gasolina pipeline \
    && rm -rf data/raw data/cusn data/ag

RUN useradd --create-home app && mkdir -p logs && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "gasolina_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
