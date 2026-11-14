# Orquestación local del ciclo de vida (C5). `make help` lista los comandos.
RUN = uv run
CLI = $(RUN) gasolina
SHOCK = data_shock

.PHONY: help install lint test simulate pipeline monitor drift-demo serve smoke up down clean

help:  ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

install:  ## Instala dependencias (incluye dev) desde uv.lock
	uv sync --locked

lint:  ## Revisa estilo con ruff
	$(RUN) ruff check src tests

test:  ## Ejecuta las pruebas
	$(RUN) pytest

simulate:  ## Genera los JSON de fotos simuladas en data/raw/fotos
	$(CLI) simulate

pipeline: simulate  ## raw -> cusn -> ag -> au -> modelo -> predicción de la próxima semana
	$(CLI) pipeline

monitor:  ## Drift de las últimas 12 semanas contra los datos de entrenamiento
	$(CLI) monitor

drift-demo:  ## Simula un alza fuerte de precios y muestra que el monitoreo lo detecta
	$(CLI) --base-uri $(SHOCK) simulate --shock
	$(CLI) --base-uri $(SHOCK) cusn
	$(CLI) --base-uri $(SHOCK) ag
	$(CLI) --base-uri $(SHOCK) au
	$(CLI) monitor --current $(SHOCK)/au/modelo.parquet

serve:  ## API local en http://localhost:8000/docs
	$(RUN) uvicorn gasolina_mlops.api:app --reload --port 8000

smoke:  ## Smoke test contra la API en ejecución (PORT=8000 por defecto)
	curl -fsS localhost:$${PORT:-8000}/health
	curl -fsS -X POST localhost:$${PORT:-8000}/predict -H 'Content-Type: application/json' \
	  -d '{"series":[{"departamento":"Petén","combustible":"diesel","precios":[29.8,30.0,30.3,30.6]}]}'

up:  ## Levanta la API con Docker Compose (construye la imagen localmente)
	docker compose up -d --build

down:  ## Detiene el stack (conserva el volume de logs)
	docker compose down

clean:  ## Borra artefactos generados
	rm -rf data $(SHOCK) models reports logs .pytest_cache .ruff_cache
