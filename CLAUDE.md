# Contexto para Claude Code

Responder en español.

## Curso

CC3105 Machine Learning Engineering (MLOps), UVG. El proyecto final se hace en 7 equipos:
monitorear precios de gasolina en Guatemala a partir de fotos y predecir si suben o bajan.

- Medallón en Databricks con datos en un bucket de S3: capas `cusn` (bronce), `ag` (plata),
  `au` (oro), más un external location para las imágenes.
- Databricks = desarrollo y pruebas. DigitalOcean (droplets con Docker, usuario `cc3105`) =
  producción simulada.
- Entregables en GitHub: diagrama de arquitectura, código del pipeline, presentación. Se
  presenta ante un jurado externo.

## Este repositorio

Proyecto de prueba de punta a punta que sirve de referencia para ese proyecto final (ver
`README.md` y `docs/`). Usa las mismas decisiones que `docker-action-demo`: uv con Python 3.13,
imagen en GHCR y sidecar `s3-sync` en Compose (perfil `s3`).

- Código en `src/gasolina_mlops/`. La lógica en pandas es la fuente de verdad que se prueba en
  CI, y `databricks/*.py` es su traducción a Spark.
- Antes de cerrar un cambio: `make lint test pipeline`.
- Gitflow: `main` = producción, `develop` = integración, `feature/*`, `release/*`, `hotfix/*`.
  No hacer push directo a `main`.

## Reglas

- Nunca escribir secretos en el repo (PATs de Databricks, access keys de AWS, tokens). Van en
  `.env` (ignorado) o en secrets de GitHub.
- Los datos son simulados. No presentarlos como precios reales.
