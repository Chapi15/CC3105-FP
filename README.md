# ¿Sube o baja la gasolina? — Proyecto de prueba CC3105

[![CI](https://github.com/Chapi15/CC3105-FP/actions/workflows/ci.yml/badge.svg)](https://github.com/Chapi15/CC3105-FP/actions/workflows/ci.yml)

Proyecto de referencia para el proyecto final de **CC3105 Machine Learning Engineering (MLOps)**,
UVG. Recorre de punta a punta lo que hacen los equipos: fotos de rótulos de precios en
Guatemala → arquitectura medallón (`cusn` / `ag` / `au`) → modelo que predice si el precio de la
próxima semana **sube o baja** → API en Docker → CI/CD con GitHub Actions → monitoreo de drift.

> Los datos son **simulados** (`gasolina simulate`): imitan los JSON que saldrían de las fotos,
> con errores típicos de captura. Las marcas de las estaciones son ficticias.

## Inicio rápido

Requisitos: [uv](https://docs.astral.sh/uv/) y, para contenedores, Docker.

```bash
make install     # uv sync --locked
make pipeline    # simula fotos y corre raw -> cusn -> ag -> au -> modelo -> predicción
make test        # 35 pruebas
make serve       # API en http://localhost:8000/docs
```

Sin `make`, todo es un comando del CLI: `uv run gasolina --help`.

## Estructura

```
configs/config.yaml        configuración (rutas, capas, modelo, umbrales)
src/gasolina_mlops/
  simulate.py              genera JSON de fotos (raw)
  layers.py                raw -> cusn -> ag -> au (pandas)
  validation.py            validaciones entre capas
  features.py              features y Pipeline de scikit-learn
  train.py                 split temporal, baseline, quality gates, model card
  predict.py               inferencia por lotes
  api.py                   FastAPI: /health, /model-info, /predict
  monitoring.py            PSI (drift) + resumen del log de la API
  storage.py               mismo código para carpeta local o s3://
  cli.py                   comando `gasolina`
tests/                     pruebas de capas, modelo, monitoreo y API
databricks/                notebooks 01_cusn, 02_ag, 03_au, 04_entrenamiento (Spark)
notebooks/                 exploración (formato percent)
docs/                      arquitectura y propuesta por entrega
Dockerfile, compose.yaml   imagen de servicio y stack (con sidecar s3-sync opcional)
.github/workflows/         ci.yml (CI) y deploy.yml (GHCR + DigitalOcean)
```

## El problema de ML

- **Unidad:** combustible (`super`, `regular`, `diesel`) × departamento × semana.
- **Entrada:** precio promedio de las últimas 4 semanas.
- **Objetivo:** `sube = 1` si el promedio de la próxima semana es mayor que el de esta.
- **Evaluación:** split **temporal**: las últimas 18 semanas son la prueba, sin barajar.
- **Quality gates:** ROC AUC ≥ 0.65 **y** superar la regla ingenua "si subió, sube".

Resultado con la semilla por defecto: regresión logística con AUC ≈ 0.74 y accuracy ≈ 0.72,
frente a 0.66 de la regla ingenua. Gradient boosting y random forest salen peores: como el
precio se sale del rango visto en entrenamiento, los árboles extrapolan mal. Sirve como
experimento para clase.

## Monitoreo

`make monitor` compara las últimas 12 semanas con los datos de entrenamiento (PSI por feature):

| Estado | PSI | Acción |
|---|---|---|
| ok | ≤ 0.25 | nada |
| aviso | > 0.25 | revisar |
| alerta | > 1.5 | investigar y re-entrenar |

`make drift-demo` simula un alza fuerte con mercado volátil y debe terminar en **alerta**.
`precio_t0` se reporta como informativa: el nivel de precio siempre cambia con el tiempo.

## Ambientes

| Ambiente | Rol | En este repo |
|---|---|---|
| Local / CI | pruebas automáticas | `make test`, `make pipeline`, `.github/workflows/ci.yml` |
| Databricks | desarrollo y pruebas | `databricks/*.py` (importar como notebooks) |
| AWS S3 | storage de las capas | `storage.base_uri: s3://...` con `uv sync --extra s3` |
| DigitalOcean | producción simulada | `deploy.yml` + `compose.yaml` |

### Usar S3 en lugar de la carpeta local

```bash
cp .env.example .env    # completar credenciales del usuario IAM del equipo
uv sync --extra s3
set -a; source .env; set +a
uv run gasolina --base-uri s3://<bucket>/<equipo> pipeline
```

### Despliegue en DigitalOcean

`deploy.yml` publica la imagen en GHCR en cada push a `main` y en tags `v*`. Si existe la
variable de repo `DO_HOST`, también despliega por SSH. Configuración en *Settings → Secrets and
variables → Actions*:

| Nombre | Tipo | Ejemplo |
|---|---|---|
| `DO_HOST` | variable | IP del droplet |
| `DO_USER` | variable | `cc3105` |
| `DO_SSH_KEY` | secret | llave privada con acceso a ese usuario |

En el servidor, el despliegue queda en `~/gasolina` y la API en el puerto 8000. La imagen de
GHCR nace privada: el workflow hace `docker login` con el `GITHUB_TOKEN` en cada despliegue.

## Flujo de trabajo (Gitflow)

| Rama | Propósito | Despliega |
|---|---|---|
| `main` | producción (DigitalOcean) | sí, automático |
| `develop` | integración | no (solo CI) |
| `feature/<tema>` | trabajo nuevo, sale de `develop` y vuelve por PR | no |
| `release/<versión>` | estabilización, de `develop` a `main` + tag `vX.Y.Z` | al fusionar |
| `hotfix/<tema>` | corrección urgente desde `main`, se fusiona en `main` y `develop` | al fusionar |

Regla: nadie hace push directo a `main`. Conviene protegerla en *Settings → Branches*.

## Secretos

Nunca en el repo: PATs de Databricks, access keys de AWS, tokens. Van en `.env` (ignorado por
git) o en los secrets de GitHub.

## Qué está probado y qué no

| Componente | Estado |
|---|---|
| Pipeline local, modelo, monitoreo, API (35 pruebas) | ✅ probado |
| Imagen Docker: build, usuario sin root, `/health`, `/predict`, log | ✅ probado localmente* |
| `compose.yaml` (`docker compose config`) | ✅ válido |
| Workflows de GitHub Actions | ⏳ sin correr todavía |
| Notebooks de Databricks | ⏳ sin probar en un workspace |
| Lectura y escritura en S3, `s3-sync` | ⏳ sin probar con un bucket real |
| Despliegue en DigitalOcean | ⏳ sin probar |

\* En el entorno de prueba, GHCR estaba bloqueado y uv se instaló con pip, con la misma
versión. El `COPY --from=ghcr.io/astral-sh/uv:0.11.32` del Dockerfile se valida en la primera
corrida de CI.
