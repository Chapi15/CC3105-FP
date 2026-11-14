# Propuesta: qué cubre este proyecto en cada entrega

> Propuesta armada a partir del programa del curso (competencias C1–C8 y parciales). No
> reemplaza la rúbrica oficial de cada entrega.

| Entrega | Parcial | Competencias | Qué mostrar con este repo |
|---|---|---|---|
| **0** | P2 · Proyectos de ML | C1, C4 | Problema y pregunta de negocio (¿sube o baja?), CRISP-DM, estructura del repo (`src/`, `tests/`, `configs/`), `pyproject.toml` + `uv.lock`, diagrama de [arquitectura](arquitectura.md) |
| **1** | P1 · Conceptos de ML | C2, C7 | Notebook de exploración, features (`features.py`), split temporal, baseline ingenuo, comparación de algoritmos, model card |
| **2** | P3 · Automatizaciones | C2, C4, C5, C7, C8 | Pipeline medallón `cusn → ag → au` (local y Databricks), validaciones entre capas, Makefile, CI en GitHub Actions, reporte de drift |
| **3** | PF · Plataformas | C3, C5, C6, C8 | API FastAPI, Docker y Compose, imagen en GHCR, despliegue en DigitalOcean, logs de predicciones hacia S3 |

## Lista de verificación para la presentación

- [ ] Diagrama de arquitectura actualizado con los nombres reales del bucket y del workspace.
- [ ] `make test` y CI en verde.
- [ ] Model card con métricas y comparación contra el baseline.
- [ ] Demo en vivo de `/docs` en el servidor de DigitalOcean.
- [ ] Reporte de drift (`make drift-demo`) y explicación de qué harían si se activa.
- [ ] Ningún secreto en el repo (`git log -p | grep -i secret` no encuentra nada).
