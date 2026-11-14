# %% [markdown]
# # Exploración de la capa `au` (C2)
#
# Notebook en formato *percent*: se abre como notebook en VS Code o Jupyter (con jupytext) y a la
# vez es un `.py` normal que se versiona bien en git. Sirve para **explorar**; todo lo que se
# decide aquí se pasa a `src/gasolina_mlops/` con pruebas.
#
# Requiere haber corrido `make pipeline`. Para graficar: `uv add --dev matplotlib`.

# %%
import pandas as pd

semanal = pd.read_parquet("data/au/precios_semanales.parquet")
nacional = pd.read_parquet("data/au/precio_nacional.parquet")
modelo = pd.read_parquet("data/au/modelo.parquet")
semanal.head()

# %% [markdown]
# ## Precio nacional por combustible

# %%
nacional.pivot(index="semana", columns="combustible", values="precio_promedio").plot(
    figsize=(10, 4), title="Precio promedio nacional (Q/galón)"
)

# %% [markdown]
# ## ¿Qué tan balanceado está el objetivo?

# %%
modelo["sube"].value_counts(normalize=True)

# %% [markdown]
# ## ¿Hay inercia? Si subió esta semana, ¿sube la próxima?

# %%
subio = modelo["precio_t0"] > modelo["precio_t1"]
pd.crosstab(subio, modelo["sube"], normalize="index")

# %% [markdown]
# ## Diferencia de precio por departamento (recargo por transporte)

# %%
(
    semanal.groupby(["departamento", "combustible"])["precio_promedio"].mean().unstack()
    .sub(semanal[semanal["departamento"] == "Guatemala"]
         .groupby("combustible")["precio_promedio"].mean())
    .round(2)
)
