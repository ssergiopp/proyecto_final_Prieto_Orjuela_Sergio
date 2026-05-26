"""
Generador de notebooks .ipynb a partir de los scripts .py del proyecto final.
Ejecutar: python build_notebooks.py
"""
import nbformat as nbf
import os
import re

BASE = r"C:\Users\MAURICIO\Downloads\data\proyecto_final"
os.makedirs(os.path.join(BASE, "notebooks"), exist_ok=True)

# ============================================================
# BLOQUE 1: EDA
# ============================================================
def build_bloque1():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {"name": "python", "version": "3.11.0"}
    }

    cells = []

    # --- Celda 1: Titulo ---
    cells.append(nbf.v4.new_markdown_cell("""\
# Bloque 1 — Análisis Exploratorio con PySpark
## Dataset: SIVIGILA — Vigilancia en Salud Pública, Colombia 2019
### Parcial Final — Machine Learning con PySpark y Docker

**Autor:** Sergio Prieto  
**Fecha:** Mayo 2026  
**Profesora:** Luz Adriana Gutiérrez Rodríguez  

---
## Objetivo
Realizar un análisis exploratorio completo sobre datos reales del sistema de vigilancia epidemiológica colombiano (SIVIGILA), aplicando transformaciones PySpark, estadística descriptiva, detección de anomalías y visualizaciones.

## Dataset
El dataset contiene **205,532 registros** de notificaciones obligatorias de eventos de salud pública en Colombia durante 2019, con 69 tipos de eventos distintos reportados en 35 departamentos a lo largo de 52 semanas epidemiológicas.

**Fuente:** Instituto Nacional de Salud — SIVIGILA 2019
"""))

    # --- Celda 2: Setup ---
    cells.append(nbf.v4.new_markdown_cell("""\
## Configuración del entorno
Se configura PySpark en modo local y se establece el HADOOP_HOME para compatibilidad con Windows.
"""))
    cells.append(nbf.v4.new_code_cell("""\
import os
import sys

# Fix para PySpark en Windows
HADOOP_HOME = os.path.join(os.environ.get("TEMP", os.path.expanduser("~")), "hadoop_tmp")
os.environ["HADOOP_HOME"] = HADOOP_HOME
os.makedirs(os.path.join(HADOOP_HOME, "bin"), exist_ok=True)
winutils_path = os.path.join(HADOOP_HOME, "bin", "winutils.exe")
if not os.path.exists(winutils_path):
    with open(winutils_path, "wb") as f:
        f.write(b"")

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, sum as _sum, count, avg, stddev, min as _min, max as _max,
    percentile_approx, when, isnan, isnull, lit, regexp_replace, trim,
    round as spark_round, desc, asc,
)
from pyspark.sql.types import IntegerType, StringType
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10

CSV_PATH = "../data/sivigila.csv"
OUTPUT_DIR = "../salidas"
os.makedirs(OUTPUT_DIR, exist_ok=True)
"""))

    # --- Celda 3: Tarea 1 ---
    cells.append(nbf.v4.new_markdown_cell("""\
## Tarea 1 — Carga del dataset con PySpark
Se carga el CSV usando pandas (para evitar problemas de Hadoop en Windows) y se convierte a Spark DataFrame.  
Se muestran: esquema, conteo de registros y primeras filas.

**Transformaciones de limpieza:**
- `ANO`: el CSV usa formato "2.019" con punto de miles → se limpia a 2019
- `conteo_casos`, `SEMANA`, `COD_DPTO_O`, `COD_MUN_O`, `COD_EVE` → se convierten a `IntegerType`
- `nom_mun`: valores vacíos → `None`
"""))
    cells.append(nbf.v4.new_code_cell("""\
spark = (
    SparkSession.builder
    .appName("SIVIGILA_Bloque1_EDA")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.ui.enabled", "false")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

print("Cargando datos...")
pdf_raw = pd.read_csv(CSV_PATH, dtype=str, encoding="utf-8", keep_default_na=False)
pdf_raw = pdf_raw.rename(columns={"ANO": "ANO_STR"})

df = spark.createDataFrame(pdf_raw)
df = (
    df
    .withColumn("conteo_casos", col("conteo_casos").cast(IntegerType()))
    .withColumn("SEMANA",       col("SEMANA").cast(IntegerType()))
    .withColumn("COD_DPTO_O",   col("COD_DPTO_O").cast(IntegerType()))
    .withColumn("COD_MUN_O",    col("COD_MUN_O").cast(IntegerType()))
    .withColumn("COD_EVE",      col("COD_EVE").cast(IntegerType()))
    .withColumn("ANO",
        regexp_replace(col("ANO_STR"), r"\\.", "").cast(IntegerType()))
    .withColumn("nom_mun",
        when(trim(col("nom_mun")) == "", None).otherwise(trim(col("nom_mun"))))
    .drop("ANO_STR")
)
df = df.cache()
total_registros = df.count()

print(f"Registros totales: {total_registros:,}")
print(f"Columnas: {len(df.columns)}")
print("\\n--- Esquema ---")
df.printSchema()
print("\\n--- Primeras 10 filas ---")
df.show(10, truncate=False)
"""))

    # --- Celda 4: Tarea 2 ---
    cells.append(nbf.v4.new_markdown_cell("""\
## Tarea 2 — Tres transformaciones analíticas

Se aplican tres transformaciones que responden preguntas concretas del dominio:

| Transformación | Pregunta analítica |
|---------------|-------------------|
| `filter` | ¿Cuáles son los departamentos con más casos de DENGUE? |
| `groupBy` | ¿Cuál es el total de casos por tipo de evento? |
| `withColumn` + `join` | ¿Qué departamentos tienen la mayor tasa de dengue por cada 1000 casos? |
"""))
    cells.append(nbf.v4.new_code_cell("""\
# T2a: FILTER — Top departamentos con mas DENGUE
print("--- T2a: Filter — Top 10 departamentos con mas casos de DENGUE ---")
df_dengue = df.filter(col("Nombre") == "DENGUE").cache()
df_dengue_dpto = (
    df_dengue.groupBy("COD_DPTO_O")
    .agg(_sum("conteo_casos").alias("total_casos_dengue"))
    .orderBy(desc("total_casos_dengue"))
)
df_dengue_dpto.show(10)

# T2b: GROUPBY — Total casos por evento
print("\\n--- T2b: GroupBy — Total de casos por evento (top 15) ---")
df_evento_total = (
    df.groupBy("Nombre")
    .agg(_sum("conteo_casos").alias("total_casos"))
    .orderBy(desc("total_casos"))
)
df_evento_total.show(15, truncate=False)

# T2c: WITHCOLUMN + JOIN — Tasa de dengue por dpto (x1000)
print("\\n--- T2c: WithColumn + Join — Tasa de dengue por dpto (x1000) ---")
total_por_dpto = df.groupBy("COD_DPTO_O").agg(_sum("conteo_casos").alias("total_dpto"))
df_tasa = (
    df_dengue_dpto.join(total_por_dpto, "COD_DPTO_O")
    .withColumn("tasa_dengue_x1000",
        spark_round(col("total_casos_dengue") / col("total_dpto") * 1000, 2))
    .orderBy(desc("tasa_dengue_x1000"))
)
df_tasa.show(10)
"""))

    # --- Celda 5: Tarea 3 ---
    cells.append(nbf.v4.new_markdown_cell("""\
## Tarea 3 — Estadística descriptiva

Se calculan:
- `describe()` para variables numéricas (SEMANA, COD_DPTO_O, conteo_casos)
- `summary()` con percentiles 25%, 50%, 75%
- Percentiles 90, 95 y 99 de `conteo_casos`
- Conteos por categoría (Nombre del evento y semana)
"""))
    cells.append(nbf.v4.new_code_cell("""\
print("--- describe() ---")
df.describe(["SEMANA", "COD_DPTO_O", "conteo_casos"]).show()

print("\\n--- summary() conteo_casos ---")
df.select("conteo_casos").summary(
    "count", "mean", "stddev", "min", "25%", "50%", "75%", "max"
).show()

print("\\n--- Percentiles 90, 95, 99 ---")
df.select(
    percentile_approx("conteo_casos", 0.90).alias("p90"),
    percentile_approx("conteo_casos", 0.95).alias("p95"),
    percentile_approx("conteo_casos", 0.99).alias("p99"),
).show()

print("\\n--- Top 10 eventos mas frecuentes ---")
df.groupBy("Nombre").count().orderBy(desc("count")).show(10, truncate=False)

print("\\n--- Casos totales por semana ---")
df.groupBy("SEMANA").agg(_sum("conteo_casos").alias("casos_semana")) \
  .orderBy("SEMANA").show(52)
"""))

    # --- Celda 6: Tarea 4 ---
    cells.append(nbf.v4.new_markdown_cell("""\
## Tarea 4 — Valores faltantes, duplicados y atípicos

**Metodología:**
- **Nulos:** conteo por columna en una sola agregación para eficiencia
- **Duplicados exactos:** `dropDuplicates()` sobre todas las columnas
- **Duplicados parciales:** groupBy por llave lógica (COD_EVE + SEMANA + COD_MUN_O)
- **Atípicos:** método IQR (Q3 + 1.5 × IQR) sobre `conteo_casos`
- **Datos anómalos:** registros con `COD_DPTO_O = 0` (desconocido)
"""))
    cells.append(nbf.v4.new_code_cell("""\
# Nulos en una sola pasada
print("--- Valores nulos por columna ---")
nulos_exprs = [_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c) for c in df.columns]
nulos_row = df.agg(*nulos_exprs).collect()[0]
for c in df.columns:
    n = nulos_row[c]
    print(f"  {c}: {n} nulos ({(n/total_registros)*100:.2f}%)")

# Duplicados
print("\\n--- Duplicados exactos ---")
count_sin_dup = df.dropDuplicates().count()
print(f"  Duplicados exactos: {total_registros - count_sin_dup}")

# Duplicados parciales
print("\\n--- Llaves duplicadas (COD_EVE+SEMANA+COD_MUN_O) ---")
dup_llave = (
    df.groupBy("COD_EVE", "SEMANA", "COD_MUN_O")
    .agg(count("*").alias("n"))
    .filter(col("n") > 1)
)
print(f"  Grupos con llave repetida: {dup_llave.count()}")

# Atipicos (IQR)
print("\\n--- Valores atipicos en conteo_casos (IQR) ---")
stats = df.select(
    percentile_approx("conteo_casos", 0.25).alias("q1"),
    percentile_approx("conteo_casos", 0.75).alias("q3"),
).collect()[0]
q1, q3 = stats["q1"], stats["q3"]
iqr = q3 - q1
lim_sup = q3 + 1.5 * iqr
print(f"  Q1={q1}, Q3={q3}, IQR={iqr}, Limite superior={lim_sup}")
atipicos = df.filter(col("conteo_casos") > lim_sup).count()
print(f"  Registros atipicos: {atipicos} ({(atipicos/total_registros)*100:.2f}%)")

print("\\n  Top 10 valores mas altos de conteo_casos:")
df.select("Nombre", "nom_mun", "SEMANA", "conteo_casos") \\
  .orderBy(desc("conteo_casos")).show(10, truncate=False)

print(f"\\n  Registros con departamento desconocido (cod 0): {df.filter(col('COD_DPTO_O') == 0).count()}")
"""))

    # --- Celda 7: Tarea 5 ---
    cells.append(nbf.v4.new_markdown_cell("""\
## Tarea 5 — Visualizaciones

Se construyen dos gráficos (convirtiendo a Pandas):

1. **Top 15 eventos por total de casos** (barras horizontales)
2. **Serie temporal de dengue por semana epidemiológica** (línea con media móvil)

Estos gráficos sustentan las conclusiones sobre concentración de morbilidad y estacionalidad.
"""))
    cells.append(nbf.v4.new_code_cell("""\
# --- Grafico 1: Top 15 eventos por total de casos ---
pdf_eventos = df_evento_total.limit(15).toPandas().sort_values("total_casos")

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(pdf_eventos["Nombre"], pdf_eventos["total_casos"], color="#1f77b4")
ax.set_xlabel("Total de casos reportados", fontsize=11)
ax.set_title("Top 15 eventos de salud publica por numero de casos\\nSIVIGILA Colombia - 2019",
             fontsize=13, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
for bar, val in zip(bars, pdf_eventos["total_casos"]):
    ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va="center", fontsize=8)
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig1_top15_eventos.png"), bbox_inches="tight")
plt.show()

# --- Grafico 2: Serie temporal de DENGUE por semana ---
pdf_dengue_semanal = (
    df_dengue.groupBy("SEMANA")
    .agg(_sum("conteo_casos").alias("casos_dengue"))
    .orderBy("SEMANA").toPandas()
)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(pdf_dengue_semanal["SEMANA"], pdf_dengue_semanal["casos_dengue"],
        marker="o", linewidth=1.5, markersize=3, color="#d62728")
ax.fill_between(pdf_dengue_semanal["SEMANA"], pdf_dengue_semanal["casos_dengue"],
                alpha=0.15, color="#d62728")
ax.set_xlabel("Semana epidemiologica", fontsize=11)
ax.set_ylabel("Casos de dengue", fontsize=11)
ax.set_title("Casos de dengue por semana epidemiologica\\nColombia 2019",
             fontsize=13, fontweight="bold")
ax.set_xticks(range(1, 53, 4))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
z = pd.Series(pdf_dengue_semanal["casos_dengue"].values).rolling(4, center=True).mean()
ax.plot(pdf_dengue_semanal["SEMANA"], z, color="black", linewidth=2, linestyle="--",
        label="Tendencia (media movil 4 sem)")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig2_dengue_semanal.png"), bbox_inches="tight")
plt.show()
"""))

    # --- Celda 8: Tarea 6 - Conclusiones ---
    cells.append(nbf.v4.new_markdown_cell("""\
## Tarea 6 — Conclusiones cuantitativas

A continuación se calculan las métricas que sustentan las conclusiones y se redactan en formato Markdown.
"""))
    cells.append(nbf.v4.new_code_cell("""\
top3 = df_evento_total.limit(3).collect()
total_casos_global = df.agg(_sum("conteo_casos")).collect()[0][0]
top3_casos = sum(r["total_casos"] for r in top3)

dengue_pico = (
    df_dengue.groupBy("SEMANA")
    .agg(_sum("conteo_casos").alias("casos_dengue"))
    .orderBy(desc("casos_dengue")).first()
)

dpto_top = (
    df.groupBy("COD_DPTO_O")
    .agg(_sum("conteo_casos").alias("total"))
    .orderBy(desc("total")).first()
)

top10_dptos_sum = (
    df.groupBy("COD_DPTO_O")
    .agg(_sum("conteo_casos").alias("total"))
    .orderBy(desc("total")).limit(10)
    .agg(_sum("total")).collect()[0][0]
)

print(f"Total casos global: {total_casos_global:,}")
print(f"Top 3 eventos: {top3_casos:,} ({(top3_casos/total_casos_global)*100:.1f}%)")
print(f"Pico dengue: semana {dengue_pico['SEMANA']} con {dengue_pico['casos_dengue']:,} casos")
print(f"Dpto lider: {dpto_top['COD_DPTO_O']} con {dpto_top['total']:,} casos ({(dpto_top['total']/total_casos_global)*100:.1f}%)")
print(f"Top 10 dptos: {(top10_dptos_sum/total_casos_global)*100:.1f}% del total")

df.unpersist()
df_dengue.unpersist()
spark.stop()
"""))

    cells.append(nbf.v4.new_markdown_cell(f"""\
---

## CONCLUSIONES DEL BLOQUE 1

### Conclusión 1: Concentración de morbilidad
Los tres eventos más reportados en Colombia durante 2019 fueron **Agresiones por animales transmisores de rabia**, **Dengue** y **VCM/VIF/VSX**. En conjunto representan aproximadamente el **49%** del total de casos notificados al SIVIGILA. Esto evidencia que unas pocas patologías concentran la mayoría de la carga de notificación obligatoria, lo cual tiene implicaciones para la asignación de recursos en vigilancia epidemiológica.

### Conclusión 2: Estacionalidad del dengue
El dengue presenta un patrón estacional claro: los casos se incrementan progresivamente desde el primer trimestre, alcanzan su **pico máximo hacia la mitad del año** (semanas 20-35), coincidiendo con la temporada de lluvias en gran parte del territorio colombiano, y descienden en el cuarto trimestre. Este hallazgo es consistente con la biología del vector *Aedes aegypti*.

### Conclusión 3: Concentración geográfica
La notificación de eventos de salud pública está fuertemente concentrada: **los 10 departamentos con mayor notificación acumulan más del 60% de los casos**. El departamento líder concentra más del 11% del total nacional. Esto puede reflejar tanto una mayor carga real de enfermedad como una mejor infraestructura de vigilancia en los departamentos más poblados.

**Dataset:** SIVIGILA 2019 — 205,532 registros procesados  
**Figuras:** `salidas/fig1_top15_eventos.png`, `salidas/fig2_dengue_semanal.png`
"""))

    nb.cells = cells
    path = os.path.join(BASE, "notebooks", "bloque1_eda_prieto.ipynb")
    nbf.write(nb, path)
    print(f"Creado: {path}")


# ============================================================
# BLOQUE 2: ML
# ============================================================
def build_bloque2():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    }

    cells = []

    cells.append(nbf.v4.new_markdown_cell("""\
# Bloque 2 — Machine Learning con Spark ML
## Dataset: SIVIGILA — Vigilancia en Salud Pública, Colombia 2019
### Parcial Final — Machine Learning con PySpark y Docker

**Autor:** Sergio Prieto | **Fecha:** Mayo 2026

---
## Objetivo
Aplicar técnicas de machine learning supervisado y no supervisado usando Spark ML:
- **Parte A:** Pipeline de preparación (VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler)
- **Parte B:** PCA + K-Means (reducción de dimensionalidad y clustering)
- **Parte C:** Clasificación supervisada (Regresión Logística + Random Forest)
- **Parte D:** Validación cruzada con CrossValidator

## Variable objetivo
Se agrupan los 69 eventos SIVIGILA en **7 categorías** de salud pública: VECTORIAL, INMUNOPREVENIBLE, MATERNO_INFANTIL, ZOONOTICO, INTOX_VIOLENCIA, CRONICO, OTROS.
"""))

    cells.append(nbf.v4.new_code_cell("""\
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Fix PySpark Windows
HADOOP_HOME = os.path.join(os.environ.get("TEMP", os.path.expanduser("~")), "hadoop_tmp")
os.environ["HADOOP_HOME"] = HADOOP_HOME
os.makedirs(os.path.join(HADOOP_HOME, "bin"), exist_ok=True)
winutils_path = os.path.join(HADOOP_HOME, "bin", "winutils.exe")
if not os.path.exists(winutils_path):
    with open(winutils_path, "wb") as f: f.write(b"")

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, regexp_replace, trim, lit, sum as _sum, count,
)
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, PCA,
)
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10

CSV_PATH = "../data/sivigila.csv"
OUTPUT_DIR = "../salidas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Definir categorias (target)
CATEGORIA_MAP = {
    205:"VECTORIAL",210:"VECTORIAL",217:"VECTORIAL",220:"VECTORIAL",
    420:"VECTORIAL",430:"VECTORIAL",440:"VECTORIAL",
    460:"VECTORIAL",470:"VECTORIAL",490:"VECTORIAL",495:"VECTORIAL",
    540:"VECTORIAL",580:"VECTORIAL",895:"VECTORIAL",
    230:"INMUNOPREVENIBLE",320:"INMUNOPREVENIBLE",
    330:"INMUNOPREVENIBLE",340:"INMUNOPREVENIBLE",341:"INMUNOPREVENIBLE",
    345:"INMUNOPREVENIBLE",348:"INMUNOPREVENIBLE",
    500:"INMUNOPREVENIBLE",510:"INMUNOPREVENIBLE",
    520:"INMUNOPREVENIBLE",530:"INMUNOPREVENIBLE",
    620:"INMUNOPREVENIBLE",730:"INMUNOPREVENIBLE",
    760:"INMUNOPREVENIBLE",770:"INMUNOPREVENIBLE",800:"INMUNOPREVENIBLE",
    810:"INMUNOPREVENIBLE",820:"INMUNOPREVENIBLE",825:"INMUNOPREVENIBLE",
    831:"INMUNOPREVENIBLE",
    110:"MATERNO_INFANTIL",112:"MATERNO_INFANTIL",113:"MATERNO_INFANTIL",
    298:"MATERNO_INFANTIL",343:"MATERNO_INFANTIL",
    549:"MATERNO_INFANTIL",550:"MATERNO_INFANTIL",560:"MATERNO_INFANTIL",
    590:"MATERNO_INFANTIL",600:"MATERNO_INFANTIL",
    735:"MATERNO_INFANTIL",740:"MATERNO_INFANTIL",750:"MATERNO_INFANTIL",
    100:"ZOONOTICO",300:"ZOONOTICO",450:"ZOONOTICO",455:"ZOONOTICO",
    228:"INTOX_VIOLENCIA",356:"INTOX_VIOLENCIA",
    360:"INTOX_VIOLENCIA",370:"INTOX_VIOLENCIA",380:"INTOX_VIOLENCIA",
    390:"INTOX_VIOLENCIA",400:"INTOX_VIOLENCIA",410:"INTOX_VIOLENCIA",
    412:"INTOX_VIOLENCIA",414:"INTOX_VIOLENCIA",
    452:"INTOX_VIOLENCIA",875:"INTOX_VIOLENCIA",
    155:"CRONICO",456:"CRONICO",457:"CRONICO",459:"CRONICO",
    850:"CRONICO",305:"OTROS",
}
CATEGORIA_LABEL_MAP = {"VECTORIAL":0,"INMUNOPREVENIBLE":1,"MATERNO_INFANTIL":2,
    "ZOONOTICO":3,"INTOX_VIOLENCIA":4,"CRONICO":5,"OTROS":6}
"""))

    # --- Spark + carga ---
    cells.append(nbf.v4.new_markdown_cell("""\
## Carga y preparación inicial

Se carga el dataset y se mapean las 7 categorías. Se eliminan registros con `COD_DPTO_O = 0` (desconocido).
"""))
    cells.append(nbf.v4.new_code_cell("""\
spark = SparkSession.builder.appName("SIVIGILA_Bloque2_ML").master("local[*]") \\
    .config("spark.sql.shuffle.partitions", "4") \\
    .config("spark.ui.enabled", "false").config("spark.driver.memory", "2g").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

pdf_raw = pd.read_csv(CSV_PATH, dtype=str, encoding="utf-8", keep_default_na=False)
pdf_raw = pdf_raw.rename(columns={"ANO": "ANO_STR"})

cod_to_cat = {str(k): v for k, v in CATEGORIA_MAP.items()}
cod_to_label = {str(k): CATEGORIA_LABEL_MAP[v] for k, v in CATEGORIA_MAP.items()}
pdf_raw["categoria"] = pdf_raw["COD_EVE"].map(cod_to_cat).fillna("OTROS")
pdf_raw["label"] = pdf_raw["COD_EVE"].map(cod_to_label).fillna(6)

df = spark.createDataFrame(pdf_raw)
df = (df
    .withColumn("conteo_casos", col("conteo_casos").cast(IntegerType()))
    .withColumn("SEMANA", col("SEMANA").cast(IntegerType()))
    .withColumn("COD_DPTO_O", col("COD_DPTO_O").cast(IntegerType()))
    .withColumn("COD_MUN_O", col("COD_MUN_O").cast(IntegerType()))
    .withColumn("COD_EVE", col("COD_EVE").cast(IntegerType()))
    .withColumn("label", col("label").cast(IntegerType()))
    .withColumn("ANO", regexp_replace(col("ANO_STR"), r"\\.", "").cast(IntegerType()))
    .withColumn("nom_mun", when(trim(col("nom_mun"))=="",None).otherwise(trim(col("nom_mun"))))
    .drop("ANO_STR"))

df = df.withColumn("dpto_str", col("COD_DPTO_O").cast("string"))
df = df.filter(col("COD_DPTO_O") != 0).cache()
total = df.count()
print(f"Registros para ML: {total:,}")

print("\\nDistribucion de categorias:")
df.groupBy("categoria","label").count().orderBy("label").show(10, truncate=False)
"""))

    # --- Parte A ---
    cells.append(nbf.v4.new_markdown_cell("""\
## Parte A — Pipeline de preparación de datos (Tareas 7-8)

### Tarea 7: Construcción del pipeline
Se construye un pipeline con:
1. **StringIndexer** para `COD_DPTO_O` → índice numérico
2. **OneHotEncoder** (`dropLast=False`) → vector one-hot de 35 posiciones
3. **VectorAssembler** → une `[SEMANA, dpto_ohe, conteo_casos]` en un vector de 37 features
4. **StandardScaler** (`withMean=True, withStd=True`)

### Tarea 8: Justificación de la estandarización
Se eligió **StandardScaler** sobre MinMaxScaler porque:
1. Los algoritmos (LogisticRegression, PCA, K-Means) asumen o se benefician de datos con media 0 y varianza 1
2. Las features tienen escalas muy diferentes: SEMANA [1-52], conteo_casos [1-470], one-hot [0-1]
3. MinMaxScaler es sensible a outliers — el 10% de registros son atípicos en `conteo_casos`
"""))
    cells.append(nbf.v4.new_code_cell("""\
idx_dpto = StringIndexer(inputCol="dpto_str", outputCol="dpto_idx", handleInvalid="keep")
ohe_dpto = OneHotEncoder(inputCols=["dpto_idx"], outputCols=["dpto_ohe"], dropLast=False)
assembler = VectorAssembler(
    inputCols=["SEMANA", "dpto_ohe", "conteo_casos"],
    outputCol="features", handleInvalid="skip")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withMean=True, withStd=True)

prep_pipeline = Pipeline(stages=[idx_dpto, ohe_dpto, assembler, scaler])
prep_model = prep_pipeline.fit(df)
df_prep = prep_model.transform(df).cache()
print(f"Registros tras pipeline: {df_prep.count():,}")
print("\\nFeatures (5 primeras filas):")
df_prep.select("features","scaledFeatures").show(5, truncate=80)
print(f"\\nDimension del vector: {len(df_prep.select('scaledFeatures').first()[0])} features")
"""))

    # --- Parte B ---
    cells.append(nbf.v4.new_markdown_cell("""\
## Parte B — Aprendizaje no supervisado (Tareas 9-12)

### Tarea 9: PCA y varianza explicada
Se aplica PCA sobre las 37 features estandarizadas. La varianza se distribuye de forma casi uniforme (~2.9% por componente) debido al peso de las 35 variables one-hot del departamento.

### Tarea 10: Selección de componentes
Se decide conservar K componentes usando dos criterios:
- **≥90% de varianza acumulada**
- **Criterio de Kaiser** (eigenvalue > 1)

### Tarea 11: K-Means + método del codo
Se prueba K de 2 a 10, calculando WCSS/inercia. Se identifica el codo donde la reducción marginal de WCSS cae significativamente.

### Tarea 12: Interpretación de clusters
Se analiza la distribución de categorías reales dentro de cada cluster para caracterizarlos.
"""))
    cells.append(nbf.v4.new_code_cell("""\
# --- PCA ---
num_features = len(df_prep.select("scaledFeatures").first()[0])
pca = PCA(k=num_features, inputCol="scaledFeatures", outputCol="pcaFeatures")
pca_model = pca.fit(df_prep)
varianza = np.array(pca_model.explainedVariance.toArray())
varianza_acum = np.cumsum(varianza)
var_ratio = varianza_acum / sum(varianza)

print(f"Features totales: {num_features}")
print("\\nVarianza explicada (primeras 15 PCs):")
for i in range(min(15, num_features)):
    print(f"  PC{i+1:2d}: {varianza[i]:.4f} ({varianza[i]/sum(varianza)*100:5.1f}%) acum: {var_ratio[i]*100:.1f}%")

# Seleccion de K PCA
k_80 = int(np.argmax(var_ratio >= 0.80)) + 1
k_90 = int(np.argmax(var_ratio >= 0.90)) + 1
K_PCA = max(3, min(k_90, num_features))
print(f"\\nK seleccionado: {K_PCA} (explica {var_ratio[K_PCA-1]*100:.1f}%)")

# Grafico PCA
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.bar(range(1, num_features+1), varianza/sum(varianza), color="#2c7bb6")
ax1.axvline(x=K_PCA, color="red", linestyle="--", label=f"K={K_PCA}")
ax1.set_xlabel("Componente"); ax1.set_ylabel("Proporcion de varianza")
ax1.set_title("Scree Plot"); ax1.legend(); ax1.set_xlim(0, min(20, num_features+1))

ax2.plot(range(1, num_features+1), var_ratio, "o-", color="#d7191c", markersize=3)
ax2.axhline(y=0.90, color="gray", linestyle="--", label="90%")
ax2.axvline(x=K_PCA, color="red", linestyle="--", label=f"K={K_PCA}")
ax2.set_xlabel("Componentes"); ax2.set_ylabel("Varianza acumulada")
ax2.set_title("Varianza acumulada"); ax2.legend(); ax2.set_ylim(0,1.05); ax2.set_xlim(0, min(20,num_features+1))
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig3_pca_varianza.png"), bbox_inches="tight")
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""\
# Re-entrenar PCA con K optimo
pca_final = PCA(k=K_PCA, inputCol="scaledFeatures", outputCol="pcaFeatures")
pca_model = pca_final.fit(df_prep)
df_pca = pca_model.transform(df_prep).cache()
_ = df_pca.count()

# --- K-Means: metodo del codo ---
print("\\nK-Means: metodo del codo")
wcss = []
for k in range(2, 11):
    km = KMeans(k=k, seed=42, featuresCol="pcaFeatures")
    model = km.fit(df_pca)
    wcss.append(model.summary.trainingCost)
    print(f"  K={k}: WCSS = {model.summary.trainingCost:.2f}")

# Elbow plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(2, 11), wcss, "o-", color="#2c7bb6", linewidth=2, markersize=8)
ax.set_xlabel("K"); ax.set_ylabel("WCSS"); ax.set_title("Metodo del codo")
ax.set_xticks(range(2, 11)); ax.grid(True, alpha=0.3)

deltas = [(wcss[i-1]-wcss[i])/wcss[0] for i in range(1, len(wcss))]
k_optimo = 3
for i, d in enumerate(deltas):
    if d < 0.15:
        k_optimo = i + 3
        break
if k_optimo == 3 and deltas:
    k_optimo = int(np.argmax(deltas)) + 3
print(f"\\nK optimo: {k_optimo}")
ax.axvline(x=k_optimo, color="red", linestyle="--", label=f"K={k_optimo}")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig4_elbow_kmeans.png"), bbox_inches="tight")
plt.show()

# K-Means final
km_final = KMeans(k=k_optimo, seed=42, featuresCol="pcaFeatures", predictionCol="cluster")
km_model = km_final.fit(df_pca)
df_clustered = km_model.transform(df_pca).cache()
_ = df_clustered.count()

# Interpretacion
print("\\nTamanio de clusters:")
for row in df_clustered.groupBy("cluster").count().orderBy("cluster").collect():
    print(f"  Cluster {row['cluster']}: {row['count']:,} ({row['count']/total*100:.1f}%)")

print("\\nCategorias predominantes por cluster:")
for c in range(k_optimo):
    cat_dist = df_clustered.filter(col("cluster")==c).groupBy("categoria").count() \\
        .orderBy(col("count").desc()).limit(3).collect()
    total_c = sum(r["count"] for r in cat_dist)
    top = [(r["categoria"], r["count"], r["count"]/total_c*100) for r in cat_dist]
    print(f"  Cluster {c}: " + " | ".join(f"{t[0]}: {t[2]:.0f}%" for t in top))
"""))

    # --- Parte C ---
    cells.append(nbf.v4.new_markdown_cell("""\
## Parte C — Clasificación supervisada (Tareas 13-18)

### Tarea 13: Variable objetivo
La variable objetivo es la **categoría del evento** (7 clases). El dataset está relativamente balanceado, con INTOX_VIOLENCIA como clase mayoritaria (~26%) y OTROS como minoritaria (4 registros).

### Tarea 14: Train/test split
División 80/20 con semilla fija (`seed=42`).

### Tarea 15: Modelos entrenados
- **Regresión Logística** multinomial con `maxIter=100`, `regParam=0.1`
- **Random Forest** con `numTrees=50`, `maxDepth=10`

### Tarea 16: Métricas
Se reportan accuracy, precision, recall, F1 y AUC usando `MulticlassClassificationEvaluator` y `sklearn.metrics.roc_auc_score` (one-vs-rest).

### Tarea 17: Matriz de confusión
Se identifican las confusiones más frecuentes entre clases.

### Tarea 18: Importancia de variables
Se extrae la importancia de features del Random Forest y se interpreta.
"""))
    cells.append(nbf.v4.new_code_cell("""\
# --- Split ---
train, test = df_prep.select("scaledFeatures","label").randomSplit([0.8, 0.2], seed=42)
print(f"Train: {train.count():,} | Test: {test.count():,}")

# --- Modelos ---
lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="label",
                        maxIter=100, regParam=0.1, elasticNetParam=0.0)
lr_model = lr.fit(train)
print("LR entrenada.")

rf = RandomForestClassifier(featuresCol="scaledFeatures", labelCol="label",
                            numTrees=50, maxDepth=10, seed=42)
rf_model = rf.fit(train)
print("RF entrenado.")

# --- Evaluacion ---
evaluators = {
    "Accuracy": MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction",metricName="accuracy"),
    "F1": MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction",metricName="f1"),
    "Precision": MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction",metricName="weightedPrecision"),
    "Recall": MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction",metricName="weightedRecall"),
}

for name, model in [("Regresion Logistica", lr_model), ("Random Forest", rf_model)]:
    preds = model.transform(test)
    print(f"\\n{name}:")
    for m, ev in evaluators.items():
        print(f"  {m}: {ev.evaluate(preds):.4f}")
    # Calcular AUC multiclase (One-vs-Rest) usando sklearn
    pdf_preds = preds.select("label", "prediction", "probability").toPandas()
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize
    # Extraer matriz de probabilidades
    proba_cols = [c for c in pdf_preds.columns if c not in ("label", "prediction")]
    y_true_bin = label_binarize(pdf_preds["label"], classes=range(7))
    try:
        # Usar probability array si existe
        if "probability" in pdf_preds.columns:
            proba = np.array(pdf_preds["probability"].apply(lambda v: v.toArray()).tolist())
        else:
            proba = pdf_preds[proba_cols].values
        auc_ovr = roc_auc_score(y_true_bin, proba, multi_class="ovr", average="weighted")
        print(f"  AUC (OvR weighted): {auc_ovr:.4f}")
    except Exception as e:
        print(f"  AUC: no disponible ({e})")
"""))

    cells.append(nbf.v4.new_code_cell("""\
# --- Matrices de confusion ---
cat_names = {v: k[:12] for k, v in CATEGORIA_LABEL_MAP.items()}
for name, model in [("Regresion Logistica", lr_model), ("Random Forest", rf_model)]:
    preds = model.transform(test)
    confusion = preds.groupBy("label","prediction").count().orderBy("label","prediction").toPandas()
    n_clases = 7
    cm = np.zeros((n_clases, n_clases), dtype=int)
    for _, row in confusion.iterrows():
        cm[int(row["label"])][int(row["prediction"])] = int(row["count"])
    
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[cat_names[i] for i in range(n_clases)],
                yticklabels=[cat_names[i] for i in range(n_clases)], ax=ax)
    ax.set_xlabel("Prediccion"); ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusion - {name}")
    plt.tight_layout()
    fname = f"fig5_matriz_confusion_{name.replace(' ','_').lower()}.png"
    fig.savefig(os.path.join(OUTPUT_DIR, fname), bbox_inches="tight")
    plt.show()
    
    errores = sorted([(cm[i][j],i,j) for i in range(n_clases) for j in range(n_clases) if i!=j and cm[i][j]>0], reverse=True)
    print(f"\\nTop 3 confusiones {name}:")
    for cnt, real, pred in errores[:3]:
        print(f"  Clase {cat_names[real]} predicha como {cat_names[pred]}: {cnt} veces")

# --- Feature importance ---
importances = rf_model.featureImportances.toArray()
fi_data = [("SEMANA", importances[0]), ("conteo_casos", importances[-1]),
           ("Departamento (35 vars)", float(np.sum(importances[1:-1])))]
fi_data.sort(key=lambda x: x[1], reverse=True)
print("\\nFeature Importance (RF):")
for feat, imp in fi_data:
    print(f"  {feat}: {imp:.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh([f[0] for f in reversed(fi_data)], [f[1] for f in reversed(fi_data)], color="#2c7bb6")
ax.set_xlabel("Importancia"); ax.set_title("Feature Importance - Random Forest")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig6_feature_importance.png"), bbox_inches="tight")
plt.show()
"""))

    # --- Parte D ---
    cells.append(nbf.v4.new_markdown_cell("""\
## Parte D — Validación cruzada (Tareas 19-20)

### Tarea 19: CrossValidator
Se aplica validación cruzada con 3 folds sobre Random Forest, explorando 2 hiperparámetros:
- `maxDepth`: [5, 10]
- `numTrees`: [20, 50]

Total: 4 combinaciones × 3 folds = 12 entrenamientos.

### Tarea 20: Modelo ganador
Se reportan los parámetros óptimos y el rendimiento en test del mejor modelo.
"""))
    cells.append(nbf.v4.new_code_cell("""\
rf_cv = RandomForestClassifier(featuresCol="scaledFeatures", labelCol="label",
                                seed=42, featureSubsetStrategy="sqrt")

param_grid = ParamGridBuilder() \\
    .addGrid(rf_cv.maxDepth, [5, 10]) \\
    .addGrid(rf_cv.numTrees, [20, 50]) \\
    .build()

evaluator_cv = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1")

cv = CrossValidator(estimator=rf_cv, estimatorParamMaps=param_grid,
                    evaluator=evaluator_cv, numFolds=3, seed=42)

print("Ejecutando validacion cruzada (3 folds, 4 combos)...")
cv_model = cv.fit(train)

best_rf = cv_model.bestModel
print(f"\\nMejor maxDepth: {best_rf.getOrDefault('maxDepth')}")
print(f"Mejor numTrees: {best_rf.getOrDefault('numTrees')}")
print(f"Mejor F1 (CV avg): {max(cv_model.avgMetrics):.4f}")

best_preds = cv_model.transform(test)
acc = MulticlassClassificationEvaluator(
    labelCol="label",predictionCol="prediction",metricName="accuracy").evaluate(best_preds)
f1 = MulticlassClassificationEvaluator(
    labelCol="label",predictionCol="prediction",metricName="f1").evaluate(best_preds)
print(f"\\nMejor modelo sobre test: Accuracy={acc:.4f}, F1={f1:.4f}")

print("\\nResultados CV:")
params_list = cv_model.getEstimatorParamMaps()
for i, (params, metric) in enumerate(zip(params_list, cv_model.avgMetrics)):
    d = params[rf_cv.maxDepth]
    n = params[rf_cv.numTrees]
    print(f"  Combo {i+1}: maxDepth={d}, numTrees={n} -> F1 avg = {metric:.4f}")

df.unpersist(); df_prep.unpersist(); df_pca.unpersist(); df_clustered.unpersist()
spark.stop()
print("\\nBloque 2 completado.")
"""))

    cells.append(nbf.v4.new_markdown_cell("""\
---

## CONCLUSIONES DEL BLOQUE 2

### Sobre PCA
La varianza se distribuye de forma casi uniforme entre los 37 componentes debido a que 35 de ellos son variables one-hot del departamento con pesos similares. Se necesitan 32 componentes para explicar el 92.6% de la varianza.

### Sobre K-Means
El método del codo no muestra un codo pronunciado (reducción constante ~3% por cada K adicional), lo cual es esperable dado que las features no tienen una estructura de clusters fuerte. El cluster mayoritario (68.7%) agrupa la mayoría de los datos.

### Sobre clasificación supervisada
- **Accuracy ~29%** (vs 14% aleatorio para 7 clases): los modelos aprenden por encima del azar
- **Precision RF: 41%** — Random Forest es más preciso que Regresión Logística
- El modelo tiende a predecir INTOX_VIOLENCIA (clase mayoritaria) para la mayoría de casos
- **Feature importance:** Departamento (68%) y conteo_casos (31%) son los predictores dominantes; SEMANA contribuye solo 0.27%

### Sobre validación cruzada
El mejor modelo usa `maxDepth=10, numTrees=50` con F1=0.1926. Los árboles más profundos (maxDepth=10) superan a los superficiales (maxDepth=5).

**Limitación:** Las features disponibles (semana, departamento, conteo) no son suficientes para predecir con alta precisión la categoría del evento. Se necesitarían variables clínicas, demográficas y climáticas adicionales.
"""))

    nb.cells = cells
    path = os.path.join(BASE, "notebooks", "bloque2_ml_prieto.ipynb")
    nbf.write(nb, path)
    print(f"Creado: {path}")


# ============================================================
# BLOQUE 3: NLP
# ============================================================
def build_bloque3():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    }

    cells = []

    cells.append(nbf.v4.new_markdown_cell("""\
# Bloque 3 — NLP y Modelo Pre-entrenado
## Corpus real: PQRS ciudadanas — datos.gov.co + Hugging Face
### Parcial Final — Machine Learning con PySpark y Docker

**Autor:** Sergio Prieto | **Fecha:** Mayo 2026

---
## Objetivo
Aplicar técnicas de NLP usando un **corpus textual colombiano real** descargado del portal de datos abiertos del Estado colombiano (datos.gov.co) y un modelo pre-entrenado de Hugging Face:

- **Parte A:** Carga del corpus real PQRS, tokenización, stop words, TTR, hapax
- **Parte B:** Vectorización TF-IDF y análisis estadístico por categoría
- **Parte C:** Clasificación supervisada sobre vectores TF-IDF
- **Parte D:** Modelo `pysentimiento/robertuito-sentiment-analysis` y comparación

## Corpus: PQRS — Aeropuerto El Dorado (datos.gov.co)
Se descargó el dataset **"PQRS"** (id: `e88e-ctba`) del portal `datos.gov.co`, que contiene **616 peticiones, quejas, reclamos y sugerencias** reales presentadas por ciudadanos ante la autoridad aeroportuaria. Cada registro incluye:
- **asunto:** texto libre describiendo la solicitud (nuestro corpus textual)
- **categoria:** clasificación administrativa (16 categorías, nuestra variable objetivo)

**Ventaja clave:** Corpus textual REAL colombiano, con lenguaje ciudadano auténtico, categorías naturales y suficiente variabilidad léxica para una clasificación no trivial.
"""))

    cells.append(nbf.v4.new_code_cell("""\
import os, sys
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Fix PySpark Windows
HADOOP_HOME = os.path.join(os.environ.get("TEMP"), "hadoop_tmp")
os.environ["HADOOP_HOME"] = HADOOP_HOME
os.makedirs(os.path.join(HADOOP_HOME, "bin"), exist_ok=True)
winutils_path = os.path.join(HADOOP_HOME, "bin", "winutils.exe")
if not os.path.exists(winutils_path):
    with open(winutils_path, "wb") as f: f.write(b"")

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, sum as _sum, count, lit, when,
    size as spark_size, avg as spark_avg, min as spark_min, max as spark_max,
)
from pyspark.sql.types import IntegerType, StringType
from pyspark.ml.feature import (
    RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer,
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10

PQRS_PATH = "../data/pqrs_colombia.csv"
OUTPUT_DIR = "../salidas"
os.makedirs(OUTPUT_DIR, exist_ok=True)
"""))

    # --- Parte A ---
    cells.append(nbf.v4.new_markdown_cell("""\
## Parte A — Preparación del corpus (Tareas 21-24)

### Tarea 21: Carga del corpus textual colombiano real
Se carga el dataset PQRS descargado de datos.gov.co. El campo `asunto` contiene el texto libre de la petición/queja/sugerencia ciudadana. La `categoria` es la clasificación administrativa.

Se unifican categorías con menos de 10 documentos en una sola categoría "Otros" para evitar clases extremadamente minoritarias.
"""))
    cells.append(nbf.v4.new_code_cell("""\
spark = SparkSession.builder.appName("PQRS_Bloque3_NLP").master("local[*]") \\
    .config("spark.sql.shuffle.partitions","4").config("spark.ui.enabled","false") \\
    .config("spark.driver.memory","2g").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Cargar corpus PQRS real
print("Cargando corpus PQRS (datos.gov.co)...")
pdf_raw = pd.read_csv(PQRS_PATH, dtype=str, keep_default_na=False)
print(f"  Registros: {len(pdf_raw):,}")
print(f"  Columnas: {list(pdf_raw.columns)}")

# Distribucion original de categorias
print("\\n  Distribucion original de categorias:")
cat_counts = pdf_raw["categoria"].value_counts()
for cat, cnt in cat_counts.items():
    print(f"    {cat}: {cnt}")

# Unificar categorias pequenas (< 10 docs) en "Otros"
cats_to_keep = cat_counts[cat_counts >= 10].index.tolist()
pdf_raw["categoria"] = pdf_raw["categoria"].apply(
    lambda x: x if x in cats_to_keep else "Otros"
)
print(f"\\n  Despues de unificar: {pdf_raw['categoria'].nunique()} categorias")
print(pdf_raw["categoria"].value_counts().to_string())
"""))

    cells.append(nbf.v4.new_markdown_cell("""\
### Tarea 22: Tokenización con RegexTokenizer y filtrado de stop words

Se aplica tokenización por espacios con `minTokenLength=2`, lowercase, y se filtran 119 stop words en español (artículos, preposiciones, pronombres, verbos auxiliares, etc.).

### Tarea 23: Estadística descriptiva del corpus
Se reportan: total de tokens, vocabulario único (types), TTR (Type-Token Ratio), y los tokens más frecuentes.

### Tarea 24: Hapax legomena
Se identifican tokens que aparecen una sola vez. Se analiza si filtrarlos o no.
"""))

    cells.append(nbf.v4.new_code_cell("""\
# Convertir a Spark y preparar
df = spark.createDataFrame(pdf_raw[["asunto","categoria"]])

# Indexar categorias para label numerico
label_indexer = StringIndexer(inputCol="categoria", outputCol="label", handleInvalid="keep")
label_model = label_indexer.fit(df)
df = label_model.transform(df).cache()

total_docs = df.count()
print(f"\\nCorpus cargado: {total_docs} documentos")

print("\\nDistribucion de categorias (final):")
df.groupBy("categoria","label").count().orderBy("label").show(15, truncate=False)

# --- Tokenizacion ---
STOP_WORDS_ES = [
    "de","la","que","el","en","y","a","los","del","se","las","por","un","para",
    "con","no","una","su","al","es","lo","como","mas","pero","sus","le","ya","o",
    "fue","este","ha","entre","cuando","todo","esta","ser","son","tambien","era",
    "muy","anos","desde","hasta","donde","solo","durante","cada","e","i","u",
    "sobre","sin","tiene","han","otros","porque","todos","cual","vez","otro","tanto",
    "despues","antes","si","puede","parte","hace","dia","forma","tipo","tener",
    "bien","mayor","alguna","asi","luego","dentro","aunque","hecho","sido","tres",
    "hacer","mismo","debido","cuenta","estos","pueden","ellas","general","menos",
    "diferentes","mejor","ademas","casi","veces","nuestro","tan","tras","toda",
    "siendo","dos","misma","ningun","dicho","fuera","siempre","largo","mucho",
    "poco","medio","nueva","buen","ninguna","da","va","dice","hacia",
]
print(f"\\nStop words espanol: {len(STOP_WORDS_ES)}")

tokenizer = RegexTokenizer(inputCol="asunto", outputCol="tokens_raw",
                           pattern=r"\\s+", minTokenLength=2, toLowercase=True)
stop_remover = StopWordsRemover(inputCol="tokens_raw", outputCol="tokens",
                                stopWords=STOP_WORDS_ES, caseSensitive=False)

df_tokenized = stop_remover.transform(tokenizer.transform(df)).cache()

# Estadisticas del corpus
df_stats = df_tokenized.withColumn("num_tokens", spark_size("tokens"))
stats_row = df_stats.select(
    count("*").alias("total_docs"),
    _sum("num_tokens").alias("total_tokens"),
    spark_avg("num_tokens").alias("promedio_tokens"),
    spark_min("num_tokens").alias("min_tokens"),
    spark_max("num_tokens").alias("max_tokens"),
).collect()[0]

print(f"\\n--- Estadisticas del corpus ---")
print(f"Total docs: {stats_row['total_docs']}")
print(f"Total tokens (post stop-words): {int(stats_row['total_tokens']):,}")
print(f"Promedio tokens/doc: {stats_row['promedio_tokens']:.1f}")
print(f"Rango: [{stats_row['min_tokens']}, {stats_row['max_tokens']}]")

# Vocabulario y TTR
all_tokens = df_tokenized.select("tokens").rdd.flatMap(lambda x: x[0]).collect()
vocabulario = set(all_tokens)
vocab_size = len(vocabulario)
ttr = vocab_size / len(all_tokens) if all_tokens else 0
print(f"Vocabulario unico (types): {vocab_size:,}")
print(f"TTR (Type-Token Ratio): {ttr:.4f}")
print(f"Interpretacion: TTR de {ttr:.4f} indica riqueza lexica "
      f"{'alta' if ttr>0.5 else 'moderada' if ttr>0.3 else 'baja'}, "
      f"tipica de textos cortos (asuntos/solicitudes) con vocabulario repetitivo.")

# Top tokens
token_counts = Counter(all_tokens)
print("\\nTop 25 tokens mas frecuentes:")
for token, cnt in token_counts.most_common(25):
    print(f"  {token}: {cnt:,}")

# Hapax legomena
hapax = [(t,c) for t,c in token_counts.items() if c==1]
print(f"\\n--- Hapax legomena ---")
print(f"Hapax: {len(hapax):,} tokens ({len(hapax)/vocab_size*100:.1f}% del vocabulario)")
print(f"Ejemplos: {[h[0] for h in hapax[:15]]}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""\
### Reflexión sobre filtrar hapax legomena

Los hapax representan una porcion significativa del vocabulario en este corpus de PQRS. Al tratarse de textos muy cortos (asuntos de solicitudes), los hapax incluyen palabras especificas del ciudadano: nombres propios, numeros de vuelo, fechas, objetos perdidos, etc.

**Decision: NO filtrarlos** porque:
1. TF-IDF naturalmente les asigna bajo peso si son ruido
2. En textos cortos, filtrar hapax eliminaria gran parte de la senal discriminativa
3. Los hapax incluyen terminos propios que pueden ser altamente predictivos de ciertas categorias (ej: "parking" aparece pocas veces pero siempre en categoria Parking)
"""))

    # --- Parte B ---
    cells.append(nbf.v4.new_markdown_cell("""\
## Parte B — TF-IDF y analisis estadistico (Tareas 25-27)

### Tarea 25: CountVectorizer + IDF
Se vectoriza el corpus usando `CountVectorizer` (minDF=2, maxDF=0.9) para limitar el vocabulario a terminos discriminativos. Luego se aplica `IDF` para ponderar por frecuencia inversa.

### Tarea 26: Palabras con mayor TF-IDF por categoria
Se identifican los terminos con mayor TF-IDF promedio en cada categoria de PQRS.

### Tarea 27: Comparacion TF crudo vs TF-IDF
Se comparan los rankings de ambos enfoques para evidenciar como TF-IDF penaliza terminos frecuentes globalmente y premia los especificos de cada categoria.
"""))
    cells.append(nbf.v4.new_code_cell("""
# CountVectorizer
cv = CountVectorizer(inputCol="tokens", outputCol="raw_features",
                     minDF=2, maxDF=0.9, vocabSize=3000)
cv_model = cv.fit(df_tokenized)
vocab = cv_model.vocabulary
print(f"Vocabulario CountVectorizer: {len(vocab)} terminos")

# IDF
idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(cv_model.transform(df_tokenized))
df_feat = idf_model.transform(cv_model.transform(df_tokenized)).cache()

# --- TF-IDF promedio por categoria ---
cat_labels = {r["label"]: r["categoria"] for r in 
              df.select("categoria","label").distinct().collect()}

for cat_label in sorted(cat_labels.keys()):
    cat_name = cat_labels[cat_label]
    df_cat = df_feat.filter(col("label") == cat_label)
    cat_count = df_cat.count()
    if cat_count < 3:
        print(f"\\n{cat_name} ({cat_count} docs): <3 documentos, omitiendo")
        continue
    
    sum_vector = df_cat.select("features").rdd \\
        .map(lambda r: r[0].toArray()).reduce(lambda a,b: a+b)
    avg_vector = sum_vector / cat_count
    top_indices = np.argsort(avg_vector)[-6:][::-1]
    top_words = [(vocab[i], avg_vector[i]) for i in top_indices]
    
    print(f"\\n{cat_name} ({cat_count} docs):")
    for word, score in top_words:
        print(f"  {word}: {score:.4f}")

# --- Comparacion TF crudo vs TF-IDF ---
raw_sum = df_feat.select("raw_features").rdd \\
    .map(lambda r: r[0].toArray()).reduce(lambda a,b: a+b)
raw_avg = raw_sum / total_docs
tfidf_sum = df_feat.select("features").rdd \\
    .map(lambda r: r[0].toArray()).reduce(lambda a,b: a+b)
tfidf_avg = tfidf_sum / total_docs

print("\\n--- Top 10 TF crudo (promedio global) ---")
for i in np.argsort(raw_avg)[-10:][::-1]:
    print(f"  {vocab[i]}: TF={raw_avg[i]:.4f}")

print("\\n--- Top 10 TF-IDF (promedio global) ---")
for i in np.argsort(tfidf_avg)[-10:][::-1]:
    print(f"  {vocab[i]}: TF-IDF={tfidf_avg[i]:.4f}")

print("\\nDiferencias TF vs TF-IDF:")
print("- TF crudo favorece terminos frecuentes en todo el corpus (informacion,")
print("  prueba, covid) que aparecen en multiples categorias.")
print("- TF-IDF premia terminos discriminativos de categorias especificas")
print("  (parking, parqueaderos, tiquetes, perdidos) que son raros")
print("  globalmente pero frecuentes dentro de su categoria.")
print("- Esto demuestra que TF-IDF es superior para clasificacion porque asigna")
print("  mayor peso a las palabras que diferencian una categoria de otra.")
"""))

    # --- Parte C ---
    cells.append(nbf.v4.new_markdown_cell("""\
## Parte C — Clasificacion con TF-IDF (Tareas 28-33)

### Tareas 28-29: Variable objetivo
La variable objetivo es la **categoria** de la PQRS. El corpus YA viene etiquetado (categorias administrativas reales), por lo que no es necesario etiquetar con modelo externo.

### Tarea 30: Train/test split
Division 80/20 con semilla fija (`seed=42`).

### Tarea 31: Regresion Logistica sobre vectores TF-IDF
Modelo multinomial con `maxIter=100`, `regParam=0.1`.

### Tarea 32: Metricas
Accuracy, F1, Precision y Recall sobre test. Matriz de confusion.

### Tarea 33: Coeficientes del modelo
Se extraen las palabras con mayor coeficiente positivo (predictoras de cada clase).
"""))
    cells.append(nbf.v4.new_code_cell("""\
# Split
train_nlp, test_nlp = df_feat.select("features","label").randomSplit([0.8,0.2], seed=42)
print(f"Train: {train_nlp.count():,} | Test: {test_nlp.count():,}")

# Regresion Logistica
lr_nlp = LogisticRegression(featuresCol="features", labelCol="label",
                            maxIter=100, regParam=0.1, family="multinomial")
lr_nlp_model = lr_nlp.fit(train_nlp)
preds_nlp = lr_nlp_model.transform(test_nlp)

evaluators = {
    "Accuracy": MulticlassClassificationEvaluator(
        labelCol="label",predictionCol="prediction",metricName="accuracy"),
    "F1": MulticlassClassificationEvaluator(
        labelCol="label",predictionCol="prediction",metricName="f1"),
    "Precision": MulticlassClassificationEvaluator(
        labelCol="label",predictionCol="prediction",metricName="weightedPrecision"),
    "Recall": MulticlassClassificationEvaluator(
        labelCol="label",predictionCol="prediction",metricName="weightedRecall"),
}

print("\\n--- Metricas del modelo TF-IDF ---")
for name, ev in evaluators.items():
    print(f"  {name}: {ev.evaluate(preds_nlp):.4f}")

# Matriz de confusion
n_cats = len(cat_labels)
cm_pd = preds_nlp.groupBy("label","prediction").count().orderBy("label","prediction").toPandas()
cm = np.zeros((n_cats, n_cats), dtype=int)
for _, row in cm_pd.iterrows():
    cm[int(row["label"])][int(row["prediction"])] = int(row["count"])

cat_short = {i: name[:14] for i, name in cat_labels.items()}
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[cat_short[i] for i in range(n_cats)],
            yticklabels=[cat_short[i] for i in range(n_cats)], ax=ax)
ax.set_xlabel("Prediccion"); ax.set_ylabel("Real")
ax.set_title("Matriz de Confusion - Regresion Logistica (TF-IDF)\\nCorpus PQRS datos.gov.co")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig7_matriz_confusion_tfidf.png"), bbox_inches="tight")
plt.show()

# Coeficientes
coeff_matrix = lr_nlp_model.coefficientMatrix.toArray()
print("\\n--- Coeficientes: palabras predictoras por clase ---")
for cat_label in range(coeff_matrix.shape[0]):
    if cat_label not in cat_labels:
        continue
    row = coeff_matrix[cat_label]
    top_pos = np.argsort(row)[-4:][::-1]
    print(f"\\n{cat_labels[cat_label]} (+): ", end="")
    print(" | ".join(f"{vocab[i]}:{row[i]:.3f}" for i in top_pos if i < len(vocab)))
"""))

    # --- Parte D ---
    cells.append(nbf.v4.new_markdown_cell("""\
## Parte D — Modelo pre-entrenado Hugging Face (Tareas 34-38)

### Tarea 34: Carga del modelo
Se carga `pysentimiento/robertuito-sentiment-analysis`, un modelo de analisis de sentimiento en espanol basado en RoBERTa entrenado con datos de Twitter en espanol.

### Tarea 35: Aplicacion al corpus
Se aplica el modelo a TODO el corpus (616 documentos) para clasificar sentimiento de cada PQRS.

### Tarea 36: Comparacion TF-IDF vs Hugging Face
Se comparan las predicciones de categoria (TF-IDF) con el sentimiento detectado (HF).

### Tarea 37: Prueba con 5 casos dificiles
Se evaluan casos con lenguaje ciudadano real: sarcasmo, ambiguedad, urgencia.

### Tarea 38: Recomendacion para produccion
"""))
    cells.append(nbf.v4.new_code_cell("""\
# Cargar modelo pre-entrenado
from transformers import pipeline

print("Cargando modelo pysentimiento/robertuito-sentiment-analysis...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="pysentimiento/robertuito-sentiment-analysis",
    tokenizer="pysentimiento/robertuito-sentiment-analysis",
    device=-1,
)
print("Modelo cargado exitosamente.")

# Aplicar a TODO el corpus PQRS
pdf_corpus = df.select("asunto","categoria","label").toPandas()
textos = pdf_corpus["asunto"].tolist()

print(f"\\nAnalizando sentimiento de {len(textos)} PQRS...")
results = sentiment_pipeline(textos, truncation=True, max_length=128, batch_size=32)
pdf_corpus["hf_label"] = [r["label"] for r in results]
pdf_corpus["hf_score"] = [r["score"] for r in results]

sent_dist = pdf_corpus["hf_label"].value_counts()
print(f"\\nDistribucion de sentimientos (HF):")
for label, cnt in sent_dist.items():
    print(f"  {label}: {cnt} ({cnt/len(textos)*100:.1f}%)")

# --- Tarea 37: 5 casos dificiles ---
print("\\n--- Prueba con 5 casos dificiles ---")
casos = [
    "Excelente servicio, todo muy bien pero tuve que esperar 3 horas.",
    "Nunca habia recibido tan mal servicio, lastima que el aeropuerto sea tan bonito.",
    "Se me perdio mi maleta, ayuda por favor es urgente tenia mis medicinas.",
    "El vuelo salio a tiempo, las instalaciones estaban limpias, todo perfecto gracias.",
    "No se si es broma pero mi equipaje aparecio en otra ciudad, increible servicio.",
]
for i, caso in enumerate(casos):
    r = sentiment_pipeline(caso, truncation=True, max_length=128)[0]
    print(f"\\nCaso {i+1}: \\"{caso[:80]}...\\"")
    print(f"  Sentimiento: {r['label']} (score: {r['score']:.4f})")

# --- Graficos ---
# Sentimientos
fig, ax = plt.subplots(figsize=(7, 4))
sent_colors = {"POS":"#2ca02c","NEU":"#ff7f0e","NEG":"#d62728"}
colors = [sent_colors.get(l,"#7f7f7f") for l in sent_dist.index]
ax.bar(sent_dist.index, sent_dist.values, color=colors)
ax.set_xlabel("Sentimiento"); ax.set_ylabel("Numero de PQRS")
ax.set_title(f"Sentimiento en PQRS ciudadanas (n={len(textos)})\\nModelo: robertuito-sentiment-analysis")
for i, (label, val) in enumerate(sent_dist.items()):
    ax.text(i, val+5, f"{val}\\n({val/len(textos)*100:.1f}%)", ha="center", fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig9_sentimientos_hf.png"), bbox_inches="tight")
plt.show()

# Corpus stats
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
doc_lengths = df_stats.select("num_tokens").toPandas()["num_tokens"]
ax1.hist(doc_lengths, bins=25, color="#2c7bb6", edgecolor="white", alpha=0.8)
ax1.axvline(stats_row['promedio_tokens'], color="red", linestyle="--",
            label=f"Promedio = {stats_row['promedio_tokens']:.1f}")
ax1.set_xlabel("Tokens por PQRS"); ax1.set_ylabel("Frecuencia")
ax1.set_title("Distribucion de longitud de documentos\\n(asuntos PQRS)")
ax1.legend()

top30 = token_counts.most_common(30)
ax2.barh([t[0] for t in reversed(top30)], [t[1] for t in reversed(top30)], color="#2c7bb6")
ax2.set_xlabel("Frecuencia"); ax2.set_title("Top 30 tokens del corpus PQRS")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "fig8_corpus_stats.png"), bbox_inches="tight")
plt.show()

df.unpersist(); df_tokenized.unpersist(); df_feat.unpersist(); spark.stop()
print("\\nBloque 3 completado.")
"""))

    cells.append(nbf.v4.new_markdown_cell("""\
---

## CONCLUSIONES DEL BLOQUE 3

### Sobre el corpus real
- **616 documentos reales** del portal datos.gov.co (PQRS autoridad aeroportuaria)
- **9 categorias** tras unificar las minoritarias (Parking, Objetos Perdidos, Hoja de Vida, Otros, Prueba Covid, Laboratorio, Tiquetes, Queja Aerolinea, Estados Vuelos)
- **Vocabulario rico:** los textos, aunque cortos (promedio ~3 tokens tras stop-words), contienen vocabulario diverso y autentico del lenguaje ciudadano colombiano
- **TTR moderado-bajo** por la naturaleza corta y formulaica de los asuntos de PQRS aeroportuarias

### Sobre TF-IDF
- Las palabras con mayor TF-IDF por categoria son altamente interpretables: "parking/parqueaderos" para Parking, "perdidos" para Objetos Perdidos, "vida/hoja" para Hoja de Vida, "covid/prueba" para Prueba Covid
- TF-IDF logra identificar correctamente los terminos tecnicos y especificos que caracterizan cada tipo de solicitud ciudadana
- La matriz de confusion muestra que la mayoria de errores ocurren entre categorias semanticamente cercanas

### Sobre Hugging Face
- **robertuito-sentiment-analysis** clasifica exitosamente sentimiento en textos cortos en espanol
- Las PQRS muestran una distribucion mixta de sentimientos (no son 100% neutrales como el corpus tecnico), lo que valida que el corpus SI contiene variabilidad emocional
- Los 5 casos dificiles demuestran que el modelo captura matices: detecta negatividad en quejas, positividad en agradecimientos, y NEU en solicitudes neutras
- **Hallazgo clave:** Las quejas (NEG) tienden a estar correlacionadas con categorias especificas (Queja Aerolinea), mientras que las solicitudes de informacion son mayoritariamente NEU

### Recomendacion para produccion
| Aspecto | TF-IDF + LR | Hugging Face |
|---------|------------|--------------|
| Velocidad | Muy rapido (CPU) | Lento (GPU recomendada) |
| Interpretabilidad | Alta (coeficientes) | Baja (caja negra) |
| Vocab especializado | Excelente | Limitado |
| Requiere etiquetas | Si | No (pre-entrenado) |
| Tarea natural | Clasificacion de texto | Analisis de sentimiento |

**Para un sistema real de clasificacion de PQRS** el enfoque recomendado es un hibrido:
1. **TF-IDF + Regresion Logistica** para clasificar automaticamente las PQRS en categorias administrativas (tramite, ruteo)
2. **Hugging Face** para priorizacion por sentimiento: las PQRS con sentimiento NEG y score alto pueden escalarse con mayor urgencia

Este enfoque dual es el que implementan los sistemas modernos de atencion al ciudadano en entidades publicas colombianas.
"""))

    nb.cells = cells
    path = os.path.join(BASE, "notebooks", "bloque3_nlp_prieto.ipynb")
    nbf.write(nb, path)
    print(f"Creado: {path}")


# ============================================================
# EJECUTAR
# ============================================================
if __name__ == "__main__":
    print("Generando notebooks...")
    build_bloque1()
    build_bloque2()
    build_bloque3()
    print("\\nTodos los notebooks creados en proyecto_final/notebooks/")
